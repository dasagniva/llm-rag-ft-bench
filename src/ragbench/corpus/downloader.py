"""Download SEC EDGAR 10-K filings and build the retrieval corpus manifest.

Usage (from scripts/build_corpus.py):
    downloader = EdgarDownloader(raw_dir=Path("data/raw/edgar"), email="you@example.com")
    paths = downloader.download_batch(tickers=["AAPL", "MSFT"], years=[2021, 2022, 2023])
    manifest = downloader.build_manifest(paths)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)

# Tickers used for the primary retrieval corpus (S&P 500 large-cap sample).
# Chosen for broad sector coverage and likely overlap with FinQA/TAT-QA source filings.
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "JPM",
    "BAC",
    "GS",
    "WFC",
    "C",
    "XOM",
    "CVX",
    "COP",
    "JNJ",
    "PFE",
    "UNH",
    "ABT",
    "WMT",
    "HD",
    "TGT",
]


class EdgarDownloader:
    def __init__(self, raw_dir: Path, email: str, company_name: str = "ragbench") -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._dl = Downloader(company_name, email, str(raw_dir))

    def download_batch(
        self,
        tickers: list[str],
        years: list[int],
        filing_type: str = "10-K",
    ) -> list[Path]:
        """Download filings for each (ticker, year) pair. Returns list of downloaded paths."""
        downloaded: list[Path] = []
        for ticker in tickers:
            try:
                self._dl.get(
                    filing_type,
                    ticker,
                    after=f"{min(years) - 1}-12-31",
                    before=f"{max(years) + 1}-01-01",
                )
                ticker_dir = self.raw_dir / "sec-edgar-filings" / ticker / filing_type
                if ticker_dir.exists():
                    downloaded.extend(ticker_dir.rglob("*.htm"))
                    downloaded.extend(ticker_dir.rglob("*.html"))
                    downloaded.extend(ticker_dir.rglob("*.txt"))
            except Exception:
                logger.warning("Failed to download %s %s — skipping", ticker, filing_type)
        return downloaded

    def build_manifest(self, paths: list[Path]) -> list[dict[str, str]]:
        """Return a list of manifest entries (path + SHA-256) for the given files."""
        entries: list[dict[str, str]] = []
        for p in sorted(paths):
            sha256 = _sha256(p)
            entries.append(
                {
                    "path": str(p.relative_to(self.raw_dir)),
                    "sha256": sha256,
                    "size_bytes": str(p.stat().st_size),
                }
            )
        return entries

    def save_manifest(self, manifest: list[dict[str, str]], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"files": manifest, "count": len(manifest)}, f, indent=2)
        logger.info("Manifest saved: %d files → %s", len(manifest), out_path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
