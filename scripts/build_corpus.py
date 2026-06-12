#!/usr/bin/env python
"""Download SEC EDGAR 10-K filings, clean, chunk, and write the corpus manifest.

Usage:
    uv run scripts/build_corpus.py --config configs/corpus.yaml --email you@example.com

The raw filings are saved to data/raw/edgar/ (.gitignored).
Committed artifact: data/corpus_manifest.json (file list + SHA-256 hashes).
Chunk parameters are taken from the config and recorded in DECISIONS.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragbench.corpus.chunker import chunk_documents
from ragbench.corpus.cleaner import clean_filing_text, clean_text_file
from ragbench.corpus.downloader import DEFAULT_TICKERS, EdgarDownloader
from ragbench.generation.config import CorpusConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the EDGAR retrieval corpus")
    p.add_argument("--config", default="configs/corpus.yaml")
    p.add_argument("--email", default="", help="Your email for SEC EDGAR identification")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip downloading; chunk and manifest existing files only",
    )
    return p.parse_args()


def load_corpus_config(path: str, email_override: str = "") -> CorpusConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = CorpusConfig(
        raw_dir=raw.get("raw_dir", "data/raw/edgar"),
        manifest_path=raw.get("manifest_path", "data/corpus_manifest.json"),
        chunk_size=raw.get("chunk_size", 400),
        chunk_overlap=raw.get("chunk_overlap", 40),
        tickers=raw.get("tickers", DEFAULT_TICKERS),
        years=raw.get("years", [2020, 2021, 2022, 2023]),
        edgar_email=email_override or raw.get("edgar_email", ""),
    )
    if not cfg.edgar_email:
        logger.error("edgar_email is required. Pass --email or set edgar_email in %s", path)
        sys.exit(1)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_corpus_config(args.config, args.email)
    raw_dir = Path(cfg.raw_dir)
    manifest_path = Path(cfg.manifest_path)

    if not args.dry_run:
        logger.info("Downloading filings for %d tickers, years %s", len(cfg.tickers), cfg.years)
        downloader = EdgarDownloader(raw_dir=raw_dir, email=cfg.edgar_email)
        paths = downloader.download_batch(tickers=cfg.tickers, years=cfg.years)
        logger.info("Downloaded %d filing files", len(paths))
        downloader.save_manifest(downloader.build_manifest(paths), manifest_path)
    else:
        paths = (
            list(raw_dir.rglob("*.htm"))
            + list(raw_dir.rglob("*.html"))
            + list(raw_dir.rglob("*.txt"))
        )
        logger.info("Dry-run: found %d existing files", len(paths))

    # Build chunked corpus for the vector index (Phase 2)
    chunks_path = raw_dir.parent / "chunks.jsonl"
    logger.info(
        "Chunking with chunk_size=%d, overlap=%d → %s",
        cfg.chunk_size,
        cfg.chunk_overlap,
        chunks_path,
    )
    documents: list[dict[str, str]] = []
    for p in sorted(paths):
        content = p.read_text(errors="replace")
        if p.suffix in {".htm", ".html"}:
            text = clean_filing_text(content)
        else:
            text = clean_text_file(content)
        if text:
            # Derive a unique ID from the path: sec-edgar-filings/{TICKER}/10-K/{ACCESSION}/...
            # → "{TICKER}_{ACCESSION}", e.g. "AMZN_0001018724-23-000004"
            parts = p.relative_to(raw_dir).parts
            doc_id = f"{parts[1]}_{parts[3]}" if len(parts) >= 4 else p.stem
            documents.append({"id": doc_id, "text": text, "source_path": str(p)})

    chunks = chunk_documents(documents, cfg.chunk_size, cfg.chunk_overlap)
    logger.info("Total chunks: %d", len(chunks))

    with open(chunks_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    logger.info("Chunks written to %s", chunks_path)


if __name__ == "__main__":
    main()
