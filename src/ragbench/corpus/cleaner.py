"""Extract and clean plain text from SEC EDGAR 10-K HTML filings."""

from __future__ import annotations

import re
import unicodedata


def clean_filing_text(html: str) -> str:
    """Extract readable text from a 10-K HTML filing.

    Steps:
    1. Parse HTML and strip scripts/styles.
    2. Extract visible text with BeautifulSoup.
    3. Normalise whitespace and Unicode characters.
    4. Remove EDGAR submission header boilerplate.
    """
    from bs4 import BeautifulSoup  # lazy import — bs4 has Python 3.13 typing issues at module level

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # Normalise Unicode (smart quotes, dashes, etc.) to ASCII-compatible form
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Drop the EDGAR submission header that precedes the actual filing body
    # (everything up to the first occurrence of "FORM 10-K" or "ANNUAL REPORT")
    match = re.search(r"\bFORM\s+10-K\b|\bANNUAL\s+REPORT\b", text, re.IGNORECASE)
    if match:
        text = text[match.start() :]

    return text


def clean_text_file(raw_text: str) -> str:
    """Minimal cleaning for pre-extracted plain-text filings."""
    text = unicodedata.normalize("NFKD", raw_text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()
