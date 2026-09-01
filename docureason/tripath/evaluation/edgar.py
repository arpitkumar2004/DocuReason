"""edgar.py — fetch real SEC filings to evaluate against.

Everything the harness could score before came from eight fixture files totalling
about 3 KB, plus one PDF, with hand-written questions whose "answers" were invented
(``benchmark_dataset.py``: ``"answer": "2.4M"`` against a table that does not contain
2.4M).  Audit §10 is blunt about the consequence: with a corpus that small and gold
labels that arbitrary, no retrieval number the system reports is falsifiable.  This
module fetches the real thing.

Why EDGAR: filings are public, free, stable at a permanent URL, and they are exactly
the document class the system claims to serve - multi-column prose, dense financial
tables, footnotes and scale qualifiers ("in thousands") all in one file.

Access rules, which this module follows rather than works around:

* SEC requires a descriptive ``User-Agent`` with contact information.  Set
  ``DOCUREASON_SEC_USER_AGENT``; there is no usable default, and requests are refused
  without one rather than sent with a fake identity.
* Fair-access asks for at most 10 requests/second.  Requests are serialized with a
  delay, so a corpus build is slow by design.
* Filings are cached on disk by accession number.  A second build re-reads them
  instead of re-fetching, so iterating on the query generator costs SEC nothing.

Provenance is written next to the documents: URL, accession number, form type, filing
date and a SHA-256 of the bytes.  Without it a corpus is unreproducible, and "we
evaluated on 24 filings" is not a checkable statement.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docureason.tripath.utils import get_logger

logger = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

#: Environment variable carrying the SEC-required contact identity.
USER_AGENT_ENV = "DOCUREASON_SEC_USER_AGENT"

#: Seconds between requests. SEC asks for <=10/s; this is deliberately well under.
REQUEST_DELAY_SEC = 0.25

#: A spread of large filers across sectors, so a corpus built from the default set is
#: not four companies with the same disclosure template. (name, CIK).
DEFAULT_COMPANIES: Tuple[Tuple[str, int], ...] = (
    ("Apple", 320193),
    ("Microsoft", 789019),
    ("JPMorgan Chase", 19617),
    ("Johnson & Johnson", 200406),
    ("Exxon Mobil", 34088),
    ("Walmart", 104169),
    ("Caterpillar", 18230),
    ("Pfizer", 78003),
    ("Coca-Cola", 21344),
    ("Boeing", 12927),
    ("Verizon", 732712),
    ("Procter & Gamble", 80424),
)


class EdgarAccessError(RuntimeError):
    """Raised when EDGAR cannot be reached or refuses the request."""


@dataclass
class FetchedFiling:
    """One filing on disk, with enough provenance to re-fetch or verify it."""

    company: str
    cik: int
    form: str
    filing_date: str
    accession: str
    document: str
    url: str
    path: str
    bytes: int
    sha256: str
    fetched_at: str
    from_cache: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def user_agent() -> str:
    """The configured SEC contact identity, or raise.

    Refusing is the point: sending a made-up identity to a government service to keep
    a build green is not a tradeoff worth making, and SEC blocks anonymous scrapers
    anyway - so a fake default would fail later and less clearly.
    """
    value = os.getenv(USER_AGENT_ENV, "").strip()
    if not value:
        raise EdgarAccessError(
            f"SEC requires a descriptive User-Agent with contact info. Set {USER_AGENT_ENV}, e.g.\n"
            f'  {USER_AGENT_ENV}="AcmeResearch corpus-builder (research@acme.example)"'
        )
    return value


def _get(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent(),
            "Accept-Encoding": "gzip, deflate",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                import gzip

                payload = gzip.decompress(payload)
            return payload
    except urllib.error.HTTPError as exc:
        raise EdgarAccessError(f"{url} -> HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EdgarAccessError(f"{url} -> {exc}") from exc


def list_filings(cik: int, form: str = "10-K", limit: int = 2) -> List[Dict[str, Any]]:
    """The most recent *limit* filings of type *form* for *cik*."""
    payload = json.loads(_get(SUBMISSIONS_URL.format(cik=cik)).decode("utf-8", errors="replace"))
    recent = (payload.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    dates = recent.get("filingDate") or []

    out: List[Dict[str, Any]] = []
    for index, form_type in enumerate(forms):
        if form_type != form:
            continue
        if index >= min(len(accessions), len(documents), len(dates)):
            break
        document = documents[index]
        # Filings whose primary document is not HTML would land in the rule-based
        # segmenter, producing a corpus entry with no tables and no reading order.
        if not str(document).lower().endswith((".htm", ".html")):
            continue
        out.append(
            {
                "accession": accessions[index],
                "document": document,
                "filing_date": dates[index],
                "form": form_type,
            }
        )
        if len(out) >= limit:
            break
    return out


def fetch_filings(
    dest: Path | str,
    companies: Sequence[Tuple[str, int]] = DEFAULT_COMPANIES,
    form: str = "10-K",
    per_company: int = 1,
    limit: Optional[int] = None,
    delay: float = REQUEST_DELAY_SEC,
) -> List[FetchedFiling]:
    """Download up to *limit* filings into *dest*, round-robin across *companies*.

    Round-robin rather than company-by-company: a partial build (a rate limit, a
    dropped connection) then still covers several issuers instead of exhausting one.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    user_agent()  # fail before any request if the identity is not configured

    wanted: List[Tuple[str, int, Dict[str, Any]]] = []
    per_company_lists: List[List[Tuple[str, int, Dict[str, Any]]]] = []
    for name, cik in companies:
        try:
            filings = list_filings(cik, form=form, limit=per_company)
        except EdgarAccessError as exc:
            logger.warning("EDGAR listing failed for %s (CIK %d): %s", name, cik, exc)
            continue
        per_company_lists.append([(name, cik, filing) for filing in filings])
        time.sleep(delay)

    for depth in range(per_company):
        for entries in per_company_lists:
            if depth < len(entries):
                wanted.append(entries[depth])
    if limit is not None:
        wanted = wanted[:limit]

    fetched: List[FetchedFiling] = []
    for name, cik, filing in wanted:
        accession = str(filing["accession"]).replace("-", "")
        url = ARCHIVE_URL.format(cik=cik, accession=accession, document=filing["document"])
        target = dest / f"{_slug(name)}_{filing['form'].replace('/', '-')}_{filing['filing_date']}.html"

        if target.exists() and target.stat().st_size > 0:
            body = target.read_bytes()
            from_cache = True
        else:
            try:
                body = _get(url)
            except EdgarAccessError as exc:
                logger.warning("EDGAR fetch failed for %s: %s", url, exc)
                continue
            target.write_bytes(body)
            from_cache = False
            time.sleep(delay)

        fetched.append(
            FetchedFiling(
                company=name,
                cik=cik,
                form=str(filing["form"]),
                filing_date=str(filing["filing_date"]),
                accession=str(filing["accession"]),
                document=str(filing["document"]),
                url=url,
                path=str(target),
                bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                from_cache=from_cache,
            )
        )
        logger.info(
            "%s %s %s (%.1f KB)%s",
            name,
            filing["form"],
            filing["filing_date"],
            len(body) / 1024.0,
            " [cached]" if from_cache else "",
        )

    _write_provenance(dest, fetched, form=form)
    return fetched


def _write_provenance(dest: Path, fetched: Sequence[FetchedFiling], form: str) -> Path:
    path = dest / "provenance.json"
    path.write_text(
        json.dumps(
            {
                "source": "SEC EDGAR",
                "form": form,
                "filing_count": len(fetched),
                "user_agent": user_agent(),
                "filings": [item.as_dict() for item in fetched],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _slug(text: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "_" for ch in text]
    return "".join(keep).strip("_").replace("__", "_")


__all__ = [
    "DEFAULT_COMPANIES",
    "EdgarAccessError",
    "FetchedFiling",
    "USER_AGENT_ENV",
    "fetch_filings",
    "list_filings",
    "user_agent",
]
