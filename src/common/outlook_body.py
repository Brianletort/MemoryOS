"""Extract full email body text from .olk15Message files using macOS mdimport.

Microsoft Outlook for Mac stores email bodies in a proprietary binary format.
The Outlook Spotlight importer plugin (bundled with Outlook.app) knows how to
parse these files and extract the plain-text body.  We invoke it via:

    mdimport -t -d3 <path-to-olk15Message>

and parse the ``kMDItemTextContent`` field from the output.

Performance: ~63ms per email (measured).  For backfill of 65K emails at 8
parallel workers, total time is ~9 minutes.
"""

from __future__ import annotations

import logging
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("memoryos.outlook_body")


def extract_body(message_path: Path | str) -> str | None:
    """Extract the full plain-text body from an .olk15Message file.

    Returns the body text, or None if extraction fails.
    """
    path = Path(message_path)
    if not path.exists():
        logger.warning("Message file not found: %s", path)
        return None

    try:
        result = subprocess.run(
            ["mdimport", "-t", "-d3", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning("mdimport timed out for %s", path)
        return None
    except OSError as exc:
        logger.error("Failed to run mdimport: %s", exc)
        return None

    # mdimport writes the attribute dict to stderr (the -d flag sends diagnostics there)
    output = result.stderr if result.stderr else result.stdout
    return _parse_text_content(output)


def _parse_text_content(mdimport_output: str) -> str | None:
    """Parse kMDItemTextContent from mdimport -d3 output."""
    # The field looks like: kMDItemTextContent = "...text...";
    # It may span many lines, ending with ";
    match = re.search(
        r'kMDItemTextContent\s*=\s*"(.*?)"\s*;',
        mdimport_output,
        re.DOTALL,
    )
    if not match:
        return None

    text = match.group(1)
    # Unescape unicode sequences like \U2019 -> '
    text = _unescape_unicode(text)
    # Remove any stray Unicode surrogates that can't be encoded to UTF-8
    # surrogatepass lets us encode surrogates, then replace on decode
    text = text.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="replace")
    # Normalize whitespace (mdimport collapses paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or None


def _unescape_unicode(text: str) -> str:
    """Replace \\UXXXX sequences (Apple plist style) with actual characters."""
    def _replace(m: re.Match) -> str:
        codepoint = int(m.group(1), 16)
        try:
            return chr(codepoint)
        except (ValueError, OverflowError):
            return m.group(0)

    return re.compile(r"\\U([0-9A-Fa-f]{4,6})").sub(_replace, text)


def extract_bodies_batch(
    message_paths: list[Path],
    max_workers: int = 8,
) -> dict[Path, str | None]:
    """Extract bodies from multiple .olk15Message files in parallel.

    Returns a dict mapping each path to its extracted body text (or None).
    """
    results: dict[Path, str | None] = {}

    if not message_paths:
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {
            pool.submit(extract_body, p): p for p in message_paths
        }
        done_count = 0
        total = len(message_paths)
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            done_count += 1
            try:
                results[path] = future.result()
            except Exception as exc:
                logger.error("Body extraction failed for %s: %s", path, exc)
                results[path] = None
            if done_count % 1000 == 0:
                logger.info("Extracted %d/%d email bodies", done_count, total)

    return results
