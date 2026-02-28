"""PPTX generation wrapper for in-chat slide building.

Accepts a structured slide spec (JSON) and produces a .pptx file
using the existing pptx_helpers library.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.chat.pptx")

HELPERS_DIR = Path.home() / ".cursor" / "skills" / "pptx-builder" / "scripts"
GENERATED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "generated"


def _ensure_helpers() -> None:
    """Add pptx_helpers to sys.path if not already present."""
    helpers_str = str(HELPERS_DIR)
    if helpers_str not in sys.path:
        sys.path.insert(0, helpers_str)


def build_slides(spec: dict[str, Any]) -> dict[str, Any]:
    """Build a .pptx file from a structured spec.

    Spec format::

        {
            "title": "Deck Title",
            "subtitle": "Optional subtitle",
            "author": "Author Name",
            "theme": "light" | "dark",
            "footer": "Footer text",
            "slides": [
                {
                    "type": "title",
                    "title": "Slide Title",
                    "subtitle": "Slide Subtitle"
                },
                {
                    "type": "header",
                    "title": "Section Header"
                },
                {
                    "type": "content",
                    "title": "Slide Title",
                    "bullets": ["Point 1", "Point 2", "Point 3"]
                },
                {
                    "type": "metrics",
                    "title": "Key Metrics",
                    "metrics": [["$1.2M", "Revenue"], ["42%", "Growth"]]
                },
                {
                    "type": "comparison",
                    "title": "Comparison",
                    "headers": ["Feature", "Option A", "Option B"],
                    "rows": [["Speed", "Fast", "Slow"], ["Cost", "$10", "$50"]]
                },
                {
                    "type": "takeaways",
                    "title": "Key Takeaways",
                    "items": ["Takeaway 1", "Takeaway 2", "Takeaway 3"]
                }
            ]
        }

    Returns dict with ``ok``, ``filename``, ``path``, and ``slide_count``.
    """
    _ensure_helpers()

    try:
        from pptx_helpers import (
            create_presentation,
            add_title_slide,
            add_header_slide,
            add_footer,
            add_text_block,
            add_metric_boxes,
            add_comparison_table,
            add_numbered_takeaways,
            Inches,
            Pt,
        )
    except ImportError as exc:
        return {"ok": False, "error": f"pptx_helpers not available: {exc}"}

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    theme = spec.get("theme", "light")
    prs, colors = create_presentation(theme=theme)

    slides_spec = spec.get("slides", [])
    footer_text = spec.get("footer", "")

    for slide_spec in slides_spec:
        slide_type = slide_spec.get("type", "content")

        if slide_type == "title":
            add_title_slide(
                prs,
                title=slide_spec.get("title", ""),
                subtitle=slide_spec.get("subtitle", ""),
                author=spec.get("author", ""),
                role=slide_spec.get("role", ""),
                org=slide_spec.get("org", ""),
                date=slide_spec.get("date", ""),
                colors=colors,
            )

        elif slide_type == "header":
            slide = add_header_slide(prs, slide_spec.get("title", ""), colors)
            if footer_text:
                add_footer(slide, footer_text, colors)

        elif slide_type == "content":
            slide = add_header_slide(prs, slide_spec.get("title", ""), colors)
            bullets = slide_spec.get("bullets", [])
            if bullets:
                lines = [
                    {"text": b, "size": Pt(16), "bold": False, "color_key": "text_dark"}
                    for b in bullets
                ]
                add_text_block(
                    slide,
                    left=Inches(0.8),
                    top=Inches(1.8),
                    width=Inches(11.7),
                    height=Inches(5.0),
                    lines=lines,
                    colors=colors,
                )
            if footer_text:
                add_footer(slide, footer_text, colors)

        elif slide_type == "metrics":
            slide = add_header_slide(prs, slide_spec.get("title", ""), colors)
            metrics = slide_spec.get("metrics", [])
            if metrics:
                metric_tuples = [(m[0], m[1]) for m in metrics if len(m) >= 2]
                add_metric_boxes(slide, metric_tuples, Inches(2.5), colors)
            if footer_text:
                add_footer(slide, footer_text, colors)

        elif slide_type == "comparison":
            slide = add_header_slide(prs, slide_spec.get("title", ""), colors)
            headers = slide_spec.get("headers", [])
            rows = slide_spec.get("rows", [])
            if headers and rows:
                add_comparison_table(
                    slide, headers, rows,
                    left=Inches(0.8), top=Inches(1.8),
                    w=Inches(11.7), colors=colors,
                )
            if footer_text:
                add_footer(slide, footer_text, colors)

        elif slide_type == "takeaways":
            slide = add_header_slide(prs, slide_spec.get("title", "Key Takeaways"), colors)
            items = slide_spec.get("items", [])
            if items:
                add_numbered_takeaways(
                    slide, items,
                    left=Inches(0.8), top=Inches(1.8),
                    w=Inches(11.7), colors=colors,
                )
            if footer_text:
                add_footer(slide, footer_text, colors)

        else:
            slide = add_header_slide(prs, slide_spec.get("title", slide_type), colors)
            if footer_text:
                add_footer(slide, footer_text, colors)

    file_id = uuid.uuid4().hex[:10]
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in spec.get("title", "slides"))
    safe_title = safe_title.strip().replace(" ", "_")[:40] or "slides"
    filename = f"{safe_title}_{file_id}.pptx"
    out_path = GENERATED_DIR / filename

    prs.save(str(out_path))
    logger.info("Generated PPTX: %s (%d slides)", out_path, len(slides_spec))

    return {
        "ok": True,
        "filename": filename,
        "path": str(out_path),
        "slide_count": len(slides_spec),
    }
