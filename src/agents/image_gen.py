"""Image generation service using OpenAI gpt-image-1.5.

Generates hero images, thumbnails, infographics, and concept diagrams for
MemoryOS skill reports. Caches images to avoid regeneration.

gpt-image-1.5 features used:
  - Text rendering (crisp titles/labels embedded in images)
  - Transparent backgrounds (icons, badges)
  - Complex structured visuals (infographics, architecture diagrams)
  - Quality tiers (low/medium/high for cost control)
  - Returns b64_json only -- decoded and saved to disk
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("memoryos.agents.imagegen")

_MODEL = "gpt-image-1.5"

_STYLE_PROMPTS: dict[str, str] = {
    "editorial": (
        "Dark-themed professional editorial illustration. Rich deep blues, "
        "purples, and teals with glowing accent highlights. Modern tech aesthetic "
        "with clean geometric shapes. Suitable for an executive intelligence briefing. "
        "No watermarks or borders."
    ),
    "infographic": (
        "Dark-themed data infographic with clean modern design. Use deep navy "
        "background with bright accent colors for data points. Include structured "
        "layout with clear visual hierarchy. Professional business intelligence style. "
        "No watermarks."
    ),
    "diagram": (
        "Technical architecture diagram on dark background. Clean lines, labeled "
        "components, directional arrows showing data flow. Modern flat design with "
        "glowing neon accents on dark surface. Professional and readable."
    ),
    "calendar": (
        "Modern calendar/planner visualization on dark background. Clean grid layout "
        "with color-coded blocks. Professional planning tool aesthetic with subtle "
        "gradients and rounded corners."
    ),
    "thumbnail": (
        "Compact, visually striking thumbnail image. Dark background with a single "
        "bold visual element. Clean, minimal, high contrast. Suitable for a news card."
    ),
}


def _get_client() -> Any:
    """Lazy-init OpenAI client."""
    from openai import OpenAI
    return OpenAI()


def _cache_dir(skill_name: str, date_str: str) -> Path:
    """Return (and create) the image cache directory for a skill+date."""
    from src.agents.skill_runner import _get_vault_path
    try:
        vault = _get_vault_path()
    except Exception:
        vault = Path.home() / "Documents" / "Obsidian" / "MyVault"
    reports_dir = vault / "90_reports" / skill_name / "images" / date_str
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _image_hash(topic: str, summary: str, style: str) -> str:
    """Deterministic short hash for cache key."""
    content = f"{topic}|{summary[:200]}|{style}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _build_prompt(
    topic_name: str,
    summary: str,
    style: str = "editorial",
    overlay_text: str | None = None,
) -> str:
    """Build an image generation prompt combining style + topic + summary."""
    style_desc = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["editorial"])

    parts = [style_desc]

    if overlay_text:
        parts.append(
            f'Include the text "{overlay_text}" prominently rendered in a clean, '
            f"modern sans-serif font. The text should be crisp and clearly legible."
        )

    parts.append(
        f"Subject: {topic_name}. "
        f"Context: {summary[:300]}"
    )

    return " ".join(parts)


def generate_image(
    topic_name: str,
    summary: str,
    skill_name: str = "news-pulse",
    date_str: str | None = None,
    style: str = "editorial",
    quality: str = "medium",
    size: str = "1536x1024",
    overlay_text: str | None = None,
    background: str = "opaque",
    force: bool = False,
) -> Path | None:
    """Generate an image using gpt-image-1.5 with caching.

    Parameters
    ----------
    topic_name:
        Human-readable topic (used in prompt and cache key).
    summary:
        Brief context for the image content.
    skill_name:
        Which skill this image belongs to (for cache path).
    date_str:
        Date string for cache directory (defaults to today).
    style:
        One of: editorial, infographic, diagram, calendar, thumbnail.
    quality:
        low (~$0.009), medium (~$0.034), high (~$0.133) per 1024x1024.
    size:
        Image dimensions: 1024x1024, 1024x1536, 1536x1024.
    overlay_text:
        Text to render directly on the image (gpt-image-1.5 excels at this).
    background:
        "opaque" (default) or "transparent" for overlay elements.
    force:
        Regenerate even if cached.

    Returns
    -------
    Path to the saved PNG file, or None on failure.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, skipping image generation")
        return None

    from datetime import datetime
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    cache = _cache_dir(skill_name, date_str)
    img_hash = _image_hash(topic_name, summary, style)
    filename = f"{img_hash}_{style}.png"
    out_path = cache / filename

    if out_path.exists() and not force:
        logger.info("Image cache hit: %s", out_path)
        return out_path

    prompt = _build_prompt(topic_name, summary, style, overlay_text)

    try:
        client = _get_client()
        result = client.images.generate(
            model=_MODEL,
            prompt=prompt,
            size=size,
            quality=quality,
            output_format="png",
            background=background,
        )
        image_b64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)
        out_path.write_bytes(image_bytes)
        logger.info("Generated image: %s (%d bytes)", out_path, len(image_bytes))
        return out_path
    except Exception as exc:
        logger.error("Image generation failed for '%s': %s", topic_name, exc)
        return None


def generate_topic_hero(
    topic_name: str,
    summary: str,
    skill_name: str = "news-pulse",
    date_str: str | None = None,
) -> Path | None:
    """Generate a hero image for a news topic section."""
    return generate_image(
        topic_name=topic_name,
        summary=summary,
        skill_name=skill_name,
        date_str=date_str,
        style="editorial",
        quality="medium",
        size="1536x1024",
        overlay_text=topic_name,
    )


def generate_thumbnail(
    topic_name: str,
    article_title: str,
    skill_name: str = "news-pulse",
    date_str: str | None = None,
) -> Path | None:
    """Generate a small thumbnail for an article card."""
    return generate_image(
        topic_name=topic_name,
        summary=article_title,
        skill_name=skill_name,
        date_str=date_str,
        style="thumbnail",
        quality="low",
        size="1024x1024",
    )


def generate_infographic(
    title: str,
    data_summary: str,
    skill_name: str = "weekly-status",
    date_str: str | None = None,
) -> Path | None:
    """Generate an infographic-style image for status reports."""
    return generate_image(
        topic_name=title,
        summary=data_summary,
        skill_name=skill_name,
        date_str=date_str,
        style="infographic",
        quality="high",
        size="1536x1024",
        overlay_text=title,
    )


def generate_calendar_visual(
    week_summary: str,
    skill_name: str = "plan-my-week",
    date_str: str | None = None,
) -> Path | None:
    """Generate a visual calendar/planner image."""
    return generate_image(
        topic_name="Weekly Plan",
        summary=week_summary,
        skill_name=skill_name,
        date_str=date_str,
        style="calendar",
        quality="medium",
        size="1536x1024",
        overlay_text="Week at a Glance",
    )


def generate_diagram(
    concept: str,
    description: str,
    skill_name: str = "news-pulse",
    date_str: str | None = None,
) -> Path | None:
    """Generate a technical architecture/concept diagram."""
    return generate_image(
        topic_name=concept,
        summary=description,
        skill_name=skill_name,
        date_str=date_str,
        style="diagram",
        quality="medium",
        size="1536x1024",
        overlay_text=concept,
    )


def generate_dashboard_infographic(
    skill_name: str,
    title: str,
    metrics_summary: str,
    date_str: str | None = None,
) -> Path | None:
    """Generate a dashboard-style infographic for any skill report.

    Produces a dark-themed visual with key metrics suitable for email headers
    and report hero images.
    """
    return generate_image(
        topic_name=title,
        summary=metrics_summary,
        skill_name=skill_name,
        date_str=date_str,
        style="infographic",
        quality="medium",
        size="1536x1024",
        overlay_text=title,
    )


def image_path_to_api_url(image_path: Path, skill_name: str) -> str:
    """Convert a local image path to a dashboard API URL.

    The dashboard serves images via /api/agents/images/{skill}/{date}/{filename}.
    """
    date_dir = image_path.parent.name
    filename = image_path.name
    return f"/api/agents/images/{skill_name}/{date_dir}/{filename}"
