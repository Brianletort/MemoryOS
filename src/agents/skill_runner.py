#!/usr/bin/env python3
"""Headless skill runner -- executes MemoryOS skills without Cursor.

Reads a SKILL.md + manifest.yaml, gathers vault data, calls the configured
LLM provider, and optionally emails the result.

Supports dual-format output: structured JSON (for rich dashboard views)
alongside Markdown (for email/Obsidian vault).

Usage::

    python3 -m src.agents.skill_runner --skill morning-brief --email
    python3 -m src.agents.skill_runner --skill news-pulse --dry-run
    python3 -m src.agents.skill_runner --skill plan-my-week --provider ollama --model ollama/llama4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents import llm_provider, emailer
from src.common.config import load_config

logger = logging.getLogger("memoryos.agents.runner")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_DIR / "config" / "config.yaml"
ENV_LOCAL = REPO_DIR / ".env.local"
DEFAULT_SKILLS_DIR = Path.home() / ".cursor" / "skills"


def _load_env() -> None:
    """Load .env.local into os.environ if present."""
    if ENV_LOCAL.is_file():
        for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _get_vault_path() -> Path:
    """Resolve the Obsidian vault path from config."""
    try:
        cfg = load_config(CONFIG_PATH)
        return Path(cfg["obsidian_vault"]).expanduser()
    except Exception:
        return Path.home() / "Documents" / "Obsidian" / "MyVault"


def _get_skills_dir() -> Path:
    """Resolve the skills directory from config."""
    try:
        cfg = load_config(CONFIG_PATH)
        agents = cfg.get("agents", {})
        sd = agents.get("skills_dir", str(DEFAULT_SKILLS_DIR))
        return Path(sd).expanduser()
    except Exception:
        return DEFAULT_SKILLS_DIR


def _compute_week_dates() -> list[str]:
    """Return YYYY/MM/DD paths for each day in the current Sun-Sat week."""
    today = datetime.now()
    sunday = today - timedelta(days=today.weekday() + 1)
    if today.weekday() == 6:
        sunday = today
    return [(sunday + timedelta(days=i)).strftime("%Y/%m/%d") for i in range(7)]


def _compute_yesterday() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y/%m/%d")


def _compute_today() -> str:
    return datetime.now().strftime("%Y/%m/%d")


def _load_core_context(vault: Path) -> str:
    """Load _context/core.md if it exists. Truncate if too large."""
    core_path = vault / "_context" / "core.md"
    if not core_path.is_file():
        return ""
    try:
        content = core_path.read_text(encoding="utf-8", errors="replace")
        if len(content) > 30000:
            content = content[:30000] + "\n\n[... truncated for token limits ...]\n"
        return content
    except OSError:
        return ""


def _load_shared_context(skills_dir: Path) -> str:
    """Load _shared/context.yaml and format as prompt context."""
    ctx_file = skills_dir / "_shared" / "context.yaml"
    if not ctx_file.is_file():
        return ""

    try:
        data = yaml.safe_load(ctx_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""

    parts: list[str] = []

    mc = data.get("my_context", {})
    if mc:
        parts.append("=== MY CONTEXT ===")
        if mc.get("name"):
            parts.append(f"Name: {mc['name']}")
        if mc.get("title"):
            parts.append(f"Title: {mc['title']}")
        if mc.get("company"):
            parts.append(f"Company: {mc['company']}")
        if mc.get("reports_to"):
            parts.append(f"Reports to: {mc['reports_to']}")
        rels = mc.get("relationships", [])
        if rels:
            parts.append("\nKey relationships:")
            for r in rels:
                parts.append(f"- {r.get('name', '?')} ({r.get('role', '?')}): {r.get('relationship', '')}")
        parts.append("")

    gc = data.get("global_context", {})
    if gc:
        parts.append("=== GLOBAL CONTEXT ===")
        if gc.get("company"):
            parts.append(f"Company: {gc['company']}")
        if gc.get("industry"):
            parts.append(f"Industry: {gc['industry']}")
        comps = gc.get("competitors", [])
        if comps:
            parts.append(f"Competitors: {', '.join(comps)}")
        pris = gc.get("strategic_priorities", [])
        if pris:
            parts.append("Strategic priorities:")
            for p in pris:
                parts.append(f"- {p}")
        tools = gc.get("tools", [])
        platforms = gc.get("platforms", [])
        if tools or platforms:
            parts.append(f"Tools/platforms: {', '.join(tools + platforms)}")
        parts.append("")

    return "\n".join(parts)


def _read_vault_file(
    vault: Path, rel_path: str, max_chars: int = 50_000,
) -> str | None:
    """Read a file from the vault, returning None if it doesn't exist.

    Large files (activity logs, chat dumps) are truncated to *max_chars*
    to stay within LLM context-window budgets.
    """
    full = vault / rel_path
    if full.is_file():
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... truncated for token limits ...]\n"
            return content
        except OSError:
            return None
    return None


def _run_cli_recent(source_type: str | None = None, limit: int = 10, hours: int = 72) -> str:
    """Run a memory CLI recent query and return the output."""
    cmd = [
        sys.executable, "-m", "src.memory.cli",
        "recent", "--hours", str(hours), "--limit", str(limit),
    ]
    if source_type:
        cmd.extend(["--type", source_type])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(REPO_DIR),
            env={**os.environ, "PYTHONPATH": str(REPO_DIR)},
        )
        return result.stdout[:4000] if result.stdout else "(no results)"
    except Exception as exc:
        return f"(recent query error: {exc})"


def _run_cli_search(query: str, source_type: str | None = None, limit: int = 10) -> str:
    """Run a memory CLI search and return the output."""
    cmd = [
        sys.executable, "-m", "src.memory.cli",
        "search", query, "--limit", str(limit),
    ]
    if source_type:
        cmd.extend(["--type", source_type])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(REPO_DIR),
            env={**os.environ, "PYTHONPATH": str(REPO_DIR)},
        )
        return result.stdout[:4000] if result.stdout else "(no results)"
    except Exception as exc:
        return f"(search error: {exc})"


def _gather_data(manifest: dict[str, Any], vault: Path) -> str:
    """Gather all vault data specified in the manifest."""
    parts: list[str] = []
    ds = manifest.get("data_sources", {})

    for ctx_file in ds.get("context_files", []):
        content = _read_vault_file(vault, ctx_file)
        if content:
            parts.append(f"=== {ctx_file} ===\n{content}\n")

    vault_files = ds.get("vault_files", {})
    if isinstance(vault_files, dict):
        pattern = vault_files.get("pattern", "")
        if pattern and "{week_dates}" in pattern:
            for date_path in _compute_week_dates():
                resolved = pattern.replace("{week_dates}", date_path)
                content = _read_vault_file(vault, resolved)
                if content:
                    parts.append(f"=== {resolved} ===\n{content}\n")
        elif pattern and "{yesterday}" in pattern:
            resolved = pattern.replace("{yesterday}", _compute_yesterday())
            content = _read_vault_file(vault, resolved)
            if content:
                parts.append(f"=== {resolved} ===\n{content}\n")
        elif pattern and "{today}" in pattern:
            resolved = pattern.replace("{today}", _compute_today())
            content = _read_vault_file(vault, resolved)
            if content:
                parts.append(f"=== {resolved} ===\n{content}\n")

        for extra in vault_files.get("extra", []):
            resolved = extra.replace("{yesterday}", _compute_yesterday())
            resolved = resolved.replace("{today}", _compute_today())
            content = _read_vault_file(vault, resolved)
            if content:
                parts.append(f"=== {resolved} ===\n{content}\n")

        for extra_pat in vault_files.get("extra_patterns", []):
            if "{week_dates}" in extra_pat:
                for date_path in _compute_week_dates():
                    resolved = extra_pat.replace("{week_dates}", date_path)
                    content = _read_vault_file(vault, resolved)
                    if content:
                        parts.append(f"=== {resolved} ===\n{content[:3000]}\n")

    searches = ds.get("cli_searches", [])
    if searches:
        search_jobs: list[tuple[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(len(searches), 8)) as pool:
            for search in searches:
                query = search.get("query", "")
                stype = search.get("type")
                limit = search.get("limit", 10)
                recent_hours = search.get("recent_hours")

                if recent_hours and not query:
                    label = f"CLI recent: type={stype}, hours={recent_hours}"
                    fut = pool.submit(_run_cli_recent, stype, limit, recent_hours)
                    search_jobs.append((label, fut))
                elif query:
                    label = f"CLI search: '{query}' (type={stype})"
                    fut = pool.submit(_run_cli_search, query, stype, limit)
                    search_jobs.append((label, fut))

        for label, fut in search_jobs:
            try:
                result = fut.result(timeout=60)
            except Exception as exc:
                result = f"(search error: {exc})"
            parts.append(f"=== {label} ===\n{result}\n")

    return "\n".join(parts)


def _gather_web_data(skill_dir: Path) -> str:
    """Gather web search data for news-pulse style skills.

    Returns the personal context, topic configs with their context fields,
    and web search results so the LLM can produce editorially-framed output.
    """
    topics_file = skill_dir / "topics.yaml"
    if not topics_file.is_file():
        return "(no topics.yaml found)"

    try:
        topics_cfg = yaml.safe_load(topics_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return "(failed to parse topics.yaml)"

    parts: list[str] = []

    personal_ctx = topics_cfg.get("personal_context", "")
    if personal_ctx:
        parts.append(f"=== Personal Context ===\n{personal_ctx}\n")

    topics = topics_cfg.get("topics", [])
    if topics:
        parts.append("=== Topic Configuration ===")
        for t in topics:
            parts.append(f"- {t.get('name', '?')} (depth: {t.get('depth', 'brief')})")
            ctx = t.get("context", "")
            if ctx:
                parts.append(f"  Why it matters: {ctx}")
        parts.append("")

    vault_signals_cfg = topics_cfg.get("vault_signals", {})
    if vault_signals_cfg.get("enabled"):
        parts.append("=== Vault Signal Detection ===")
        parts.append(f"Auto-detect up to {vault_signals_cfg.get('max_auto_topics', 3)} "
                      f"emerging topics from vault data (lookback: "
                      f"{vault_signals_cfg.get('lookback_hours', 72)}h).")
        parts.append("Analyze the vault data provided above for recurring names, projects, ")
        parts.append("companies, or themes that appear 2+ times but are NOT already covered ")
        parts.append("by the static topics. Generate web searches for those auto-detected topics.\n")

    if not topics:
        return "\n".join(parts) + "\n(no topics configured)"

    from src.agents.web_search import search_all_topics
    results = search_all_topics(topics, use_ai_grounding=True)

    for topic_name, articles in results.items():
        topic_cfg = next((t for t in topics if t.get("name") == topic_name), {})
        ctx = topic_cfg.get("context", "")
        parts.append(f"=== Web search: {topic_name} ===")
        if ctx:
            parts.append(f"[Topic context: {ctx}]")
        if not articles:
            parts.append("(no results)")
        for a in articles:
            parts.append(f"- {a['title']}")
            if a.get("snippet"):
                parts.append(f"  {a['snippet'][:300]}")
            if a.get("url"):
                parts.append(f"  URL: {a['url']}")
            if a.get("date"):
                parts.append(f"  Date: {a['date']}")
            if a.get("thumbnail_url"):
                parts.append(f"  Thumbnail: {a['thumbnail_url']}")
            if a.get("ai_summary"):
                parts.append(f"  AI Summary: {a['ai_summary'][:400]}")
        parts.append("")

    return "\n".join(parts)


def _gather_web_data_structured(skill_dir: Path) -> dict[str, Any]:
    """Gather web search data as structured dict for JSON reports."""
    topics_file = skill_dir / "topics.yaml"
    if not topics_file.is_file():
        return {"topics": [], "personal_context": ""}

    try:
        topics_cfg = yaml.safe_load(topics_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"topics": [], "personal_context": ""}

    topics = topics_cfg.get("topics", [])
    if not topics:
        return {"topics": [], "personal_context": topics_cfg.get("personal_context", "")}

    from src.agents.web_search import search_all_topics
    results = search_all_topics(topics, use_ai_grounding=True)

    structured_topics: list[dict[str, Any]] = []
    for topic_cfg in topics:
        name = topic_cfg.get("name", "Unknown")
        articles = results.get(name, [])
        structured_topics.append({
            "name": name,
            "depth": topic_cfg.get("depth", "brief"),
            "context": topic_cfg.get("context", ""),
            "keywords": topic_cfg.get("keywords", []),
            "articles": articles,
        })

    return {
        "personal_context": topics_cfg.get("personal_context", ""),
        "topics": structured_topics,
        "vault_signals": topics_cfg.get("vault_signals", {}),
    }


def _load_prior_report(vault: Path, skill_name: str) -> dict[str, Any] | None:
    """Load the most recent prior JSON report for trend comparison.

    Looks for the newest .json file in the reports directory that is older
    than today's date, enabling delta / trend detection across runs.
    """
    try:
        cfg = load_config(CONFIG_PATH)
        reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    except Exception:
        reports_dir_name = "90_reports"

    reports_dir = vault / reports_dir_name / skill_name
    if not reports_dir.is_dir():
        return None

    today_str = datetime.now().strftime("%Y-%m-%d")
    json_files = sorted(reports_dir.glob("*.json"), reverse=True)
    for jf in json_files:
        if jf.stem != today_str:
            try:
                return json.loads(jf.read_text(encoding="utf-8", errors="replace"))
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _run_analysis_pass(
    report_md: str,
    prior_report: dict[str, Any] | None,
    skill_name: str,
    today_str: str,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> str:
    """Run an LLM analysis pass that produces executive insights and trend data.

    This intermediate step enriches the final JSON with:
    - Trend detection vs prior report
    - Risk scoring and priority ranking
    - "So What?" executive insight per section
    - Predicted next actions and recommended focus
    """
    system_prompt = (
        "You are an elite executive intelligence analyst. Analyze the report below "
        "and produce a concise ANALYSIS ADDENDUM in Markdown. This addendum will be "
        "merged into the structured JSON output.\n\n"
        f"Today is {today_str}.\n\n"
        "Your analysis MUST include these sections:\n"
        "## Executive Insight\n"
        "One punchy sentence summarizing the most important takeaway.\n\n"
        "## Trend Analysis\n"
        "Compare to the prior report (if provided). For each major metric or "
        "section, note: improving / stable / declining and by how much.\n\n"
        "## Risk Assessment\n"
        "Rank the top 3 risks. For each: risk title, severity (critical/high/medium/low), "
        "likelihood, impact, and recommended mitigation.\n\n"
        "## Recommended Focus\n"
        "The single most impactful action to take today/this week.\n\n"
        "## Predictions\n"
        "2-3 things likely to happen next based on current trajectory.\n\n"
        "Be specific, quantitative where possible, and brutally honest. No fluff."
    )

    user_parts = [f"=== CURRENT REPORT ({skill_name}) ===\n{report_md}\n"]
    if prior_report:
        prior_summary = json.dumps(prior_report, indent=2, default=str)
        if len(prior_summary) > 30000:
            prior_summary = prior_summary[:30000] + "\n... [truncated]"
        user_parts.append(f"=== PRIOR REPORT (for trend comparison) ===\n{prior_summary}\n")
    else:
        user_parts.append("(No prior report available for comparison. Skip trend analysis.)\n")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    try:
        return llm_provider.complete(
            messages,
            skill_name=skill_name,
            provider_override=provider_override,
            model_override=model_override,
        )
    except Exception as exc:
        logger.warning("Analysis pass failed for %s: %s", skill_name, exc)
        return ""


def _save_report(vault: Path, skill_name: str, content: str) -> Path:
    """Save the generated Markdown report to the vault."""
    try:
        cfg = load_config(CONFIG_PATH)
        reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    except Exception:
        reports_dir_name = "90_reports"

    today = datetime.now()
    out_dir = vault / reports_dir_name / skill_name
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = today.strftime("%Y-%m-%d") + ".md"
    out_path = out_dir / filename
    out_path.write_text(content, encoding="utf-8")
    logger.info("Report saved: %s", out_path)
    return out_path


def _save_json_report(vault: Path, skill_name: str, data: dict[str, Any]) -> Path:
    """Save the structured JSON report alongside the Markdown report."""
    try:
        cfg = load_config(CONFIG_PATH)
        reports_dir_name = cfg.get("agents", {}).get("reports_dir", "90_reports")
    except Exception:
        reports_dir_name = "90_reports"

    today = datetime.now()
    out_dir = vault / reports_dir_name / skill_name
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = today.strftime("%Y-%m-%d") + ".json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("JSON report saved: %s", out_path)
    return out_path


def _generate_images_for_report(
    skill_name: str,
    json_data: dict[str, Any],
    date_str: str,
) -> dict[str, Any]:
    """Generate images for a structured report and update paths in the JSON.

    Dispatches to skill-specific image generation logic.
    """
    try:
        from src.agents.image_gen import (
            generate_topic_hero,
            generate_thumbnail,
            generate_infographic,
            generate_calendar_visual,
            generate_dashboard_infographic,
            image_path_to_api_url,
        )
    except ImportError:
        logger.warning("image_gen module not available, skipping image generation")
        return json_data

    if skill_name == "news-pulse":
        for topic in json_data.get("topics", []):
            topic_name = topic.get("name", "")
            articles = topic.get("articles", [])
            summary = topic.get("editorial_frame", "")
            if not summary and articles:
                summary = articles[0].get("summary", articles[0].get("title", ""))

            hero_path = generate_topic_hero(topic_name, summary, skill_name, date_str)
            if hero_path:
                topic["hero_image"] = image_path_to_api_url(hero_path, skill_name)

            for article in articles:
                thumb_path = generate_thumbnail(
                    topic_name, article.get("title", ""), skill_name, date_str
                )
                if thumb_path:
                    article["generated_thumbnail"] = image_path_to_api_url(thumb_path, skill_name)

        for signal in json_data.get("internal_signals", []):
            sig_title = signal.get("title", "")
            sig_type = signal.get("source_type", "internal")
            if sig_title:
                thumb_path = generate_thumbnail(
                    sig_type, sig_title, skill_name, date_str
                )
                if thumb_path:
                    signal["generated_thumbnail"] = image_path_to_api_url(thumb_path, skill_name)

    elif skill_name == "weekly-status":
        summary = json_data.get("summary", "Weekly Status Report")
        metrics = json_data.get("metrics", {})
        data_summary = (
            f"Meetings: {metrics.get('meetings_total', 0)}, "
            f"Emails: {metrics.get('emails_received', 0)} received / {metrics.get('emails_sent', 0)} sent, "
            f"Accomplishments: {metrics.get('accomplishments_count', 0)}"
        )
        hero_path = generate_infographic(
            "Weekly Status Overview", data_summary, skill_name, date_str
        )
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    elif skill_name == "plan-my-week":
        summary = json_data.get("executive_summary", "Weekly Plan")
        hero_path = generate_calendar_visual(summary, skill_name, date_str)
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    elif skill_name == "morning-brief":
        meeting_count = json_data.get("meeting_count", 0)
        email_count = json_data.get("priority_email_count", 0)
        day_score = json_data.get("day_score", 0)
        summary = f"Day Score: {day_score}/100. {meeting_count} meetings, {email_count} priority emails"
        hero_path = generate_infographic(
            "Morning Intelligence Brief", summary, skill_name, date_str
        )
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    elif skill_name == "commitment-tracker":
        total = json_data.get("total_open", 0)
        overdue = json_data.get("total_overdue", 0)
        health = json_data.get("health_score", 0)
        summary = f"Health: {health}/100. {total} open, {overdue} overdue commitments"
        hero_path = generate_dashboard_infographic(
            skill_name, "Commitment Dashboard", summary, date_str
        )
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    elif skill_name == "project-brief":
        ps = json_data.get("portfolio_summary", {})
        summary = f"{ps.get('total_projects', 0)} projects. Avg health: {ps.get('avg_health', 0)}/100"
        hero_path = generate_dashboard_infographic(
            skill_name, "Project Portfolio", summary, date_str
        )
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    elif skill_name == "focus-audit":
        score = json_data.get("productivity_score", 0)
        m = json_data.get("metrics", {})
        summary = f"Productivity: {score}/100. {m.get('deep_work_hours', 0):.1f}h deep work, {m.get('meeting_hours', 0):.1f}h meetings"
        hero_path = generate_dashboard_infographic(
            skill_name, "Focus & Productivity Audit", summary, date_str
        )
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    elif skill_name == "relationship-crm":
        count = json_data.get("active_contacts_count", 0)
        ns = json_data.get("network_summary", {})
        summary = f"{count} active contacts. {ns.get('inner_circle_count', 0)} inner circle"
        hero_path = generate_dashboard_infographic(
            skill_name, "Relationship Intelligence", summary, date_str
        )
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    elif skill_name == "team-manager":
        size = json_data.get("team_size", 0)
        health = json_data.get("team_health_score", 0)
        summary = f"Team of {size}. Health score: {health}/100"
        hero_path = generate_dashboard_infographic(
            skill_name, "Team Dashboard", summary, date_str
        )
        if hero_path:
            json_data["hero_image"] = image_path_to_api_url(hero_path, skill_name)

    return json_data


_JSON_OUTPUT_INSTRUCTION = """

IMPORTANT: You must output ONLY valid JSON. No markdown, no code fences, no explanation.
Output a single JSON object following the schema described in the skill instructions.
Ensure all strings are properly escaped. Do not truncate the output.
"""


def _extract_json_from_response(text: str) -> dict[str, Any] | None:
    """Try to parse JSON from an LLM response, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i in range(1, len(lines)):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


_CALENDAR_NOISE_PATTERNS = [
    re.compile(r"https?://teams\.microsoft\.com/\S+", re.IGNORECASE),
    re.compile(r"https?://aka\.ms/\S+", re.IGNORECASE),
    re.compile(r"Meeting\s+ID:\s*[\d\s]+", re.IGNORECASE),
    re.compile(r"Passcode:\s*\S+", re.IGNORECASE),
    re.compile(r"_{10,}"),
    re.compile(r"Need help\?.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"System reference.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Microsoft Teams meeting\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Join:?\s*$", re.MULTILINE),
    re.compile(r"\[​[^\]]*\]\s*", re.IGNORECASE),
]


def _preprocess_calendar_noise(vault_data: str) -> str:
    """Strip Teams join URLs/IDs/passcodes from calendar blocks in vault data."""
    blocks = vault_data.split("=== ")
    cleaned: list[str] = []
    for i, block in enumerate(blocks):
        if block.startswith("10_meetings/") and "calendar.md" in block:
            for pat in _CALENDAR_NOISE_PATTERNS:
                block = pat.sub("", block)
            block = re.sub(r"\n{3,}", "\n\n", block)
        cleaned.append(block)
    return "=== ".join(cleaned)


def _preprocess_activity_stats(
    vault_data: str, vault: Path,
) -> dict[str, Any] | None:
    """Parse daily activity files and return pre-computed stats.

    Replaces raw OCR data in vault_data with a compact summary so the
    LLM gets accurate app-usage numbers without context-window truncation.
    """
    from src.analyzers.activity_stats import parse_daily_activity

    activity_marker = "=== 85_activity/"
    activity_block = ""
    calendar_block = ""
    for block in vault_data.split("=== "):
        if block.startswith("85_activity/") and "daily.md" in block:
            activity_block = block
        elif block.startswith("10_meetings/") and "calendar.md" in block:
            calendar_block = block

    if not activity_block:
        logger.info("No activity file found in vault data for pre-processing")
        return None

    content_start = activity_block.find("\n")
    if content_start < 0:
        return None
    activity_text = activity_block[content_start + 1:]

    stats = parse_daily_activity(activity_text)
    if not stats.app_minutes:
        logger.info("Activity stats produced no app data")
        return None

    logger.info(
        "Pre-computed activity stats: %d active hours, %d apps, %d switches",
        stats.total_active_hours, len(stats.app_minutes), stats.total_context_switches,
    )
    return stats.to_metrics_dict()


def _inject_activity_summary(
    vault_data: str, precomputed: dict[str, Any],
) -> str:
    """Replace raw activity data in vault_data with a compact summary.

    Keeps calendar data intact while replacing the large activity block
    with the pre-computed summary that fits in ~2-3k chars.
    """
    from src.analyzers.activity_stats import parse_daily_activity

    parts: list[str] = []
    in_activity = False
    for line in vault_data.split("\n"):
        if line.startswith("=== 85_activity/") and "daily.md" in line:
            in_activity = True
            parts.append(line)
            parts.append("")
            # Re-parse just to get the compact summary
            activity_start = vault_data.find(line)
            next_section = vault_data.find("\n=== ", activity_start + len(line))
            if next_section < 0:
                raw_block = vault_data[activity_start + len(line):]
            else:
                raw_block = vault_data[activity_start + len(line):next_section]
            stats = parse_daily_activity(raw_block)
            parts.append(stats.to_compact_summary())
            parts.append("")
            parts.append("=== PRE-COMPUTED ACTIVITY STATS (JSON) ===")
            import json
            parts.append(json.dumps(precomputed, indent=2))
            parts.append("")
            continue
        if in_activity and line.startswith("=== "):
            in_activity = False
        if not in_activity:
            parts.append(line)
    return "\n".join(parts)


def _merge_precomputed_into_json(
    json_data: dict[str, Any], precomputed: dict[str, Any],
) -> dict[str, Any]:
    """Override LLM-estimated app metrics with deterministic pre-computed values."""
    if precomputed.get("app_breakdown"):
        json_data["app_breakdown"] = precomputed["app_breakdown"]
    if precomputed.get("top_apps"):
        json_data["top_apps"] = precomputed["top_apps"]

    metrics = json_data.get("metrics", {})
    if precomputed.get("total_active_hours") is not None:
        metrics["total_active_hours"] = precomputed["total_active_hours"]
    if precomputed.get("total_context_switches") is not None:
        metrics["context_switches"] = precomputed["total_context_switches"]
    json_data["metrics"] = metrics

    return json_data


# ── PAR Loop: scan → deep-dive → enrich ─────────────────────────────────────

_PAR_SCAN_PROMPT = """\
You are MemoryOS, an AI data-gathering planner. Analyze the data below and
identify items that need deeper context for a high-quality report.

For each item, output a JSON array of lookup requests. Each request is an object:
  {{"type": "vault_read", "path": "<vault-relative path>"}}
  {{"type": "search", "query": "<search terms>", "source_type": "<email|meeting|teams>"}}

Return ONLY the JSON array. Maximum 10 lookups. Focus on:
- Prior meeting transcripts for today's high-priority meetings
- Email threads referenced in commitments
- Attendee interaction history for key meetings

If no deeper lookups are needed, return an empty array: []
"""


def _run_par_deep_dive(
    vault_data: str,
    vault: Path,
    skill_name: str,
    today_str: str,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> str:
    """PAR deep-dive phase: scan data, identify gaps, fetch in parallel."""
    scan_messages = [
        {"role": "system", "content": _PAR_SCAN_PROMPT},
        {"role": "user", "content": (
            f"Today is {today_str}.\n\n"
            f"=== GATHERED DATA (first 80k chars) ===\n{vault_data[:80000]}"
        )},
    ]

    try:
        raw = llm_provider.complete(
            scan_messages,
            skill_name=skill_name,
            provider_override=provider_override,
            model_override=model_override or "gpt-5-mini",
        )
    except Exception as exc:
        logger.warning("PAR scan LLM call failed: %s", exc)
        return ""

    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start < 0 or end <= start:
            return ""
        lookups = json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("PAR scan returned invalid JSON: %s", exc)
        return ""

    if not lookups or not isinstance(lookups, list):
        return ""

    lookups = lookups[:10]
    logger.info("PAR deep-dive: %d lookups requested for '%s'", len(lookups), skill_name)

    enriched_parts: list[str] = []

    def _execute_lookup(lk: dict[str, Any]) -> tuple[str, str]:
        lk_type = lk.get("type", "")
        if lk_type == "vault_read":
            path = lk.get("path", "")
            if not path:
                return "", ""
            content = _read_vault_file(vault, path, max_chars=30_000)
            if content:
                return f"=== PAR vault_read: {path} ===", content
        elif lk_type == "search":
            query = lk.get("query", "")
            stype = lk.get("source_type")
            if not query:
                return "", ""
            result = _run_cli_search(query, stype, limit=10)
            return f"=== PAR search: '{query}' (type={stype}) ===", result
        return "", ""

    with ThreadPoolExecutor(max_workers=min(len(lookups), 6)) as pool:
        futures = [pool.submit(_execute_lookup, lk) for lk in lookups]
        for fut in futures:
            try:
                label, content = fut.result(timeout=60)
                if label and content:
                    enriched_parts.append(f"{label}\n{content}\n")
            except Exception as exc:
                logger.warning("PAR lookup failed: %s", exc)

    if enriched_parts:
        logger.info("PAR deep-dive: gathered %d enrichments (%d chars)",
                     len(enriched_parts), sum(len(p) for p in enriched_parts))
        return "\n=== PAR DEEP-DIVE ENRICHMENTS ===\n\n" + "\n".join(enriched_parts)
    return ""


# ── Calendar completeness validation ─────────────────────────────────────────

def _extract_calendar_items(vault_data: str) -> list[str]:
    """Extract all calendar ## headings from vault data."""
    items: list[str] = []
    in_calendar = False
    for line in vault_data.splitlines():
        if "# Calendar --" in line:
            in_calendar = True
            continue
        if in_calendar and line.startswith("# ") and not line.startswith("## "):
            in_calendar = False
            continue
        if in_calendar and line.startswith("## "):
            items.append(line[3:].strip())
    return items


def _validate_calendar_completeness(
    report: str,
    vault_data: str,
    skill_name: str,
) -> tuple[bool, list[str]]:
    """Check that every calendar item from vault_data is referenced in the report.

    Returns (all_present, missing_items).
    """
    cal_items = _extract_calendar_items(vault_data)
    if not cal_items:
        return True, []

    missing: list[str] = []
    for item in cal_items:
        time_match = re.match(r"(\d{1,2}:\d{2})\s*-\s*\d{1,2}:\d{2}:\s*(.+)", item)
        if time_match:
            search_terms = [time_match.group(2).strip(), time_match.group(1)]
        elif item.startswith("All Day:"):
            search_terms = [item.replace("All Day: ", "").strip()]
        else:
            search_terms = [item]

        found = any(term in report for term in search_terms)
        if not found:
            missing.append(item)

    logger.info(
        "Calendar validation for '%s': %d/%d items found, %d missing",
        skill_name, len(cal_items) - len(missing), len(cal_items), len(missing),
    )
    return len(missing) == 0, missing


def _data_fingerprint(data: str) -> str:
    """Return a short hash of gathered data for change detection."""
    return hashlib.sha256(data.encode("utf-8", errors="replace")).hexdigest()[:16]


def _load_last_fingerprint(skill_name: str) -> str | None:
    state_path = REPO_DIR / "config" / "skill_refresh_state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return state.get(skill_name)
    except Exception:
        return None


def _save_fingerprint(skill_name: str, fingerprint: str) -> None:
    state_path = REPO_DIR / "config" / "skill_refresh_state.json"
    state: dict[str, str] = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    state[skill_name] = fingerprint
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_skill(
    skill_name: str,
    send_email: bool = False,
    dry_run: bool = False,
    refresh_only: bool = False,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Execute a skill headlessly.

    Parameters
    ----------
    refresh_only:
        If True, only regenerate the report when the underlying vault data
        has changed since the last run.  Email is suppressed unless the data
        actually changed.

    Returns
    -------
    Dict with keys: ok, report_path, json_path, email_result, error.
    """
    _load_env()

    skills_dir = _get_skills_dir()
    skill_dir = skills_dir / skill_name
    skill_md = skill_dir / "SKILL.md"
    manifest_path = skill_dir / "manifest.yaml"

    if not skill_md.is_file():
        return {"ok": False, "error": f"SKILL.md not found: {skill_md}"}

    skill_content = skill_md.read_text(encoding="utf-8", errors="replace")

    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to parse manifest.yaml: %s", exc)

    vault = _get_vault_path()
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    date_str = datetime.now().strftime("%Y-%m-%d")

    logger.info("Running skill '%s' (vault=%s)", skill_name, vault)

    vault_data = _gather_data(manifest, vault)

    if refresh_only:
        fp = _data_fingerprint(vault_data)
        last_fp = _load_last_fingerprint(skill_name)
        if fp == last_fp:
            logger.info("Refresh-only: data unchanged for '%s', skipping", skill_name)
            return {"ok": True, "report_path": None, "json_path": None,
                    "email_result": None, "skipped": True}
        _save_fingerprint(skill_name, fp)
        logger.info("Refresh-only: data changed for '%s', regenerating", skill_name)

    preprocessors = manifest.get("preprocessor", "")
    if isinstance(preprocessors, str):
        preprocessors = [p.strip() for p in preprocessors.split(",") if p.strip()]

    precomputed_stats: dict[str, Any] | None = None
    if "activity_stats" in preprocessors:
        precomputed_stats = _preprocess_activity_stats(vault_data, vault)
        if precomputed_stats:
            vault_data = _inject_activity_summary(vault_data, precomputed_stats)

    if "calendar" in preprocessors:
        vault_data = _preprocess_calendar_noise(vault_data)

    web_data = ""
    web_data_structured: dict[str, Any] | None = None
    ds = manifest.get("data_sources", {})
    if ds.get("web_search"):
        web_data = _gather_web_data(skill_dir)
        web_data_structured = _gather_web_data_structured(skill_dir)

    if manifest.get("par_deep_dive", False):
        enrichment = _run_par_deep_dive(
            vault_data, vault, skill_name, today_str,
            provider_override=provider_override,
            model_override=model_override,
        )
        if enrichment:
            vault_data += "\n" + enrichment

    core_context = _load_core_context(vault)
    shared_context = _load_shared_context(skills_dir)

    max_data_chars = llm_provider.get_max_data_chars(
        model=model_override, skill_name=skill_name,
    )
    total_data = len(vault_data) + len(web_data) + len(core_context) + len(shared_context)
    if total_data > max_data_chars:
        logger.warning("Total data %d chars exceeds %d limit, truncating", total_data, max_data_chars)
        if len(core_context) > 20000:
            core_context = core_context[:20000] + "\n[... core context truncated ...]\n"
        budget = max_data_chars - len(core_context) - len(shared_context) - len(web_data)
        if budget < 50000:
            budget = 50000
            if len(web_data) > 100000:
                web_data = web_data[:100000] + "\n[... web data truncated ...]\n"
        if len(vault_data) > budget:
            vault_data = vault_data[:budget] + "\n[... vault data truncated to fit context window ...]\n"

    structured_output = manifest.get("structured_output", False)

    # -- Step 1: Generate Markdown report (always) --
    md_system_prompt = (
        "You are MemoryOS, an AI executive assistant. Generate the requested report "
        "following the skill instructions exactly. Output clean Markdown only -- "
        "no preamble, no explanations, just the report.\n\n"
        f"Today is {today_str}.\n\n"
        "CALENDAR COMPLETENESS RULE: You MUST include EVERY item from the "
        "calendar data (every ## heading in calendar.md). This includes meetings, "
        "work blocks, focus blocks, all-day events, task items, and personal blocks. "
        "Count all ## headings and verify your output includes every one. "
        "Do NOT skip items that lack attendees, location, or notes.\n\n"
    )

    if core_context:
        md_system_prompt += "=== CORE WORK CONTEXT ===\n" + core_context + "\n\n"
    if shared_context:
        md_system_prompt += shared_context + "\n"

    md_system_prompt += "=== SKILL INSTRUCTIONS ===\n" + skill_content + "\n"

    user_parts = ["Generate the report using the data below.\n"]
    if vault_data.strip():
        user_parts.append("=== VAULT DATA ===\n" + vault_data)
    if web_data.strip():
        user_parts.append("=== WEB SEARCH RESULTS ===\n" + web_data)
    if not vault_data.strip() and not web_data.strip():
        user_parts.append("(No data was gathered. Generate the best report you can with general knowledge.)")

    md_messages = [
        {"role": "system", "content": md_system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    try:
        report = llm_provider.complete(
            md_messages,
            skill_name=skill_name,
            provider_override=provider_override,
            model_override=model_override,
        )
    except Exception as exc:
        return {"ok": False, "error": f"LLM call failed: {exc}"}

    # -- Step 1.1: Calendar completeness validation (PAR review) --
    cal_valid, cal_missing = _validate_calendar_completeness(
        report, vault_data, skill_name,
    )
    if not cal_valid:
        logger.warning(
            "Calendar PAR validation failed for '%s': %d missing items: %s",
            skill_name, len(cal_missing), cal_missing,
        )
        correction_msg = (
            "CRITICAL: Your report is MISSING the following calendar items:\n"
            + "\n".join(f"  - {item}" for item in cal_missing)
            + "\n\nYou MUST include ALL items from the calendar data, including "
            "work blocks, task items, personal blocks, and all-day events -- "
            "even if they have no attendees or location. Regenerate the "
            "COMPLETE report with every calendar item included."
        )
        try:
            report = llm_provider.complete(
                md_messages + [
                    {"role": "assistant", "content": report},
                    {"role": "user", "content": correction_msg},
                ],
                skill_name=skill_name,
                provider_override=provider_override,
                model_override=model_override,
            )
            cal_valid2, cal_missing2 = _validate_calendar_completeness(
                report, vault_data, skill_name,
            )
            if not cal_valid2:
                logger.warning(
                    "Calendar PAR correction still missing %d items for '%s': %s",
                    len(cal_missing2), skill_name, cal_missing2,
                )
        except Exception as exc:
            logger.warning("Calendar PAR correction LLM call failed: %s", exc)

    if dry_run:
        print(report)
        return {"ok": True, "report_path": None, "json_path": None, "email_result": None}

    report_path = _save_report(vault, skill_name, report)

    # -- Step 1.5: Analysis pass (trend detection, risk scoring, insights) --
    prior_report = _load_prior_report(vault, skill_name) if structured_output else None
    analysis_addendum = ""
    if structured_output:
        logger.info("Running analysis pass for '%s'", skill_name)
        analysis_addendum = _run_analysis_pass(
            report, prior_report, skill_name, today_str,
            provider_override=provider_override,
            model_override=model_override,
        )

    # -- Step 2: Generate structured JSON report (if enabled) --
    # Uses the already-generated markdown report + analysis addendum as input
    # so the LLM converts the distilled report to JSON with enriched insights.
    json_path = None
    json_data: dict[str, Any] | None = None
    if structured_output:
        json_schema_file = skill_dir / "schema.json"
        schema_instruction = ""
        if json_schema_file.is_file():
            schema_instruction = (
                "\n\nOutput JSON matching this schema:\n"
                + json_schema_file.read_text(encoding="utf-8", errors="replace")
            )

        json_system_prompt = (
            "You are MemoryOS, an AI executive assistant. Convert the provided "
            "Markdown report AND analysis addendum into STRUCTURED JSON. Preserve "
            "ALL data points, meetings, emails, accomplishments, and details from "
            "the report. Incorporate the analysis insights (executive_insight, "
            "trend_vs_prior, risk_assessment, recommended_focus, predictions) "
            "into the appropriate schema fields. "
            "Output ONLY valid JSON.\n\n"
            f"Today is {today_str}.\n"
        )
        json_system_prompt += schema_instruction
        json_system_prompt += _JSON_OUTPUT_INSTRUCTION

        json_user_parts = [
            "Convert this Markdown report into the JSON schema above. "
            "Include every data point from the report. Also incorporate "
            "the analysis addendum into the 'analysis' field of the JSON.\n\n"
            "=== MARKDOWN REPORT ===\n" + report
        ]
        if analysis_addendum:
            json_user_parts.append(
                "\n=== ANALYSIS ADDENDUM ===\n" + analysis_addendum
            )

        json_messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": "\n".join(json_user_parts)},
        ]

        try:
            json_response = llm_provider.complete(
                json_messages,
                skill_name=skill_name,
                provider_override=provider_override,
                model_override=model_override,
            )

            json_data = _extract_json_from_response(json_response)
            if json_data:
                if web_data_structured:
                    for topic in json_data.get("topics", []):
                        src_topic = next(
                            (t for t in web_data_structured.get("topics", [])
                             if t.get("name") == topic.get("name")),
                            None,
                        )
                        if src_topic:
                            for article in topic.get("articles", []):
                                src_article = next(
                                    (a for a in src_topic.get("articles", [])
                                     if a.get("url") == article.get("url")),
                                    None,
                                )
                                if src_article:
                                    article.setdefault("thumbnail_url", src_article.get("thumbnail_url", ""))
                                    article.setdefault("favicon_url", src_article.get("favicon_url", ""))

                if precomputed_stats:
                    json_data = _merge_precomputed_into_json(json_data, precomputed_stats)
                json_data = _generate_images_for_report(skill_name, json_data, date_str)
                json_path = str(_save_json_report(vault, skill_name, json_data))
            else:
                logger.warning("Failed to parse JSON from LLM response for %s", skill_name)

        except Exception as exc:
            logger.warning("JSON report generation failed for %s: %s", skill_name, exc)

    # -- Step 3: Email the markdown report --
    email_result = None
    if send_email:
        subject_template = manifest.get("email_subject", f"[MemOS] {skill_name} -- {{date}}")
        subject = subject_template.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        html_override = None
        if json_data:
            try:
                from src.agents.email_renderer import render_rich_email
                images_dir = vault / "90_reports" / skill_name / "images" / date_str
                html_override = render_rich_email(skill_name, json_data, images_dir)
                if html_override:
                    logger.info("Using rich HTML email for %s", skill_name)
            except Exception as exc:
                logger.warning("Rich email render failed, falling back to markdown: %s", exc)
        email_result = emailer.send_report(
            subject=subject, markdown_body=report, html_override=html_override,
        )

    return {
        "ok": True,
        "report_path": str(report_path),
        "json_path": json_path,
        "email_result": email_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryOS Headless Skill Runner")
    parser.add_argument("--skill", required=True, help="Skill name (directory name in skills/)")
    parser.add_argument("--email", action="store_true", help="Send report via email")
    parser.add_argument("--dry-run", action="store_true", help="Print report to stdout only")
    parser.add_argument("--refresh-only", action="store_true",
                        help="Only regenerate if vault data has changed since last run")
    parser.add_argument("--provider", help="Override LLM provider")
    parser.add_argument("--model", help="Override LLM model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result = run_skill(
        skill_name=args.skill,
        send_email=args.email,
        dry_run=args.dry_run,
        refresh_only=args.refresh_only,
        provider_override=args.provider,
        model_override=args.model,
    )

    if not result["ok"]:
        logger.error("Skill run failed: %s", result.get("error"))
        sys.exit(1)

    if result.get("report_path"):
        logger.info("Report: %s", result["report_path"])
    if result.get("json_path"):
        logger.info("JSON: %s", result["json_path"])
    if result.get("email_result"):
        er = result["email_result"]
        if er["ok"]:
            logger.info("Email: %s", er["detail"])
        else:
            logger.error("Email failed: %s", er["detail"])


if __name__ == "__main__":
    main()
