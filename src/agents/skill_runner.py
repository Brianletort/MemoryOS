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
import json
import logging
import os
import subprocess
import sys
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
        return Path.home() / "Data" / "Obsidian" / "Letort"


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

    for search in ds.get("cli_searches", []):
        query = search.get("query", "")
        stype = search.get("type")
        limit = search.get("limit", 10)
        recent_hours = search.get("recent_hours")

        if recent_hours and not query:
            result = _run_cli_recent(stype, limit, recent_hours)
            parts.append(f"=== CLI recent: type={stype}, hours={recent_hours} ===\n{result}\n")
        elif query:
            result = _run_cli_search(query, stype, limit)
            parts.append(f"=== CLI search: '{query}' (type={stype}) ===\n{result}\n")

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


def run_skill(
    skill_name: str,
    send_email: bool = False,
    dry_run: bool = False,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Execute a skill headlessly.

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

    web_data = ""
    web_data_structured: dict[str, Any] | None = None
    ds = manifest.get("data_sources", {})
    if ds.get("web_search"):
        web_data = _gather_web_data(skill_dir)
        web_data_structured = _gather_web_data_structured(skill_dir)

    core_context = _load_core_context(vault)
    shared_context = _load_shared_context(skills_dir)

    # Cap total prompt size to fit within model context window (~200k tokens ~800k chars).
    # Reserve space for system prompt (~20k), skill instructions (~10k), response.
    MAX_DATA_CHARS = 600_000
    total_data = len(vault_data) + len(web_data) + len(core_context) + len(shared_context)
    if total_data > MAX_DATA_CHARS:
        logger.warning("Total data %d chars exceeds %d limit, truncating", total_data, MAX_DATA_CHARS)
        if len(core_context) > 20000:
            core_context = core_context[:20000] + "\n[... core context truncated ...]\n"
        budget = MAX_DATA_CHARS - len(core_context) - len(shared_context) - len(web_data)
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
