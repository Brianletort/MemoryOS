from __future__ import annotations

import json
import re
from typing import Any


_JSON_FENCE_RE = re.compile(r"```[^\n]*\r?\n([\s\S]*?)\r?\n```", re.IGNORECASE)


def extract_embedded_json(markdown: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object embedded in a fenced code block.

    Skills often embed a JSON object inside the markdown report (```json ... ```).
    This enables the dashboard rich-view even when the standalone .json artifact
    is missing.
    """
    if not markdown:
        return None

    for m in _JSON_FENCE_RE.finditer(markdown):
        candidate = m.group(1).strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def normalize_report(
    skill_name: str,
    data: dict[str, Any],
    markdown: str = "",
) -> dict[str, Any]:
    """Normalize older/markdown-embedded JSON into the dashboard renderer shapes."""
    if not isinstance(data, dict):
        return {}

    if skill_name == "morning-brief":
        return _normalize_morning_brief(data)
    if skill_name == "plan-my-week":
        return _normalize_plan_my_week(data)
    if skill_name == "commitment-tracker":
        return _normalize_commitment_tracker(data, markdown)
    return data


def _normalize_morning_brief(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)

    # day_score: sometimes emitted as nested object with score + composition
    day_score = out.get("day_score")
    if isinstance(day_score, dict):
        score_val = day_score.get("score")
        if isinstance(score_val, (int, float)):
            out["day_score"] = int(score_val)
        else:
            out["day_score"] = out.get("day_score") if isinstance(out.get("day_score"), (int, float)) else 0
        if isinstance(day_score.get("day_summary"), str) and not out.get("day_summary"):
            out["day_summary"] = day_score["day_summary"]
        comp = day_score.get("composition_08_18")
        if isinstance(comp, dict) and not isinstance(out.get("day_composition"), dict):
            out["day_composition"] = {
                "meeting_percent": comp.get("meeting_percent", 0),
                "focus_percent": comp.get("focus_percent", 0),
                "admin_percent": comp.get("admin_percent", 0),
            }
    elif isinstance(day_score, (int, float)):
        out["day_score"] = int(day_score)
    else:
        out["day_score"] = 0

    # energy_map: older markdown embedded JSON uses energy_map_07_19
    if not isinstance(out.get("energy_map"), list) and isinstance(out.get("energy_map_07_19"), list):
        em: list[dict[str, Any]] = []
        for slot in out["energy_map_07_19"]:
            if not isinstance(slot, dict):
                continue
            em.append({
                "hour": slot.get("hour", ""),
                "availability": slot.get("availability") or slot.get("state") or "transition",
                "suggested_use": slot.get("suggested_use") or slot.get("note") or "",
            })
        out["energy_map"] = em

    # conflicts: older markdown embedded JSON uses conflicts.conflict_map
    conflicts = out.get("conflicts")
    if isinstance(conflicts, dict) and isinstance(conflicts.get("conflict_map"), list):
        norm_conflicts: list[dict[str, Any]] = []
        for c in conflicts["conflict_map"]:
            if not isinstance(c, dict):
                continue
            norm_conflicts.append({
                "time_range": c.get("time_range") or c.get("time_block") or "",
                "meetings": c.get("meetings") or c.get("overlap") or [],
                "recommendation": c.get("recommendation") or "",
                "delegate_action": c.get("delegate_action") or "",
            })
        out["conflicts"] = norm_conflicts

    # quick_wins: can be list[str] in embedded JSON; UI expects objects
    qw = out.get("quick_wins")
    if isinstance(qw, list) and qw and all(isinstance(x, str) for x in qw):
        out["quick_wins"] = [{"action": s, "impact": "", "time_needed": ""} for s in qw]

    # prep_tonight: older markdown embedded JSON uses prep_tonight_top5
    if not isinstance(out.get("prep_tonight"), list) and isinstance(out.get("prep_tonight_top5"), list):
        prep: list[dict[str, Any]] = []
        for p in out["prep_tonight_top5"]:
            if not isinstance(p, dict):
                continue
            prep.append({
                "task": p.get("task", ""),
                "priority": p.get("priority", 5),
                "related_meeting": p.get("related_meeting") or p.get("relates_to") or "",
                "time_needed": p.get("time_needed", ""),
            })
        out["prep_tonight"] = prep

    # meetings: ensure title/time fields exist
    meetings = out.get("meetings")
    if isinstance(meetings, list):
        norm_meetings: list[dict[str, Any]] = []
        for m in meetings:
            if not isinstance(m, dict):
                continue
            mm = dict(m)
            if not mm.get("title"):
                mm["title"] = mm.get("name") or mm.get("meeting") or mm.get("summary") or ""
            if not mm.get("time"):
                mm["time"] = mm.get("time_range") or ""
            norm_meetings.append(mm)
        out["meetings"] = norm_meetings

    # Fill counts if missing
    if not isinstance(out.get("meeting_count"), int):
        out["meeting_count"] = len(out.get("meetings") or [])
    if not isinstance(out.get("priority_email_count"), int):
        out["priority_email_count"] = len(out.get("priority_emails") or [])
    if not isinstance(out.get("commitment_count"), int):
        out["commitment_count"] = len(out.get("commitments_yours") or []) + len(out.get("commitments_others") or [])

    return out


def _normalize_plan_my_week(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    days = out.get("days")
    if isinstance(days, list):
        norm_days: list[dict[str, Any]] = []
        for day in days:
            if not isinstance(day, dict):
                continue
            nd = dict(day)
            metrics = nd.get("metrics") if isinstance(nd.get("metrics"), dict) else {}
            # Hoist commonly-used fields when they are nested under metrics.
            for k in ("capacity_percent", "focus_hours", "meeting_hours", "free_hours"):
                if nd.get(k) is None and metrics.get(k) is not None:
                    nd[k] = metrics.get(k)
            # Ensure day.meetings.hours exists for the renderer.
            mtgs = nd.get("meetings")
            if isinstance(mtgs, dict):
                if mtgs.get("hours") is None and nd.get("meeting_hours") is not None:
                    mtgs["hours"] = nd.get("meeting_hours")
                if mtgs.get("count") is None and isinstance(mtgs.get("names"), list):
                    mtgs["count"] = len(mtgs["names"])
                nd["meetings"] = mtgs
            norm_days.append(nd)
        out["days"] = norm_days

    # Compute totals if missing (best-effort)
    if out.get("total_meeting_hours") is None and isinstance(out.get("days"), list):
        out["total_meeting_hours"] = round(sum(float((d.get("meeting_hours") or (d.get("meetings") or {}).get("hours") or 0) or 0) for d in out["days"] if isinstance(d, dict)), 1)
    if out.get("total_focus_hours") is None and isinstance(out.get("days"), list):
        out["total_focus_hours"] = round(sum(float((d.get("focus_hours") or 0) or 0) for d in out["days"] if isinstance(d, dict)), 1)

    return out


# ---------------------------------------------------------------------------
# Commitment Tracker normalizer
# ---------------------------------------------------------------------------

_STRIP_MD_RE = re.compile(r"\*\*([^*]+)\*\*")
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")


def _strip_md(text: str) -> str:
    """Remove markdown bold markers and backtick wrappers."""
    text = _STRIP_MD_RE.sub(r"\1", text)
    return text.strip().strip("`")


def _parse_table_rows(lines: list[str]) -> list[list[str]]:
    """Return cell lists for data rows of a markdown table (skip header + separator)."""
    rows: list[list[str]] = []
    header_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            if header_seen:
                break
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if _TABLE_SEP_RE.match(stripped):
            header_seen = True
            continue
        if not header_seen:
            header_seen = True
            continue
        rows.append(cells)
    return rows


def _status_key(raw: str) -> str:
    """Map free-text status to a schema enum value."""
    low = raw.lower()
    if "overdue" in low:
        return "overdue"
    if "blocked" in low:
        return "blocked"
    if "complete" in low or "done" in low:
        return "completed"
    if "in progress" in low or "in-progress" in low:
        return "in_progress"
    if "not started" in low:
        return "not_started"
    return "in_progress"


def _normalize_commitment_tracker(
    d: dict[str, Any],
    markdown: str = "",
) -> dict[str, Any]:
    out = dict(d)

    # Normalize analysis.recommended_focus from array to newline-joined string
    analysis = out.get("analysis")
    if isinstance(analysis, dict):
        rf = analysis.get("recommended_focus")
        if isinstance(rf, list):
            analysis["recommended_focus"] = "\n".join(str(x) for x in rf)

    # Normalize executive_header.metrics from strings to {label, value} objects
    eh = out.get("executive_header")
    if isinstance(eh, dict):
        metrics = eh.get("metrics")
        if isinstance(metrics, list):
            norm_metrics: list[dict[str, Any]] = []
            for m in metrics:
                if isinstance(m, str):
                    idx = m.find(":")
                    if idx >= 0:
                        label = m[:idx].strip()
                        value = m[idx + 1:].strip()
                        try:
                            value = int(value)
                        except (ValueError, TypeError):
                            pass
                        norm_metrics.append({"label": label, "value": value})
                    else:
                        norm_metrics.append({"label": m, "value": ""})
                elif isinstance(m, dict):
                    norm_metrics.append(m)
            eh["metrics"] = norm_metrics

    # If core arrays are already present, nothing more to do
    if out.get("your_commitments"):
        return out

    # Parse commitment tables from the markdown to fill missing arrays
    if not markdown:
        return out

    lines = markdown.split("\n")
    your_commitments: list[dict[str, Any]] = []
    others_owe_you: list[dict[str, Any]] = []
    recently_completed: list[dict[str, Any]] = []

    section = ""
    section_lines: list[str] = []

    def _flush_section() -> None:
        nonlocal section, section_lines
        if section == "your_commitments" and section_lines:
            for cells in _parse_table_rows(section_lines):
                if len(cells) < 5:
                    continue
                commitment = _strip_md(cells[1]) if len(cells) > 1 else ""
                project = _strip_md(cells[2]) if len(cells) > 2 else ""
                deadline = _strip_md(cells[3]) if len(cells) > 3 else ""
                priority = _strip_md(cells[4]) if len(cells) > 4 else ""
                effort = _strip_md(cells[5]) if len(cells) > 5 else ""
                pct_raw = _strip_md(cells[6]) if len(cells) > 6 else "0"
                category = _strip_md(cells[7]) if len(cells) > 7 else ""
                status_raw = _strip_md(cells[8]) if len(cells) > 8 else ""
                stakeholders = _strip_md(cells[9]) if len(cells) > 9 else ""
                try:
                    pct = int(pct_raw)
                except (ValueError, TypeError):
                    pct = 0
                your_commitments.append({
                    "commitment": commitment,
                    "project": project,
                    "deadline": deadline,
                    "priority": priority.upper() if priority else "",
                    "effort": effort,
                    "percent_complete": pct,
                    "category": category,
                    "status": _status_key(status_raw),
                    "owed_to": stakeholders,
                    "source": _strip_md(cells[10]) if len(cells) > 10 else "",
                })
        elif section == "others_owe_you" and section_lines:
            for cells in _parse_table_rows(section_lines):
                if len(cells) < 3:
                    continue
                person = _strip_md(cells[1]) if len(cells) > 1 else ""
                commitment = _strip_md(cells[2]) if len(cells) > 2 else ""
                project = _strip_md(cells[3]) if len(cells) > 3 else ""
                deadline = _strip_md(cells[4]) if len(cells) > 4 else ""
                status_raw = _strip_md(cells[5]) if len(cells) > 5 else ""
                others_owe_you.append({
                    "person": person,
                    "commitment": commitment,
                    "project": project,
                    "deadline": deadline,
                    "status": "open" if "open" in status_raw.lower() else status_raw.lower(),
                    "source": _strip_md(cells[6]) if len(cells) > 6 else "",
                })
        elif section == "recently_completed" and section_lines:
            for line in section_lines:
                stripped = line.strip()
                if not stripped.startswith("-"):
                    continue
                text = stripped.lstrip("- ").strip()
                date_match = re.search(
                    r"completed?\s+(\w+\s+\d+|\d{4}-\d{2}-\d{2})",
                    text, re.IGNORECASE,
                )
                recently_completed.append({
                    "commitment": text.split("—")[0].strip() if "—" in text else text,
                    "completed_date": date_match.group(1) if date_match else "",
                    "evidence": text,
                })
        section = ""
        section_lines = []

    for line in lines:
        heading = line.strip().lower()
        if "your open commitments" in heading:
            _flush_section()
            section = "your_commitments"
            continue
        if "others owe you" in heading:
            _flush_section()
            section = "others_owe_you"
            continue
        if "recently completed" in heading:
            _flush_section()
            section = "recently_completed"
            continue
        if line.startswith("## ") and section:
            _flush_section()
            continue
        if line.startswith("---") and section:
            _flush_section()
            continue
        if section:
            section_lines.append(line)

    _flush_section()

    if your_commitments:
        out["your_commitments"] = your_commitments
    if others_owe_you:
        out["others_owe_you"] = others_owe_you
    if recently_completed:
        out["recently_completed"] = recently_completed

    # Fill top-level counts from parsed data
    if "total_open" not in out:
        out["total_open"] = len(your_commitments)
    if "total_overdue" not in out:
        out["total_overdue"] = sum(
            1 for c in your_commitments if c.get("status") == "overdue"
        )

    return out


# ---------------------------------------------------------------------------
# Project Brief markdown-to-JSON parser
# ---------------------------------------------------------------------------

def parse_project_brief_markdown(markdown: str) -> dict[str, Any] | None:
    """Parse a project-brief markdown report into structured JSON.

    Used when no standalone .json file and no embedded JSON exist.
    """
    if not markdown:
        return None

    lines = markdown.split("\n")
    result: dict[str, Any] = {}

    # --- Executive header ---
    eh: dict[str, Any] = {}
    portfolio: dict[str, Any] = {}

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("# project brief:"):
            break
        if low.startswith("- **bluf:**"):
            eh["bluf"] = _strip_md(stripped.split(":**", 1)[1].strip() if ":**" in stripped else "")
        elif "portfolio status:" in low:
            raw_status = _strip_md(stripped.split(":**", 1)[1].strip() if ":**" in stripped else "")
            if "red" in raw_status.lower():
                eh["status"] = "red"
            elif "yellow" in raw_status.lower():
                eh["status"] = "yellow"
            else:
                eh["status"] = "green"
        elif low.startswith("- **as-of:**") or low.startswith("- **as of:**"):
            result["date"] = _strip_md(stripped.split(":**", 1)[1].strip() if ":**" in stripped else "")

    # Parse the portfolio metrics table (| Metric | Value |)
    _parse_portfolio_metrics(lines, portfolio)

    if portfolio:
        result["portfolio_summary"] = portfolio
        if "status" not in eh:
            if portfolio.get("blocked", 0) > 0:
                eh["status"] = "red"
            elif portfolio.get("at_risk", 0) > 0:
                eh["status"] = "yellow"
            else:
                eh["status"] = "green"
        eh.setdefault("metrics", [
            {"label": "Projects", "value": portfolio.get("total_projects", 0)},
            {"label": "On Track", "value": portfolio.get("on_track", 0), "color": "green"},
            {"label": "At Risk", "value": portfolio.get("at_risk", 0),
             "color": "yellow" if portfolio.get("at_risk", 0) > 0 else "green"},
            {"label": "Blocked", "value": portfolio.get("blocked", 0),
             "color": "red" if portfolio.get("blocked", 0) > 0 else "green"},
            {"label": "Avg Health", "value": portfolio.get("avg_health", 0)},
        ])

    if eh:
        result["executive_header"] = eh

    # --- Parse individual project sections ---
    projects = _parse_project_sections(lines)
    if projects:
        result["projects"] = projects

    if not result.get("date"):
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", markdown[:200])
        if date_match:
            result["date"] = date_match.group(0)

    # Build analysis from executive insights across projects
    insights = [p.get("executive_insight", "") for p in projects if p.get("executive_insight")]
    if insights:
        result["analysis"] = {
            "executive_insight": insights[0],
            "biggest_risk": eh.get("bluf", ""),
        }

    return result if result.get("projects") else None


def _parse_portfolio_metrics(lines: list[str], portfolio: dict[str, Any]) -> None:
    """Extract portfolio summary from a Metric|Value table."""
    metric_map = {
        "total projects": "total_projects",
        "on track": "on_track",
        "at risk": "at_risk",
        "blocked": "blocked",
        "avg health score": "avg_health",
        "avg health": "avg_health",
    }
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            if in_table:
                break
            continue
        if _TABLE_SEP_RE.match(stripped):
            in_table = True
            continue
        if not in_table:
            cells = [c.strip().lower() for c in stripped.split("|")[1:-1]]
            if any("metric" in c for c in cells):
                in_table = False
                continue
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if len(cells) >= 2:
            key = cells[0].strip().lower()
            val_str = _strip_md(cells[1])
            field = metric_map.get(key)
            if field:
                try:
                    portfolio[field] = int(val_str)
                except (ValueError, TypeError):
                    portfolio[field] = val_str


def _parse_project_sections(lines: list[str]) -> list[dict[str, Any]]:
    """Split markdown into per-project sections and parse each."""
    projects: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_name = ""

    for line in lines:
        match = re.match(r"^#\s+Project Brief:\s*(.+)", line)
        if match:
            if current_name and current_lines:
                projects.append(_parse_single_project(current_name, current_lines))
            current_name = _strip_md(match.group(1).strip())
            current_lines = []
            continue
        if current_name:
            if line.startswith("---"):
                projects.append(_parse_single_project(current_name, current_lines))
                current_name = ""
                current_lines = []
                continue
            current_lines.append(line)

    if current_name and current_lines:
        projects.append(_parse_single_project(current_name, current_lines))

    return projects


def _parse_single_project(name: str, lines: list[str]) -> dict[str, Any]:
    """Parse one project section into a structured dict."""
    proj: dict[str, Any] = {"name": name}

    # Parse the header line: **Last activity:** ... | **Status:** ... | **Health:** ...
    for line in lines[:5]:
        if "last activity:" in line.lower():
            parts = line.split("|")
            for part in parts:
                low = part.strip().lower()
                if "status:" in low:
                    raw = _strip_md(part.split(":**", 1)[1].strip() if ":**" in part else "")
                    proj["status"] = _map_project_status(raw)
                elif "health:" in low:
                    raw = _strip_md(part.split(":**", 1)[1].strip() if ":**" in part else "")
                    score_match = re.search(r"(\d+)", raw)
                    if score_match:
                        proj["health_score"] = int(score_match.group(1))
            break

    proj.setdefault("status", "On Track")
    proj.setdefault("health_score", 0)

    # Split into sub-sections
    section = ""
    section_lines: list[str] = []
    sections: dict[str, list[str]] = {}

    for line in lines:
        if line.startswith("## "):
            if section:
                sections[section] = section_lines
            section = line.strip().lstrip("#").strip().lower()
            section_lines = []
            continue
        if section:
            section_lines.append(line)
    if section:
        sections[section] = section_lines

    # Summary
    summary_lines = sections.get("summary", [])
    if summary_lines:
        proj["summary"] = "\n".join(l for l in summary_lines if l.strip()).strip()

    # Stakeholders
    stakeholder_lines = sections.get("key stakeholders", [])
    if stakeholder_lines:
        proj["stakeholders"] = []
        for cells in _parse_table_rows(stakeholder_lines):
            if len(cells) >= 2:
                proj["stakeholders"].append({
                    "name": _strip_md(cells[0]),
                    "role": _strip_md(cells[1]),
                    "last_activity": _strip_md(cells[2]) if len(cells) > 2 else "",
                })

    # Timeline
    timeline_lines = sections.get("timeline", [])
    if timeline_lines:
        proj["timeline"] = []
        for cells in _parse_table_rows(timeline_lines):
            if len(cells) >= 2:
                proj["timeline"].append({
                    "date": _strip_md(cells[0]),
                    "event": _strip_md(cells[1]),
                    "source": _strip_md(cells[2]) if len(cells) > 2 else "",
                })

    # Milestones
    milestone_lines = sections.get("milestones", [])
    if milestone_lines:
        proj["milestones"] = []
        for cells in _parse_table_rows(milestone_lines):
            if len(cells) >= 3:
                proj["milestones"].append({
                    "name": _strip_md(cells[0]),
                    "target_date": _strip_md(cells[1]),
                    "status": _strip_md(cells[2]),
                    "owner": _strip_md(cells[3]) if len(cells) > 3 else "",
                })

    # Open decisions
    decision_lines = sections.get("current status", [])
    if decision_lines:
        proj["open_decisions"] = _parse_decisions(decision_lines)
        ws = _parse_workstreams(decision_lines)
        if ws:
            proj["active_workstreams"] = ws

    # Blockers & Risks
    for key in sections:
        if "blocker" in key or "risk" in key:
            risk_lines = sections[key]
            proj["blockers"] = []
            proj["risk_matrix"] = []
            for cells in _parse_table_rows(risk_lines):
                if len(cells) >= 3:
                    proj["risk_matrix"].append({
                        "risk": _strip_md(cells[0]),
                        "probability": _strip_md(cells[1]).lower(),
                        "impact": _strip_md(cells[2]).lower(),
                        "mitigation": _strip_md(cells[3]) if len(cells) > 3 else "",
                    })
            break

    # Next steps
    for key in sections:
        if "next step" in key or "recommended" in key:
            proj["next_steps"] = []
            for line in sections[key]:
                stripped = line.strip()
                if re.match(r"^\d+[.)]\s", stripped):
                    proj["next_steps"].append(
                        _strip_md(re.sub(r"^\d+[.)]\s*", "", stripped))
                    )
            break

    # Executive insight (often at the end of the section)
    for line in lines:
        if "executive insight" in line.lower() and ":" in line:
            insight = line.split(":", 1)[1].strip() if ":" in line else ""
            proj["executive_insight"] = _strip_md(insight)
            break

    # Recent communications
    for key in sections:
        if "recent communication" in key:
            proj["recent_communications"] = []
            for line in sections[key]:
                stripped = line.strip()
                if stripped.startswith("-"):
                    text = stripped.lstrip("- ").strip()
                    subj_match = re.match(r"\*\*(.+?)\*\*\s*\(([^)]+)\)", text)
                    if subj_match:
                        proj["recent_communications"].append({
                            "subject": subj_match.group(1),
                            "date": subj_match.group(2),
                            "summary": _strip_md(text.split("—", 1)[1].strip() if "—" in text else ""),
                        })
            break

    return proj


def _parse_decisions(lines: list[str]) -> list[dict[str, Any]]:
    """Extract open decisions from the Current Status section."""
    decisions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- **Decision:**") or stripped.startswith("**Decision:**"):
            if current:
                decisions.append(current)
            current = {"decision": _strip_md(stripped.split(":**", 1)[1].strip())}
        elif current:
            low = stripped.lower()
            if "waiting on:" in low:
                current["waiting_on"] = _strip_md(stripped.split(":**", 1)[1].strip())
            elif "recommended action:" in low:
                current["recommended_action"] = _strip_md(stripped.split(":**", 1)[1].strip())
            elif "deadline:" in low:
                current["deadline"] = _strip_md(stripped.split(":**", 1)[1].strip())

    if current:
        decisions.append(current)
    return decisions


def _parse_workstreams(lines: list[str]) -> list[str]:
    """Extract active workstreams bullet points."""
    workstreams: list[str] = []
    in_ws = False
    for line in lines:
        stripped = line.strip()
        if "active workstream" in stripped.lower():
            in_ws = True
            continue
        if in_ws:
            if stripped.startswith("- "):
                workstreams.append(_strip_md(stripped.lstrip("- ").strip()))
            elif stripped.startswith("**") or not stripped:
                if workstreams:
                    break
    return workstreams


def _map_project_status(raw: str) -> str:
    low = raw.lower()
    if "blocked" in low:
        return "Blocked"
    if "at risk" in low:
        return "At Risk"
    if "complete" in low:
        return "Completed"
    return "On Track"

