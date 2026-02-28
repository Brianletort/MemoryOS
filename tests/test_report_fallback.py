from __future__ import annotations

from src.dashboard.report_fallback import extract_embedded_json, normalize_report


class TestExtractEmbeddedJson:
    def test_extracts_first_valid_json_object(self) -> None:
        md = """# Report

```text
not json
```

```json
{"a": 1, "b": {"c": 2}}
```
"""
        data = extract_embedded_json(md)
        assert data == {"a": 1, "b": {"c": 2}}

    def test_returns_none_when_no_json(self) -> None:
        md = """# Report

```text
hello
```
"""
        assert extract_embedded_json(md) is None


class TestNormalizeMorningBrief:
    def test_normalizes_markdown_embedded_shape(self) -> None:
        embedded = {
            "day_of_week": "Wednesday",
            "day_score": {
                "score": 60,
                "day_summary": "Leadership-heavy day",
                "composition_08_18": {"meeting_percent": 45, "focus_percent": 40, "admin_percent": 15},
            },
            "energy_map_07_19": [
                {"hour": "07:00", "state": "transition", "note": "Prep"},
                {"hour": "08:00", "state": "meeting", "note": "Block"},
            ],
            "conflicts": {
                "conflict_map": [
                    {
                        "time_block": "13:00-13:30",
                        "overlap": ["A", "B"],
                        "recommendation": "Attend A",
                        "delegate_action": "Ask for notes",
                    }
                ]
            },
            "quick_wins": ["Do X (5 min).", "Do Y (2 min)."],
            "prep_tonight_top5": [
                {"task": "Prep deck", "relates_to": "Exec meeting", "time_needed": "25 min", "priority": 1}
            ],
            "meetings": [{"time_range": "09:00-09:30", "name": "Standup"}],
        }

        normalized = normalize_report("morning-brief", embedded)

        assert normalized["day_score"] == 60
        assert normalized["day_summary"] == "Leadership-heavy day"
        assert normalized["day_composition"]["meeting_percent"] == 45

        assert normalized["energy_map"][0]["availability"] == "transition"
        assert normalized["energy_map"][0]["suggested_use"] == "Prep"

        assert isinstance(normalized["conflicts"], list)
        assert normalized["conflicts"][0]["time_range"] == "13:00-13:30"
        assert normalized["conflicts"][0]["meetings"] == ["A", "B"]

        assert isinstance(normalized["quick_wins"][0], dict)
        assert normalized["quick_wins"][0]["action"] == "Do X (5 min)."

        assert isinstance(normalized["prep_tonight"], list)
        assert normalized["prep_tonight"][0]["related_meeting"] == "Exec meeting"

        assert normalized["meetings"][0]["title"] == "Standup"
        assert normalized["meetings"][0]["time"] == "09:00-09:30"
        assert normalized["meeting_count"] == 1


class TestNormalizePlanMyWeek:
    def test_hoists_day_metrics_and_computes_totals(self) -> None:
        embedded = {
            "week_score": 90,
            "days": [
                {
                    "day_name": "Mon",
                    "date": "2026/02/23",
                    "meetings": {"names": ["A"], "count": 1},
                    "metrics": {"meeting_hours": 3.5, "focus_hours": 2.0, "capacity_percent": 44, "free_hours": 2.5},
                },
                {
                    "day_name": "Tue",
                    "date": "2026/02/24",
                    "meetings": {"names": ["B", "C"], "count": 2},
                    "metrics": {"meeting_hours": 1.5, "focus_hours": 3.0, "capacity_percent": 19, "free_hours": 3.5},
                },
            ],
        }
        normalized = normalize_report("plan-my-week", embedded)

        d0 = normalized["days"][0]
        assert d0["meeting_hours"] == 3.5
        assert d0["focus_hours"] == 2.0
        assert d0["capacity_percent"] == 44
        assert normalized["days"][0]["meetings"]["hours"] == 3.5

        assert normalized["total_meeting_hours"] == 5.0
        assert normalized["total_focus_hours"] == 5.0

