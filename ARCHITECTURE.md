# MemoryOS Architecture

## System Overview

MemoryOS is a three-layer system: **collection**, **memory**, and **interface**.

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                       │
│  Cursor (@context)  │  CLI  │  ChatGPT  │  Dashboard   │
│                     │       │           │  Agents ←→ HITL│
└────────────┬────────┴───┬───┴───────────┴──────┬────────┘
             │            │                      │
┌────────────▼────────────▼──────────────────────▼────────┐
│                     Memory Layer                        │
│  SQLite FTS5 Index  │  Context Generator  │  Tiers      │
│  (sub-ms search)    │  (today/week/emails) │ (hot/warm/cold)│
└────────────┬────────────────────────────────────────────┘
             │ scans
┌────────────▼────────────────────────────────────────────┐
│               Data Collection Layer                     │
│  Extractors → Obsidian Vault (Markdown source of truth) │
│  Screenpipe │ Mail/Calendar │ Outlook/Graph │ OneDrive  │
└─────────────────────────────────────────────────────────┘
```

## Data Collection Layer

### Extractors

Each extractor is an independent Python script that:
1. Reads from a data source (SQLite DB, file system, API)
2. Processes new data since the last run (incremental via cursor in `state.json`)
3. Writes structured Markdown files into the Obsidian vault
4. Updates its cursor for the next run

Extractors are **idempotent** -- running them multiple times produces the same output (deduplication is built in). All support `--dry-run` for testing.

### Data Sources

| Extractor | Source | Frequency | Notes |
|-----------|--------|-----------|-------|
| `screenpipe_extractor` | Screenpipe SQLite DB | 5 min | OCR + audio, 85% dedup threshold |
| `outlook_extractor` | Outlook local DB | 5 min | Classic Outlook for Mac |
| `graph_email_extractor` | Microsoft Graph API | 5 min | New Outlook / M365 cloud |
| `graph_calendar_extractor` | Microsoft Graph API | 5 min | Cloud calendar events |
| `mail_app_extractor` | macOS Mail.app (AppleScript) | 5 min | Zero config, works with any email provider |
| `calendar_app_extractor` | macOS Calendar.app (AppleScript) | 5 min | Zero config |
| `onedrive_extractor` | OneDrive sync folder | 15 min | Converts docx/pptx/pdf/xlsx via pandoc/pdfplumber |

### Vault Structure

```
Obsidian Vault/
├── 00_inbox/              # Emails
│   └── YYYY/MM/DD/
│       ├── _index.md      # Daily email digest
│       └── Subject_ID.md  # Individual emails
├── 10_meetings/           # Calendar + meeting audio
│   └── YYYY/MM/DD/
│       ├── calendar.md    # Calendar events for the day
│       └── audio.md       # Transcribed meeting audio
├── 20_teams-chat/         # Teams messages (from screen OCR)
│   └── YYYY/MM/DD/
│       └── teams.md
├── 40_slides/             # Converted presentations
├── 50_knowledge/          # Converted documents
├── 85_activity/           # Full screen activity timeline
│   └── YYYY/MM/DD/
│       └── daily.md       # Hourly OCR + audio summary
└── _context/              # Auto-generated AI summaries (gitignored)
    ├── today.md
    ├── this_week.md
    ├── recent_emails.md
    └── upcoming.md
```

## Memory Layer

### SQLite FTS5 Index

The index (`config/memory.db`) provides sub-millisecond full-text search across the entire vault. It uses SQLite's FTS5 extension with Porter stemming and Unicode tokenization.

**Schema:**

```sql
documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE,     -- vault-relative path
    title       TEXT,            -- extracted from frontmatter, heading, or filename
    source_type TEXT,            -- email, meetings, teams, activity, knowledge, slides
    content     TEXT,            -- full Markdown content
    created_at  TEXT,            -- ISO 8601
    modified_at TEXT,            -- ISO 8601
    tier        TEXT,            -- hot, warm, cold
    mtime_ns    INTEGER          -- file modification time (for incremental sync)
)
```

**Search ranking:** Results are ranked by FTS5 relevance, then boosted by tier -- hot documents score 2x, warm 1x, cold 0.5x. This means recent content surfaces first without excluding older material.

### Indexer (`src/memory/indexer.py`)

Runs every 5 minutes via launchd. On each run it:

1. Scans the vault for all `.md` files
2. Compares each file's `mtime_ns` against the stored value
3. Skips unchanged files (typically 99%+ of files on incremental runs)
4. Parses frontmatter and extracts title, dates, source type
5. Upserts into the SQLite index
6. Removes index entries for deleted files
7. Reclassifies all documents into hot/warm/cold tiers
8. Regenerates context files

A full reindex of 100k+ documents takes about 30-60 seconds. Incremental runs (only changed files) take under 1 second.

### Hot / Warm / Cold Tiers

Documents are classified by age:

| Tier | Age | Search Boost | Purpose |
|------|-----|-------------|---------|
| Hot | 0-7 days | 2.0x | Current work, today's context |
| Warm | 7-90 days | 1.0x | Recent history, still relevant |
| Cold | 90+ days | 0.5x | Archive, searchable but deprioritized |

Tier boundaries are configurable in `config.yaml` under `memory.tiers`. Reclassification happens on every indexer run so documents naturally age from hot to warm to cold.

### Context Generator (`src/memory/context.py`)

Produces AI-optimized Markdown files in `_context/`:

| File | Contents | Update frequency |
|------|----------|-----------------|
| `today.md` | Today's meetings, emails, activity, Teams | Every 5 min |
| `this_week.md` | Rolling 7-day summary by source type and date | Every 5 min |
| `recent_emails.md` | Last 50 emails with previews | Every 5 min |
| `upcoming.md` | Next 7 days of calendar events | Every 5 min |

These files pull from the SQLite index (fast) rather than re-scanning the vault. They only write to disk when content actually changes (avoiding unnecessary Obsidian sync churn).

## Interface Layer

### CLI (`python3 -m src.memory.cli`)

Direct access to the index from the terminal:

```bash
search "query"           # FTS5 search with tier boosting
recent --hours 24        # Documents from last N hours
meetings --date YYYY-MM-DD  # Meetings for a specific date
stats                    # Index statistics by type and tier
reindex [--full]         # Rebuild index
```

### Cursor Integration

No special integration needed. Add the vault (or `_context/`) to your workspace and reference files with `@`:

- `@_context/today.md` -- full day context
- `@_context/this_week.md` -- week summary
- `@00_inbox/2026/02/19/_index.md` -- specific email digest

For programmatic access from Cursor's terminal, use the CLI.

### Dashboard (`http://localhost:8765`)

FastAPI web app for monitoring, control, and future agent interactions. Provides:
- Extractor status and health
- Launchd agent management
- Privacy mode toggle
- File browser with Markdown preview
- Log viewer

### ChatGPT / Other LLMs

Upload context files as attachments, or use the CLI to generate targeted context:

```bash
python3 -m src.memory.cli search "project alpha" 2>/dev/null
```

## Agent Integration Pattern (Future)

MemoryOS is designed as the data layer for AI agents. The integration model:

```
┌──────────┐     read      ┌──────────────┐
│  Agent   │ ◄──────────── │ _context/*.md │
│(Clawdbot,│               └──────────────┘
│ custom)  │     query     ┌──────────────┐
│          │ ◄──────────── │   CLI / API   │
│          │               └──────────────┘
│          │     propose   ┌──────────────┐
│          │ ──────────── ►│  Dashboard   │
│          │               │  Action Queue │
└──────────┘               └──────┬───────┘
                                  │ review
                           ┌──────▼───────┐
                           │    Human     │
                           │  (approve /  │
                           │   reject)    │
                           └──────────────┘
```

### Read

Agents read `_context/` files for situational awareness. These files are plain Markdown -- any agent framework that can read files can use them.

### Query

For specific questions, agents import `src.memory.index.MemoryIndex` directly or call the CLI. The Python API:

```python
from src.memory.index import MemoryIndex
from src.common.config import load_config

cfg = load_config()
with MemoryIndex(cfg["memory"]["index_db"]) as idx:
    results = idx.search("quarterly budget", source_type="email", limit=10)
    recent = idx.get_recent(hours=24, source_type="meetings")
```

### Act (Future)

Agents propose actions through the dashboard API. Planned endpoints:

- `POST /api/actions/propose` -- agent submits a proposed action (draft email, Teams reply, calendar event)
- `GET /api/actions/pending` -- list pending actions awaiting human review
- `POST /api/actions/{id}/approve` -- human approves an action
- `POST /api/actions/{id}/reject` -- human rejects an action

This ensures **human-in-the-loop** control. No agent takes action without explicit approval.

### Approve

The dashboard will show a queue of pending agent actions. The human reviews each one and approves, edits, or rejects. This is not built yet -- it's the next major feature after the core memory system is stable.

## Roadmap

### Phase 1: Core (Current)

- [x] Data collection: 7 extractors covering screen, audio, email, calendar, documents
- [x] Obsidian vault output with structured Markdown
- [x] Incremental processing with cursor-based state management
- [x] Privacy controls (flag file, WiFi-based, work hours, app correlation)
- [x] Web dashboard for monitoring
- [x] Launchd automation (5-15 min cycles)
- [x] SQLite FTS5 index with hot/warm/cold tiers
- [x] Context file generation for AI tools
- [x] CLI for querying the index

### Phase 2: Semantic Search

- [ ] Vector embeddings via `sentence-transformers` (local, no API key)
- [ ] Embedding storage in SQLite (BLOB column or `sqlite-vec` extension)
- [ ] Cosine similarity search alongside FTS5 keyword search
- [ ] Hybrid ranking: combine FTS5 relevance + embedding similarity + tier boost
- [ ] Incremental embedding updates (only embed new/changed documents)

This upgrades keyword search to meaning-based search. "meetings about the budget" will find documents containing "financial planning" or "Q3 spend review" even if the word "budget" never appears.

### Phase 3: Agent Framework

- [ ] Dashboard action queue (propose / approve / reject)
- [ ] Agent SDK: Python base class for building agents on top of MemoryOS
- [ ] Meeting prep agent: before each meeting, generates a briefing from recent emails and past meeting notes about the same people/topics
- [ ] Email draft agent: suggests replies based on email history and context
- [ ] Task tracking agent: extracts action items from meetings and emails, tracks completion
- [ ] Notification agent: alerts on important emails, calendar conflicts, missed follow-ups

### Phase 4: Cross-Device & Collaboration

- [ ] Sync index across machines (via vault sync)
- [ ] Shared team vaults (selective sharing of non-private data)
- [ ] Mobile read access (Obsidian mobile + context files)
