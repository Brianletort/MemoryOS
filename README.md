# MemoryOS

A local-first memory pipeline for macOS that captures everything you do, see, hear, and read -- and writes it as structured Markdown into an Obsidian vault. Then indexes it all for instant search and AI-powered querying from Cursor, ChatGPT, or any tool that can read files.

**This is the data collection and memory layer.** It gives you (and your AI tools) full context about your meetings, emails, screen activity, documents, and conversations -- all searchable, all local, all yours.

## What Gets Captured

| Source | How | Output |
|--------|-----|--------|
| Screen activity | OCR every ~2 seconds via Screenpipe | `85_activity/YYYY/MM/DD/daily.md` |
| Your voice | Microphone transcription (Screenpipe) | `10_meetings/YYYY/MM/DD/audio.md` |
| Meeting audio | System audio via BlackHole loopback | `10_meetings/YYYY/MM/DD/audio.md` |
| Emails | Mail.app, Outlook Classic, or Microsoft Graph API | `00_inbox/YYYY/MM/DD/*.md` |
| Calendar | Calendar.app, Outlook, or Graph API | `10_meetings/YYYY/MM/DD/calendar.md` |
| Teams chat | Screen OCR when Teams is active | `20_teams-chat/YYYY/MM/DD/teams.md` |
| Documents | OneDrive files (docx, pptx, pdf, xlsx) | `40_slides/` and `50_knowledge/` |

## System Monitoring

MemoryOS includes a built-in **watchdog** that monitors every component and sends macOS notifications when something breaks or recovers.

| Check | What it monitors | Alert threshold |
|-------|-----------------|----------------|
| Screen Recording | Screenpipe vision pipeline health and frame rate | Stale >30 min or >95% drop rate |
| Audio Transcription | Screenpipe audio pipeline health | Stale >30 min |
| Email Pipeline | Freshness of newest file in `00_inbox/` | No new files in >4 hours |
| Meeting Transcripts | Freshness of newest file in `10_meetings/` | No new files in >8 hours |
| Activity Tracking | Freshness of newest file in `85_activity/` | No new files in >2 hours |
| Teams Chat | Freshness of newest file in `20_teams-chat/` | No new files in >24 hours |
| Dashboard | HTTP health check on port 8765 | Unreachable |
| Background Agents | All expected launchd agents loaded | Any agent missing |

Notifications fire only on **state transitions** (healthy to degraded, or recovered), not repeatedly. Check status anytime:

```bash
python3 src/monitor/watchdog.py --status
```

The dashboard also exposes watchdog status at `GET /api/watchdog`.

## Prerequisites

You need a Mac running macOS 13+ (Ventura or later). Everything runs locally.

### Required

1. **Homebrew** -- macOS package manager

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Python 3.9+**

   ```bash
   brew install python@3.12
   ```

3. **Screenpipe** -- screen recording + audio transcription engine

   Install from [screenpipe.com](https://screenpipe.com) or:
   ```bash
   brew install screenpipe
   ```
   Screenpipe must be running for screen/audio capture to work. It creates a local SQLite database at `~/.screenpipe/db.sqlite`.

4. **Obsidian** -- knowledge base where all Markdown output lands

   Download from [obsidian.md](https://obsidian.md). Create a new vault or point MemoryOS at an existing one.

5. **pandoc** -- converts Word documents to Markdown

   ```bash
   brew install pandoc
   ```

### Optional

6. **BlackHole 2ch** -- virtual audio driver for capturing meeting audio (other participants' voices, not just yours)

   ```bash
   brew install blackhole-2ch
   ```

   After installing, configure the audio routing:
   1. Open **Audio MIDI Setup** (search in Spotlight)
   2. Click **+** at bottom left, select **Create Multi-Output Device**
   3. Check both your speakers (e.g. "MacBook Pro Speakers") and **BlackHole 2ch**
   4. Go to **System Settings > Sound > Output** and select the new **Multi-Output Device**
   5. In Screenpipe settings, add **BlackHole 2ch** as an input audio device

   This routes system audio (Zoom, Teams, etc.) through BlackHole so Screenpipe can transcribe it.

7. **Microsoft Graph API** -- for richer email/calendar data from Microsoft 365

   If you use Outlook or Microsoft 365 and want full HTML email bodies and detailed calendar data, you can connect via the Graph API. This requires registering an Azure AD application:

   1. Go to [Azure Portal > App Registrations](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
   2. Click **New registration**
   3. Name: `MemoryOS` (or anything you like)
   4. Supported account types: **Accounts in any organizational directory and personal Microsoft accounts**
   5. Redirect URI: leave blank (we use device code flow)
   6. After creation, copy the **Application (client) ID**
   7. Go to **API permissions > Add a permission > Microsoft Graph > Delegated permissions**
   8. Add: `Mail.Read`, `Calendars.Read`
   9. Paste the client ID into `config/config.yaml` under `graph.client_id`

   If you don't need Graph API, skip this. Mail.app and Calendar.app work out of the box with zero configuration.

## Installation

```bash
git clone https://github.com/Brianletort/MemoryOS.git
cd MemoryOS
./scripts/setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all Python dependencies
- Install system dependencies via Homebrew
- Create `config/config.yaml` from the template

## Configuration

Edit `config/config.yaml` with your paths:

```bash
$EDITOR config/config.yaml
```

At minimum, set:

```yaml
# REQUIRED: path to your Obsidian vault
obsidian_vault: ~/Documents/Obsidian/MyVault

# Screenpipe database (usually auto-detected)
screenpipe:
  db_path: ~/.screenpipe/db.sqlite
```

### Email & Calendar -- Pick Your Path

**Easy path** (recommended to start): macOS Mail.app + Calendar.app. If your email accounts are configured in **System Settings > Internet Accounts**, you're done. The default config works out of the box.

**Power path**: Microsoft Graph API. Set `graph.client_id` to your Azure app's client ID (see Prerequisites step 7). Gives you full email HTML bodies and detailed calendar event data.

**Outlook Classic**: If you use Outlook for Mac in Classic mode (not "New Outlook"), set `outlook.db_path` to your Outlook SQLite database path. **Note:** The "New Outlook" for Mac uses cloud-only storage and does not write to the local SQLite DB. The Outlook extractor auto-detects this and falls back to the mail_app extractor.

### OneDrive (optional)

Set `onedrive.sync_dir` to your OneDrive sync folder to automatically convert documents to searchable Markdown.

### Environment Variable Overrides

Any config path can be overridden via environment variables:

| Variable | Config key |
|----------|-----------|
| `MEMORYOS_OBSIDIAN_VAULT` | `obsidian_vault` |
| `MEMORYOS_SCREENPIPE_DB` | `screenpipe.db_path` |
| `MEMORYOS_OUTLOOK_DB` | `outlook.db_path` |
| `MEMORYOS_ONEDRIVE_DIR` | `onedrive.sync_dir` |
| `MEMORYOS_STATE_FILE` | `state_file` |
| `MEMORYOS_LOG_DIR` | `log_dir` |

## Getting Started

### 1. Run the smoke test

```bash
./scripts/smoke_test.sh
```

### 2. Run extractors manually to see output

```bash
# Screen activity + audio transcription
python3 src/extractors/screenpipe_extractor.py

# Email (pick one based on your setup)
python3 src/extractors/mail_app_extractor.py
python3 src/extractors/outlook_extractor.py

# Calendar
python3 src/extractors/calendar_app_extractor.py

# OneDrive documents
python3 src/extractors/onedrive_extractor.py
```

All extractors support `--dry-run` to preview output without writing files.

### 3. Build the memory index

```bash
python3 -m src.memory.cli reindex
```

This scans your vault and builds a SQLite full-text search index. It also generates context files in `_context/` for AI tools.

### 4. Install automation

```bash
./scripts/install_launchd.sh
```

This installs 9 macOS launchd agents:

| Agent | Interval | Purpose |
|-------|----------|---------|
| `com.memoryos.screenpipe` | 5 min | Screen OCR + audio transcription extraction |
| `com.memoryos.outlook` | 5 min | Email + calendar from Outlook Classic DB |
| `com.memoryos.mail-app` | 5 min | Email from Mail.app via AppleScript |
| `com.memoryos.calendar-app` | 5 min | Calendar from Calendar.app via AppleScript |
| `com.memoryos.onedrive` | 15 min | Document conversion (docx, pptx, pdf, xlsx) |
| `com.memoryos.indexer` | 5 min | Full-text search index + context file generation |
| `com.memoryos.dashboard` | Continuous | Web dashboard on port 8765 |
| `com.memoryos.watchdog` | 5 min | Health monitoring with macOS notifications |
| `com.memoryos.wifi-monitor` | Continuous | WiFi-based auto-privacy mode |

### 5. Optional: initial backfill

If you have existing email history you want to import:

```bash
# Outlook Classic: import all emails (may take several minutes)
python3 src/extractors/outlook_extractor.py --backfill

# Mail.app: import last year
python3 src/extractors/mail_app_extractor.py --days-back 365

# After backfill, rebuild the index
python3 -m src.memory.cli reindex --full
```

## Using the System

Once running, MemoryOS generates two things: raw Markdown files (detailed, per-item) and context files (summaries optimized for AI consumption). Here's how to use them with different tools.

### With Cursor (Primary IDE)

**Add your vault to the workspace.** Open Cursor, add your Obsidian vault folder (or at minimum, the `_context/` subfolder) to your workspace.

**Reference context files with @:**

- `@_context/today.md` -- "What meetings do I have today? What emails came in?"
- `@_context/this_week.md` -- "Summarize what I worked on this week"
- `@_context/recent_emails.md` -- "Any emails about the budget proposal?"
- `@_context/upcoming.md` -- "What's on my calendar next week?"

**Use the CLI in Cursor's terminal for deeper searches:**

```bash
# Search across everything
python3 -m src.memory.cli search "quarterly review"

# Recent emails only
python3 -m src.memory.cli recent --hours 48 --type email

# Today's meetings
python3 -m src.memory.cli meetings

# Index stats
python3 -m src.memory.cli stats
```

**Tip:** Create a `.cursorrules` file in your project that tells the AI about your memory system:

```
I have a personal memory system in my Obsidian vault. Context files in _context/ are auto-updated every 5 minutes with my meetings, emails, activity, and upcoming events. Reference @_context/today.md for today's context. For deeper searches, use the CLI: python3 -m src.memory.cli search "query"
```

### With ChatGPT Enterprise

- Upload `_context/today.md` or `_context/this_week.md` as attachments for situational context
- Use the CLI to generate targeted context, then copy/paste:
  ```bash
  python3 -m src.memory.cli search "project alpha" 
  ```
- For recurring workflows, create a custom GPT with instructions about your data format

### With AI Agents

MemoryOS is designed as the **data layer** for AI agents:

- **Read**: Agents read `_context/` files for situational awareness, or import `src.memory.index.MemoryIndex` directly for programmatic queries
- **Query**: Agents call the CLI or use the Python API for search
- **Act**: Agents propose actions (draft replies, schedule items) via the dashboard API
- **Approve**: Human-in-the-loop review through the dashboard at `http://localhost:8765`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full agent integration pattern.

## Dashboard

The web dashboard provides monitoring and control:

```
http://localhost:8765
```

- **Overview**: extractor status, launchd agent health, privacy controls
- **Pipeline**: folder health, activity timeline, watchdog status
- **File Browser**: browse and preview vault Markdown files
- **Logs**: live extractor log viewer
- **Settings**: trusted networks, work apps, audio filters
- **Skills**: installed Cursor agent skills

The dashboard runs automatically via launchd. It auto-recovers from port conflicts by killing stale processes on startup. To start manually:

```bash
python3 src/dashboard/app.py
```

### Dashboard API

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Overall system status with extractor health |
| `GET /api/pipeline-health` | Per-folder file counts, freshness, health |
| `GET /api/watchdog` | Latest watchdog check results |
| `GET /api/timeline/{date}` | Hour-by-hour activity for a given date |
| `GET /api/browse?path=` | Browse vault files |
| `GET /api/file?path=` | Preview a vault Markdown file |
| `POST /api/run/{extractor}` | Manually trigger an extractor run |
| `POST /api/agent/{name}/{action}` | Control launchd agents (start/stop/restart) |

## Audio Capture Setup

For useful meeting transcripts, Screenpipe needs to hear **both** sides of a call -- your microphone and the system audio from other participants. By default, Screenpipe only captures your mic input, which means you get your own voice plus ambient noise but very little of what others say.

### 1. Install BlackHole (virtual audio loopback)

```bash
brew install blackhole-2ch
```

### 2. Create a Multi-Output Device

1. Open **Audio MIDI Setup** (Spotlight search)
2. Click **+** at bottom left > **Create Multi-Output Device**
3. Check both your real speakers (e.g. "MacBook Pro Speakers" or your headphones) **and** "BlackHole 2ch"
4. Optionally rename it to "Meeting Audio"

### 3. Route system audio through it

Go to **System Settings > Sound > Output** and select the Multi-Output Device you just created. Teams/Zoom/Meet audio now flows through BlackHole.

### 4. Tell Screenpipe to capture BlackHole

In Screenpipe's settings, add **BlackHole 2ch** as an audio **input** device. Screenpipe will now transcribe the system audio stream -- i.e., what other meeting participants are saying.

### 5. Verify

Join a test call and confirm Screenpipe captures both your voice (from the mic) and the other party's voice (from BlackHole). Check `~/.screenpipe/db.sqlite` with:

```bash
sqlite3 ~/.screenpipe/db.sqlite "SELECT id, timestamp, substr(transcription, 1, 80) FROM audio_transcriptions ORDER BY id DESC LIMIT 10;"
```

### Audio filtering

Audio noise is controlled in `config/config.yaml` under `privacy`:

- **work_apps**: Audio is only kept when one of these apps was on-screen within 2 minutes. Set to your work apps (Teams, Chrome, Cursor, etc.).
- **work_hours_only**: When `true`, audio outside the configured hours is filtered to the collapsible section.
- **min_words**: Fragments shorter than this word count are filtered.

Filtered audio is never deleted -- it goes into a collapsible `<details>` section at the bottom of each audio note.

## Privacy

MemoryOS runs **entirely locally**. No data leaves your machine unless you explicitly upload it.

### Privacy Controls

- **Privacy mode**: Create `config/.privacy_mode` to instantly disable all audio capture
- **WiFi-based auto-privacy**: Automatically enables privacy mode on untrusted networks
- **Work hours filter**: Only capture audio during configured hours
- **Work app correlation**: Only keep audio when work apps are active on screen
- **Minimum word filter**: Discard tiny audio fragments (noise)

Toggle privacy from the dashboard, CLI, or:

```bash
./scripts/privacy_toggle.sh
```

## CLI Reference

```bash
python3 -m src.memory.cli search "query"              # Full-text search
python3 -m src.memory.cli search "query" --type email  # Filter by type
python3 -m src.memory.cli recent --hours 24            # Last 24 hours
python3 -m src.memory.cli recent --type meetings       # Recent meetings
python3 -m src.memory.cli meetings --date 2026-02-19   # Specific date
python3 -m src.memory.cli stats                        # Index statistics
python3 -m src.memory.cli reindex                      # Incremental reindex
python3 -m src.memory.cli reindex --full               # Full reindex
```

Source types: `email`, `meetings`, `teams`, `activity`, `knowledge`, `slides`

### Watchdog CLI

```bash
python3 src/monitor/watchdog.py --status    # Print current health status
python3 src/monitor/watchdog.py --quiet     # Check health without notifications
python3 src/monitor/watchdog.py             # Check health and notify on changes
```

## Logs

- Application: `logs/memoryos.log` (rotating, 5 MB, 3 backups)
- Per-extractor: `logs/*_launchd.log` and `logs/*_launchd.err`
- Dashboard: `logs/dashboard.log`
- Indexer: `logs/indexer.err`
- Watchdog: `logs/watchdog.err`

## Troubleshooting

**Extractors not running?**
1. Check launchd status: `launchctl list | grep memoryos`
2. Check logs: `tail -f logs/memoryos.log`
3. Run manually with `--dry-run` to test
4. Verify paths in `config/config.yaml`

**Missing email data?**
1. If using **New Outlook** for Mac: the local SQLite DB is not updated. The Outlook extractor auto-detects this and logs a warning. Use the mail_app extractor (Mail.app) or Graph API instead.
2. If using **Mail.app**: ensure the app is running and has finished syncing. AppleScript queries can time out during active sync.
3. Check `config/state.json` for cursor positions.
4. Run with `--reset` to reprocess from scratch.

**Audio not capturing?**
1. Verify Screenpipe is running and healthy: `curl http://localhost:3030/health`
2. Check BlackHole is installed: `brew list blackhole-2ch`
3. Verify Audio MIDI Setup has the Multi-Output Device
4. Check privacy mode: `test -f config/.privacy_mode && echo "ON" || echo "OFF"`

**Screen recording degraded?**
1. Check Screenpipe health: `curl -s http://localhost:3030/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['frame_status'], d.get('message',''))"`
2. If frame_status is "stale": try restarting Screenpipe
3. If drop rate is high: toggle Screen Recording permission off/on in **System Settings > Privacy & Security > Screen Recording**
4. High-resolution multi-monitor setups may see higher drop rates

**Dashboard not loading?**
1. Check if it's running: `curl -s http://localhost:8765/api/status`
2. Check for port conflicts: `lsof -i :8765`
3. Restart via launchd: `launchctl kickstart -k gui/$(id -u)/com.memoryos.dashboard`
4. The dashboard auto-kills stale processes on startup, but manual cleanup may be needed after crashes

**Watchdog not alerting?**
1. Check watchdog state: `python3 src/monitor/watchdog.py --status`
2. Check logs: `tail -f logs/watchdog.err`
3. Verify macOS notification permissions for `osascript` in **System Settings > Notifications**
4. Alerts fire only on state transitions, not repeatedly for ongoing issues

**Index empty?**
1. Run `python3 -m src.memory.cli reindex --full`
2. Check vault path in config: `grep obsidian_vault config/config.yaml`

## Project Structure

```
MemoryOS/
├── config/
│   ├── config.yaml.example       # Template (committed)
│   ├── config.yaml               # Your personal config (gitignored)
│   ├── state.json                # Extractor cursors (gitignored)
│   ├── memory.db                 # SQLite FTS5 index (gitignored)
│   └── watchdog_state.json       # Watchdog health state (gitignored)
├── src/
│   ├── extractors/               # Data extraction modules
│   │   ├── screenpipe_extractor.py    # Screen OCR + audio transcription
│   │   ├── outlook_extractor.py       # Outlook Classic SQLite DB
│   │   ├── mail_app_extractor.py      # Mail.app via AppleScript
│   │   ├── calendar_app_extractor.py  # Calendar.app via AppleScript
│   │   └── onedrive_extractor.py      # Document conversion
│   ├── common/                   # Shared utilities
│   │   ├── config.py             # Config loader with env var overrides
│   │   ├── state.py              # Cursor state management (atomic writes)
│   │   ├── markdown.py           # Markdown helpers (sanitize, dedup, write)
│   │   └── outlook_body.py       # Email body extraction via mdimport
│   ├── memory/                   # Memory system
│   │   ├── index.py              # SQLite FTS5 index
│   │   ├── indexer.py            # Vault scanner + context generator
│   │   ├── tier.py               # Hot/warm/cold classification
│   │   ├── context.py            # Context file generator
│   │   └── cli.py                # Command-line interface
│   ├── monitor/                  # System health monitoring
│   │   └── watchdog.py           # Component health checks + macOS notifications
│   └── dashboard/
│       └── app.py                # FastAPI web dashboard (single-file, embedded UI)
├── scripts/
│   ├── setup.sh                  # First-time setup
│   ├── install_launchd.sh        # Install all launchd agents from templates
│   ├── uninstall_launchd.sh      # Remove agents
│   ├── smoke_test.sh             # Test all extractors
│   └── privacy_toggle.sh         # Toggle privacy mode
├── launchd/                      # Agent templates (.plist.template)
├── ARCHITECTURE.md               # System design & roadmap
└── requirements.txt              # Python dependencies
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, including:
- Memory system with hot/warm/cold data tiers
- How the indexer and context generator work
- Agent integration patterns (read/query/act/approve)
- Future roadmap (vector embeddings, semantic search, agent framework)
