# AI Agent Portable Memory

> One install. Any AI agent. Memory that follows you everywhere.

Install once, set one API key, and your context is automatically saved and restored — across **Claude Code, GitHub Copilot CLI, Codex, Cursor, Windsurf**, or any agent that uses the Claude Code hook protocol.

No per-agent config. No separate commands per tool. One unified memory store.

---

## The Problem

Every AI session starts blank. You lose:
- What you were building
- Decisions you already made
- Where you stopped

Copy-pasting raw chat logs wastes too many context tokens and is unmanageable.

## The Solution

A portable memory store that:
1. **Auto-detects** which agent is running (Claude, Copilot, Codex, Cursor…)
2. **Compresses** your session into a structured, token-efficient summary using an LLM
3. **Stores it** locally (SQLite) or in the cloud (Supabase, S3/R2, Firebase)
4. **Injects it** automatically at the start of your next session in any agent

---

## Install

```bash
pip install ai-agent-portable-memory
```

Then install hooks and set one API key — that's it:

```bash
memory install                        # hooks every agent on this machine

export ANTHROPIC_API_KEY=sk-ant-...   # Claude / Sonnet (auto-detected)
# OR
export OPENAI_API_KEY=sk-...          # GPT models
# OR
export GROQ_API_KEY=gsk_...           # Groq (free tier)
# Ollama needs no key — runs locally
```

Restart your AI agent. Memory is now live for **all agents** automatically.

For cloud storage backends:

```bash
pip install "ai-agent-portable-memory[supabase]"   # Supabase
pip install "ai-agent-portable-memory[s3]"         # AWS S3 or Cloudflare R2
pip install "ai-agent-portable-memory[firebase]"   # Firebase Firestore
pip install "ai-agent-portable-memory[all]"        # Everything
```

---

## Quick Start

### 1. Install hooks and set one API key

```bash
memory install

export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / GROQ_API_KEY / etc.
```

Restart your AI agent. That's it — sessions are now saved and restored automatically for **every** supported agent (Claude Code, Copilot CLI, Codex, Cursor, Windsurf).

### 2. Save a session manually (optional)

```bash
memory save
```

Paste your chat transcript, press `Ctrl+D`. The LLM compresses it into a structured summary, your editor opens for review, and it's saved.

### 3. List your sessions

```bash
memory list
```

```
╭────┬──────────┬──────────────────────────────────┬───────────┬──────────────────╮
│ #  │ ID       │ Title                            │ Project   │ Updated          │
├────┼──────────┼──────────────────────────────────┼───────────┼──────────────────┤
│ 1  │ a3f2c1b0 │ JWT Auth Middleware — halfway     │ ecommerce │ 2026-11-18 14:30 │
│ 2  │ 9e1d4a72 │ PostgreSQL schema design         │ ecommerce │ 2026-11-17 09:15 │
│ 3  │ 7c8b3f91 │ React dashboard setup            │ dashboard │ 2026-11-16 18:00 │
╰────┴──────────┴──────────────────────────────────┴───────────┴──────────────────╯
```

### 4. Load context into any AI

```bash
memory load 1          # by list number
memory load a3f2c1b0   # by ID prefix
memory load 1 --copy   # copy to clipboard
```

Paste the output at the start of your new AI chat. The AI instantly knows where you left off.

---

## What a session summary looks like

```markdown
## Title
JWT Auth Middleware — halfway through refresh token rotation

## Project
ecommerce

## What we were doing
Building JWT authentication for a React + Node.js e-commerce app.

## What was completed
- Login page UI (src/pages/Login.tsx)
- JWT token generation on backend (src/auth/jwt.js)
- httpOnly cookie setup on login endpoint

## Key decisions made
- httpOnly cookies instead of localStorage — XSS protection
- Access token TTL: 15min, Refresh token TTL: 7 days
- Library: jsonwebtoken v9

## Current state / where we stopped
Halfway through refresh token rotation middleware.
File: src/middleware/refreshToken.js — function written, error handling missing.

## Next steps
1. Finish error handling in refreshToken.js
2. Add protected route wrapper component
3. Test full auth flow with Postman

## Important context
- Backend: Express.js on port 3001
- Frontend: React + Vite on port 5173
- DB: PostgreSQL with Prisma ORM
- Auth route prefix: /api/auth

## Tags
react, nodejs, jwt, auth, express, postgresql
```

**Compact. Structured. Ready to paste.**

---

## Configuration

Create a `.env` file or set environment variables:

```env
# Storage backend (default: sqlite)
MEMORY_BACKEND=sqlite

# LLM provider for compression (default: openai)
MEMORY_LLM_PROVIDER=openai

# LLM API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

### Storage Backends

#### SQLite (default — no setup required)

```env
MEMORY_BACKEND=sqlite
SQLITE_DB_PATH=~/.ai-memory/memory.db   # optional, this is the default
```

#### Supabase

```env
MEMORY_BACKEND=supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your-anon-key
```

Run this SQL in your Supabase project:

```sql
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL DEFAULT 'default',
    title       TEXT NOT NULL,
    summary     TEXT NOT NULL,
    tags        JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON sessions USING GIN (to_tsvector('english', title || ' ' || summary));
```

#### Cloudflare R2 / AWS S3

```env
MEMORY_BACKEND=s3
S3_BUCKET_NAME=my-memory-bucket
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# For Cloudflare R2 (add this):
S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
```

#### Firebase Firestore

```env
MEMORY_BACKEND=firebase
FIREBASE_CREDENTIALS_PATH=./serviceAccountKey.json
FIREBASE_PROJECT_ID=my-firebase-project
```

---

## Auto-Save Claude Code Sessions (Optional)

Run once after installing:

```bash
memory setup --claude-hook       # save on session exit
memory setup --compact-hook      # save on context compaction (recommended)
```

**`--claude-hook`** saves when you exit Claude Code.

**`--compact-hook`** saves automatically whenever Claude Code compacts the context (when the conversation gets long). This acts as a natural mid-session checkpoint — you never lose progress even in very long sessions.

Use both together for full coverage:
```bash
memory setup --claude-hook
memory setup --compact-hook
```

To disable:
```bash
memory setup --remove-claude-hook
memory setup --remove-compact-hook
```

---

## Save from Any AI Tool

```bash
memory save                          Paste transcript manually
memory save --from-claude            Auto-read last Claude Code session
memory save --from-chatgpt           Auto-read ChatGPT export (conversations.json)
memory save --from-cursor            Auto-read last Cursor session
memory save --from-windsurf          Auto-read last Windsurf session
memory save --from-file chat.txt     Read from any plain text or JSON file
memory save --from-claude --session <id>  Specific session by ID
```

---

## CLI Reference

```
memory list                      List all sessions
memory list --project myapp      Filter by project

memory load <id>                 Print session context
memory load <id> --copy          Copy context to clipboard

memory save                      Save new session (paste transcript)
memory save --provider anthropic Use a specific LLM provider
memory save --project myapp      Set project name
memory save --no-edit            Skip editor review

memory search <query>            Search sessions by keyword
memory search auth --project api Filter search by project

memory delete <id>               Delete a session

memory setup --claude-hook         Auto-save on session exit
memory setup --compact-hook        Auto-save on context compaction
memory setup --remove-claude-hook  Disable exit auto-save
memory setup --remove-compact-hook Disable compaction auto-save

memory config                    Show current configuration
```

---

## How it works

```
Your chat  →  memory save  →  LLM compresses  →  You review  →  Stored
                                                                    ↓
New AI chat  ←  paste  ←  memory load  ←  compact summary  ←  Retrieved
```

---

## Contributing

PRs welcome. To add a new storage backend, implement the `BaseBackend` interface in `memory/backends/base.py`.

## License

MIT

---

## Codex support

Codex support is available alongside the existing Claude Code integration:

```bash
memory codex setup                       # install Codex save/load hooks
memory codex save --no-edit               # save this workspace's latest Codex session
memory codex load                         # load this project's latest context
memory codex checkpoint --summary-file memory-summary.md  # store a Codex-written summary
```

From the source checkout, use `python3 -m memory` in place of `memory`.
After setup, restart Codex and review and trust the hooks using `/hooks`.
See [CODEX.md](CODEX.md) for project-scoped setup, direct checkpoints without an
extra API call, removal, and environments where local hooks are unavailable.

---

## GitHub Copilot CLI support

GitHub Copilot CLI support is available as a separate additive integration:

```bash
memory copilot setup                     # install hooks in ~/.copilot/settings.json (user scope)
memory copilot setup --project-local     # install hooks in ./.claude/settings.json (project scope)
memory copilot load                      # load this project's latest context
memory copilot save                      # save manually
memory copilot checkpoint --summary-file memory-summary.md  # store a Copilot-written summary directly
memory copilot setup --remove            # remove hooks
```

Hooks use Copilot CLI's `SessionStart` and `Stop` events:
- **SessionStart** — injects the latest project memory as additional context at the start of each session.
- **Stop** — saves a session checkpoint whenever the agent stops.

For project-scoped setup (checked into the repo), use `--project-local`. This writes to `.claude/settings.json`, which Copilot CLI reads automatically.

From the source checkout, use `python3 -m memory` in place of `memory`.
