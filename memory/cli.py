import os
import sys
import json
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from rich import box

console = Console()


@click.group()
def cli():
    """AI Agent Portable Memory — carry your context across any AI."""


@cli.command()
@click.option("--project", "-p", default=None, help="Filter by project name")
def list(project):
    """List all saved sessions."""
    from .core import list_sessions

    sessions = list_sessions(project=project)
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("ID", style="dim", width=8)
    table.add_column("Title", min_width=30)
    table.add_column("Project", style="cyan")
    table.add_column("Tags", style="green")
    table.add_column("Updated", style="dim")

    for i, s in enumerate(sessions, 1):
        table.add_row(
            str(i),
            s.id[:8],
            s.title,
            s.project,
            ", ".join(s.tags[:3]),
            s.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    console.print("\n[dim]Use [bold]memory load <ID>[/bold] to load a session context[/dim]")


@cli.command()
@click.argument("session_id")
@click.option("--copy", "-c", is_flag=True, help="Copy context to clipboard")
def load(session_id, copy):
    """Load a session and print its context (ready to paste into any AI)."""
    from .core import load_session, list_sessions, get_backend

    session = None
    if session_id.isdigit():
        sessions = list_sessions()
        idx = int(session_id) - 1
        if 0 <= idx < len(sessions):
            session = load_session(sessions[idx].id)
    else:
        backend = get_backend()
        backend.init()
        all_sessions = backend.list()
        matches = [s for s in all_sessions if s.id.startswith(session_id)]
        if matches:
            session = backend.get(matches[0].id)

    if not session:
        console.print(f"[red]Session not found: {session_id}[/red]")
        sys.exit(1)

    context = _format_context(session)

    console.print(Panel(
        Markdown(context),
        title=f"[bold cyan]{session.title}[/bold cyan]",
        subtitle=f"[dim]Project: {session.project} | ID: {session.id[:8]}[/dim]",
        border_style="cyan",
    ))

    if copy:
        _copy_to_clipboard(context)
        console.print("\n[green]Copied to clipboard![/green]")
    else:
        console.print("\n[dim]Tip: use [bold]--copy[/bold] to copy to clipboard[/dim]")


@cli.command()
@click.option("--provider", "-P", default=None,
              help="LLM provider: openai | anthropic | groq (overrides MEMORY_LLM_PROVIDER env)")
@click.option("--model", "-m", default=None, help="Model override")
@click.option("--project", "-p", default=None, help="Project name")
@click.option("--no-edit", is_flag=True, help="Skip editor review and save directly")
@click.option("--from-claude", is_flag=True, help="Auto-read last Claude Code session transcript")
@click.option("--from-chatgpt", is_flag=True, help="Auto-read ChatGPT export (conversations.json)")
@click.option("--from-cursor", is_flag=True, help="Auto-read last Cursor session transcript")
@click.option("--from-windsurf", is_flag=True, help="Auto-read last Windsurf session transcript")
@click.option("--from-file", default=None, type=click.Path(exists=True),
              help="Read transcript from a plain text or JSON file")
@click.option("--session", "source_session_id", default=None,
              help="Specific session ID (use with --from-claude or --from-chatgpt)")
def save(provider, model, project, no_edit,
         from_claude, from_chatgpt, from_cursor, from_windsurf, from_file,
         source_session_id):
    """Save a session from any AI tool — paste, file, or auto-read."""
    from .compressor import compress, _parse_response
    from .editor import open_in_editor
    from .core import create_session

    resolved_provider = provider or os.environ.get("MEMORY_LLM_PROVIDER", "openai")

    sources = [from_claude, from_chatgpt, from_cursor, from_windsurf, from_file]
    if sum(bool(s) for s in sources) > 1:
        console.print("[red]Use only one --from-* option at a time.[/red]")
        sys.exit(1)

    transcript = ""

    if from_claude:
        transcript = _read_claude_transcript(source_session_id)
        if not transcript:
            console.print("[red]No Claude Code session found in ~/.claude/projects/[/red]")
            sys.exit(1)
        console.print(f"[dim]Claude Code transcript loaded ({len(transcript)} chars)[/dim]")

    elif from_chatgpt:
        transcript = _read_chatgpt_export(source_session_id)
        if not transcript:
            console.print("[red]No ChatGPT export found. Download from chat.openai.com → Settings → Export.[/red]")
            sys.exit(1)
        console.print(f"[dim]ChatGPT transcript loaded ({len(transcript)} chars)[/dim]")

    elif from_cursor:
        transcript = _read_cursor_transcript()
        if not transcript:
            console.print("[red]No Cursor session found.[/red]")
            sys.exit(1)
        console.print(f"[dim]Cursor transcript loaded ({len(transcript)} chars)[/dim]")

    elif from_windsurf:
        transcript = _read_windsurf_transcript()
        if not transcript:
            console.print("[red]No Windsurf session found.[/red]")
            sys.exit(1)
        console.print(f"[dim]Windsurf transcript loaded ({len(transcript)} chars)[/dim]")

    elif from_file:
        transcript = _read_file_transcript(from_file)
        if not transcript:
            console.print(f"[red]Could not read transcript from {from_file}[/red]")
            sys.exit(1)
        console.print(f"[dim]File transcript loaded ({len(transcript)} chars)[/dim]")

    else:
        console.print("[bold cyan]Paste your chat transcript below.[/bold cyan]")
        console.print("[dim]When done, press Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows).[/dim]\n")
        try:
            transcript = sys.stdin.read()
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            return

    if not transcript.strip():
        console.print("[red]No transcript provided.[/red]")
        sys.exit(1)

    console.print(f"\n[dim]Compressing with {resolved_provider}...[/dim]")
    try:
        compressed = compress(transcript, provider=resolved_provider, model=model)
    except Exception as e:
        console.print(f"[red]Compression failed: {e}[/red]")
        sys.exit(1)

    summary = compressed.summary
    resolved_project = project or compressed.project

    if not no_edit:
        console.print("\n[bold]Opening in editor for review...[/bold] [dim](save and close to continue)[/dim]")
        summary = open_in_editor(summary)

    parsed = _parse_response(summary)

    session = create_session(
        title=parsed.title,
        project=resolved_project,
        summary=summary,
        tags=parsed.tags,
    )

    console.print("\n[green]Session saved![/green]")
    console.print(f"  ID:      [bold]{session.id[:8]}[/bold]")
    console.print(f"  Title:   {session.title}")
    console.print(f"  Project: {session.project}")
    console.print(f"  Tags:    {', '.join(session.tags)}")


@cli.command()
@click.argument("query")
@click.option("--project", "-p", default=None, help="Filter by project")
def search(query, project):
    """Search sessions by keyword."""
    from .core import search_sessions

    sessions = search_sessions(query=query, project=project)
    if not sessions:
        console.print(f"[yellow]No sessions matched: {query}[/yellow]")
        return

    console.print(f"[bold]Found {len(sessions)} session(s):[/bold]\n")
    for s in sessions:
        console.print(f"  [bold cyan]{s.id[:8]}[/bold cyan]  {s.title}  [dim]({s.project})[/dim]")
    console.print("\n[dim]Use [bold]memory load <ID>[/bold] to load context[/dim]")


@cli.command()
@click.argument("session_id")
@click.confirmation_option(prompt="Are you sure you want to delete this session?")
def delete(session_id):
    """Delete a session."""
    from .core import delete_session
    delete_session(session_id)
    console.print(f"[green]Deleted session {session_id}[/green]")


@cli.command()
def config():
    """Show current configuration and how to set it up."""
    backend = os.environ.get("MEMORY_BACKEND", "sqlite")
    provider = os.environ.get("MEMORY_LLM_PROVIDER", "openai")

    console.print(Panel(
        f"""[bold]Current Configuration[/bold]

[cyan]Storage backend:[/cyan]  {backend}
[cyan]LLM provider:[/cyan]     {provider}

[bold]Configure via environment variables or a .env file:[/bold]

  MEMORY_BACKEND=sqlite|supabase|s3|firebase
  MEMORY_LLM_PROVIDER=openai|anthropic|groq|gemini|grok|mistral|ollama|deepseek

  [dim]# Storage backends[/dim]
  SQLITE_DB_PATH=~/.ai-memory/memory.db
  SUPABASE_URL=https://xxx.supabase.co
  SUPABASE_KEY=your-anon-key
  S3_BUCKET_NAME=my-bucket
  S3_ENDPOINT_URL=https://xxx.r2.cloudflarestorage.com  [dim]# R2 only[/dim]
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  FIREBASE_CREDENTIALS_PATH=./serviceAccountKey.json
  FIREBASE_PROJECT_ID=my-project

  [dim]# LLM provider API keys[/dim]
  OPENAI_API_KEY=sk-...
  ANTHROPIC_API_KEY=sk-ant-...
  GROQ_API_KEY=gsk_...
  GEMINI_API_KEY=...          [dim]# or GOOGLE_API_KEY[/dim]
  XAI_API_KEY=...             [dim]# Grok (xAI)[/dim]
  MISTRAL_API_KEY=...
  DEEPSEEK_API_KEY=...
  OLLAMA_HOST=http://localhost:11434  [dim]# Ollama (local, no key needed)[/dim]

  [dim]# Install extras[/dim]
  pip install "ai-agent-portable-memory[gemini]"
  pip install "ai-agent-portable-memory[mistral]"
  pip install "ai-agent-portable-memory[all]"
""",
        border_style="cyan",
        title="memory config",
    ))


# ── Transcript readers ────────────────────────────────────────────────────────

def _read_claude_transcript(session_id: str = None) -> str:
    """Read from ~/.claude/projects/ (Claude Code)."""
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return ""

    if session_id:
        for project_dir in projects_dir.iterdir():
            candidate = project_dir / f"{session_id}.jsonl"
            if candidate.exists():
                return _parse_claude_jsonl(candidate)
        return ""

    all_transcripts = list(projects_dir.glob("*/*.jsonl"))
    if not all_transcripts:
        return ""
    latest = max(all_transcripts, key=lambda p: p.stat().st_mtime)
    return _parse_claude_jsonl(latest)


def _parse_claude_jsonl(path: Path) -> str:
    messages = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = entry.get("type")
            content = entry.get("message", {}).get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            if role == "user" and content.strip():
                messages.append(f"User: {content.strip()}")
            elif role == "assistant" and content.strip():
                messages.append(f"Assistant: {content.strip()}")
    return "\n\n".join(messages)


def _read_chatgpt_export(conversation_id: str = None) -> str:
    """Read from ChatGPT's conversations.json export (~/.chatgpt/ or ~/Downloads/)."""
    candidates = [
        Path.home() / ".chatgpt" / "conversations.json",
        Path.home() / "Downloads" / "conversations.json",
        Path.home() / "conversations.json",
    ]
    export_file = next((p for p in candidates if p.exists()), None)
    if not export_file:
        return ""

    try:
        with open(export_file) as f:
            data = json.load(f)
    except Exception:
        return ""

    # data is a list of conversations
    if not isinstance(data, list) or not data:
        return ""

    if conversation_id:
        conv = next((c for c in data if c.get("id") == conversation_id), None)
    else:
        conv = data[-1]  # most recent

    if not conv:
        return ""

    messages = []
    mapping = conv.get("mapping", {})
    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "")
        parts = msg.get("content", {}).get("parts", [])
        text = " ".join(str(p) for p in parts if isinstance(p, str)).strip()
        if role == "user" and text:
            messages.append(f"User: {text}")
        elif role == "assistant" and text:
            messages.append(f"Assistant: {text}")

    return "\n\n".join(messages)


def _read_cursor_transcript() -> str:
    """Read from Cursor's local storage (VS Code-based, stores in SQLite workspaceStorage)."""
    # Cursor stores chat history in IndexedDB / SQLite under its app data
    candidates = [
        Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage",  # macOS
        Path.home() / ".config" / "Cursor" / "User" / "workspaceStorage",  # Linux
        Path.home() / "AppData" / "Roaming" / "Cursor" / "User" / "workspaceStorage",  # Windows
    ]
    storage_dir = next((p for p in candidates if p.exists()), None)
    if not storage_dir:
        return ""

    # Find the most recently modified .vscdb file
    db_files = list(storage_dir.glob("*/*.vscdb"))
    if not db_files:
        return ""

    latest_db = max(db_files, key=lambda p: p.stat().st_mtime)
    return _read_vscdb_chat(latest_db)


def _read_windsurf_transcript() -> str:
    """Read from Windsurf's local storage (also VS Code-based)."""
    candidates = [
        Path.home() / "Library" / "Application Support" / "Windsurf" / "User" / "workspaceStorage",  # macOS
        Path.home() / ".config" / "Windsurf" / "User" / "workspaceStorage",  # Linux
        Path.home() / "AppData" / "Roaming" / "Windsurf" / "User" / "workspaceStorage",  # Windows
    ]
    storage_dir = next((p for p in candidates if p.exists()), None)
    if not storage_dir:
        return ""

    db_files = list(storage_dir.glob("*/*.vscdb"))
    if not db_files:
        return ""

    latest_db = max(db_files, key=lambda p: p.stat().st_mtime)
    return _read_vscdb_chat(latest_db)


def _read_vscdb_chat(db_path: Path) -> str:
    """Extract chat messages from a VS Code-style workspaceStorage .vscdb SQLite file."""
    import sqlite3
    messages = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Cursor/Windsurf store chat in the 'ItemTable' key-value store
        cursor.execute("SELECT key, value FROM ItemTable WHERE key LIKE '%chat%' OR key LIKE '%conversation%'")
        rows = cursor.fetchall()
        conn.close()

        for _, value in rows:
            try:
                data = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
            # Handle various storage shapes
            if isinstance(data, list):
                for item in data:
                    _extract_vscdb_message(item, messages)
            elif isinstance(data, dict):
                _extract_vscdb_message(data, messages)
    except Exception:
        pass

    return "\n\n".join(messages)


def _extract_vscdb_message(item: dict, messages: list) -> None:
    role = item.get("role", item.get("type", ""))
    text = item.get("content", item.get("text", item.get("message", "")))
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if isinstance(t, str))
    text = str(text).strip()
    if not text:
        return
    if role in ("user", "human"):
        messages.append(f"User: {text}")
    elif role in ("assistant", "ai", "bot"):
        messages.append(f"Assistant: {text}")


def _read_file_transcript(file_path: str) -> str:
    """Read a plain text or JSON transcript file."""
    path = Path(file_path)
    try:
        with open(path) as f:
            content = f.read()
    except Exception:
        return ""

    # Try JSON first (ChatGPT export, custom format)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            messages = []
            for item in data:
                if isinstance(item, dict):
                    role = item.get("role", item.get("type", ""))
                    text = item.get("content", item.get("text", ""))
                    if isinstance(text, list):
                        text = " ".join(str(t) for t in text if isinstance(t, str))
                    text = str(text).strip()
                    if role in ("user", "human") and text:
                        messages.append(f"User: {text}")
                    elif role in ("assistant", "ai", "bot") and text:
                        messages.append(f"Assistant: {text}")
            if messages:
                return "\n\n".join(messages)
        # Fall through to raw text if JSON parse didn't yield messages
    except json.JSONDecodeError:
        pass

    return content  # plain text — return as-is


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(session) -> str:
    return f"""# Session Context

**Title:** {session.title}
**Project:** {session.project}
**Saved:** {session.updated_at.strftime('%Y-%m-%d %H:%M')} UTC

---

{session.summary}

---
*Loaded from AI Agent Portable Memory — https://github.com/Sidd11145/ai-agent-portable-memory*
"""


def _copy_to_clipboard(text: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    elif sys.platform == "win32":
        subprocess.run(["clip"], input=text.encode(), check=True)
    else:
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
        except FileNotFoundError:
            subprocess.run(["xdotool", "type", "--clearmodifiers", text], check=True)
