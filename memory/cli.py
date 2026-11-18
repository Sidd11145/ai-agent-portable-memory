import sys
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
    console.print(f"\n[dim]Use [bold]memory load <ID>[/bold] to load a session context[/dim]")


@cli.command()
@click.argument("session_id")
@click.option("--copy", "-c", is_flag=True, help="Copy context to clipboard")
def load(session_id, copy):
    """Load a session and print its context (ready to paste into any AI)."""
    from .core import load_session, list_sessions

    # Allow short ID prefix or list index (e.g. "3")
    session = None
    if session_id.isdigit():
        sessions = list_sessions()
        idx = int(session_id) - 1
        if 0 <= idx < len(sessions):
            session = load_session(sessions[idx].id)
    else:
        # Try full ID or prefix match
        from .core import get_backend
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
def save(provider, model, project, no_edit):
    """Save a new session from a chat transcript (auto-compressed by LLM)."""
    import os
    from .compressor import compress
    from .editor import open_in_editor
    from .core import create_session

    resolved_provider = provider or os.environ.get("MEMORY_LLM_PROVIDER", "openai")

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

    # Re-parse title/tags after potential edits
    from .compressor import _parse_response
    parsed = _parse_response(summary)

    session = create_session(
        title=parsed.title,
        project=resolved_project,
        summary=summary,
        tags=parsed.tags,
    )

    console.print(f"\n[green]Session saved![/green]")
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
    console.print(f"\n[dim]Use [bold]memory load <ID>[/bold] to load context[/dim]")


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
    import os
    backend = os.environ.get("MEMORY_BACKEND", "sqlite")
    provider = os.environ.get("MEMORY_LLM_PROVIDER", "openai")

    console.print(Panel(
        f"""[bold]Current Configuration[/bold]

[cyan]Storage backend:[/cyan]  {backend}
[cyan]LLM provider:[/cyan]     {provider}

[bold]Configure via environment variables or a .env file:[/bold]

  MEMORY_BACKEND=sqlite|supabase|s3|firebase
  MEMORY_LLM_PROVIDER=openai|anthropic|groq

  [dim]# SQLite (default)[/dim]
  SQLITE_DB_PATH=~/.ai-memory/memory.db

  [dim]# Supabase[/dim]
  SUPABASE_URL=https://xxx.supabase.co
  SUPABASE_KEY=your-anon-key

  [dim]# S3 / Cloudflare R2[/dim]
  S3_BUCKET_NAME=my-memory-bucket
  S3_ENDPOINT_URL=https://xxx.r2.cloudflarestorage.com  [dim]# R2 only[/dim]
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...

  [dim]# Firebase[/dim]
  FIREBASE_CREDENTIALS_PATH=./serviceAccountKey.json
  FIREBASE_PROJECT_ID=my-project

  [dim]# LLM API Keys[/dim]
  OPENAI_API_KEY=sk-...
  ANTHROPIC_API_KEY=sk-ant-...
  GROQ_API_KEY=gsk_...
""",
        border_style="cyan",
        title="memory config",
    ))


def _format_context(session) -> str:
    return f"""# Session Context

**Title:** {session.title}
**Project:** {session.project}
**Saved:** {session.updated_at.strftime('%Y-%m-%d %H:%M')} UTC

---

{session.summary}

---
*Loaded from AI Agent Portable Memory — https://github.com/your-username/AI-Agent-Portable-Memory*
"""


def _copy_to_clipboard(text: str) -> None:
    import subprocess
    import sys

    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    elif sys.platform == "win32":
        subprocess.run(["clip"], input=text.encode(), check=True)
    else:
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
        except FileNotFoundError:
            subprocess.run(["xdotool", "type", "--clearmodifiers", text], check=True)
