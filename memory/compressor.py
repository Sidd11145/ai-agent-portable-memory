import os
from dataclasses import dataclass
from typing import Optional

SYSTEM_PROMPT = """You are a memory compression assistant. Your job is to convert a raw AI chat transcript into a structured, compact session memory.

Output ONLY the following markdown format — no extra commentary:

## Title
<short descriptive title for this session, max 10 words>

## Project
<project name if mentioned, otherwise "default">

## What we were doing
<1-3 sentences describing the main goal of this session>

## What was completed
<bullet list of concrete things finished — files created, features implemented, bugs fixed>

## Key decisions made
<bullet list of important choices and the reason behind each>

## Current state / where we stopped
<1-2 sentences on exactly where things stand — what is incomplete, what file was being worked on>

## Next steps
<numbered list of what to do next>

## Important context
<any critical technical details: stack, versions, ports, credentials structure, constraints — things a new AI needs to know to continue>

## Tags
<comma-separated keywords for searching, e.g.: react, auth, jwt, python>

Be concise. The goal is maximum information density with minimum tokens."""


@dataclass
class CompressedSummary:
    title: str
    project: str
    summary: str  # full markdown body
    tags: list[str]


def compress(transcript: str, provider: str = "openai", model: Optional[str] = None) -> CompressedSummary:
    """
    Compress a raw chat transcript into a structured session summary.

    provider: "openai" | "anthropic" | "groq"
    model: optional override (defaults to a sensible cheap model per provider)
    """
    if provider == "openai":
        return _compress_openai(transcript, model or "gpt-4o-mini")
    elif provider == "anthropic":
        return _compress_anthropic(transcript, model or "claude-haiku-4-5-20251001")
    elif provider == "groq":
        return _compress_groq(transcript, model or "llama-3.1-8b-instant")
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose: openai, anthropic, groq")


def _parse_response(text: str) -> CompressedSummary:
    lines = text.strip().splitlines()
    title = "Untitled Session"
    project = "default"
    tags: list[str] = []

    for i, line in enumerate(lines):
        if line.startswith("## Title"):
            title = lines[i + 1].strip() if i + 1 < len(lines) else title
        elif line.startswith("## Project"):
            project = lines[i + 1].strip() if i + 1 < len(lines) else project
        elif line.startswith("## Tags"):
            raw = lines[i + 1].strip() if i + 1 < len(lines) else ""
            tags = [t.strip() for t in raw.split(",") if t.strip()]

    return CompressedSummary(title=title, project=project, summary=text.strip(), tags=tags)


def _compress_openai(transcript: str, model: str) -> CompressedSummary:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Compress this transcript:\n\n{transcript}"},
        ],
        temperature=0.2,
    )
    return _parse_response(resp.choices[0].message.content)


def _compress_anthropic(transcript: str, model: str) -> CompressedSummary:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Compress this transcript:\n\n{transcript}"}],
    )
    return _parse_response(resp.content[0].text)


def _compress_groq(transcript: str, model: str) -> CompressedSummary:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Compress this transcript:\n\n{transcript}"},
        ],
        temperature=0.2,
    )
    return _parse_response(resp.choices[0].message.content)
