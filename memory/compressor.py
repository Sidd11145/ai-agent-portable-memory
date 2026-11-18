from __future__ import annotations

import os
import re
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
<critical technical details: stack, versions, ports, config structure, constraints — things a new AI needs to know to continue>
IMPORTANT: Never include actual values for secrets, tokens, API keys, passwords, or credentials. Describe their names/structure only (e.g. "ANTHROPIC_API_KEY is required" not the actual key value).

## Tags
<comma-separated keywords for searching, e.g.: react, auth, jwt, python>

Be concise. The goal is maximum information density with minimum tokens."""


# Patterns that should never appear in stored summaries
_SECRET_PATTERNS = [
    # Anthropic API keys (sk-ant-api03-...) — full or LLM-truncated (sk-ant-...)
    (re.compile(r'sk-ant-[A-Za-z0-9_\-\.]{3,}'), '[ANTHROPIC_API_KEY]'),
    # OpenAI API keys (sk-... or sk-proj-...)
    (re.compile(r'sk-(?:proj-)?[A-Za-z0-9_\-\.]{3,}'), '[OPENAI_API_KEY]'),
    # Groq keys
    (re.compile(r'gsk_[A-Za-z0-9\.]{3,}'), '[GROQ_API_KEY]'),
    # UUID-like tokens: full (8-4-4-4-12) and LLM-truncated forms (bf517899-...)
    # Requires at least one hyphen segment so bare 8-char commit hashes don't match.
    (re.compile(r'(?<![A-Za-z0-9])[0-9a-f]{8}-(?:[0-9a-f]{1,4}-?|\.\.\.)+(?![A-Za-z0-9-])'), '[TOKEN]'),
    # Generic "key = <value>" patterns with long random-looking values
    (re.compile(r'(?i)(api[_-]?key|auth[_-]?token|secret|password|bearer)\s*[=:]\s*["\']?([A-Za-z0-9_\-\.]{8,})["\']?'), r'\1=[REDACTED]'),
]


def scrub_secrets(text: str) -> str:
    """Remove known secret patterns from text before storing."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


@dataclass
class CompressedSummary:
    title: str
    project: str
    summary: str  # full markdown body
    tags: list[str]


def compress(transcript: str, provider: str = "openai", model: Optional[str] = None) -> CompressedSummary:
    """
    Compress a raw chat transcript into a structured session summary.

    provider: openai | anthropic | groq | gemini | grok | mistral | ollama | deepseek
    model: optional override (defaults to a sensible cheap model per provider)
    """
    providers = {
        "openai":    (_compress_openai,    "gpt-4o-mini"),
        "anthropic": (_compress_anthropic, "claude-haiku-4-5-20251001"),
        "groq":      (_compress_groq,      "llama-3.1-8b-instant"),
        "gemini":    (_compress_gemini,    "gemini-2.0-flash"),
        "grok":      (_compress_grok,      "grok-2"),
        "mistral":   (_compress_mistral,   "mistral-small-latest"),
        "ollama":    (_compress_ollama,    "llama3.2"),
        "deepseek":  (_compress_deepseek,  "deepseek-chat"),
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose: {', '.join(providers)}"
        )

    fn, default_model = providers[provider]
    result = fn(scrub_secrets(transcript), model or default_model)
    result.summary = scrub_secrets(result.summary)
    return result


def _parse_response(text: str) -> CompressedSummary:
    lines = text.strip().splitlines()
    title = ""
    project = "default"
    tags: list[str] = []

    for i, line in enumerate(lines):
        if line.startswith("## Title"):
            title = lines[i + 1].strip() if i + 1 < len(lines) else ""
        elif line.startswith("## Project"):
            project = lines[i + 1].strip() if i + 1 < len(lines) else project
        elif line.startswith("## Tags"):
            raw = lines[i + 1].strip() if i + 1 < len(lines) else ""
            tags = [t.strip() for t in raw.split(",") if t.strip()]

    if not title:
        raise ValueError(
            "LLM did not return a structured summary (no ## Title section). "
            "The transcript may be too short or not a valid chat conversation."
        )

    return CompressedSummary(title=title, project=project, summary=text.strip(), tags=tags)


def _messages(transcript: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Compress this transcript:\n\n{transcript}"},
    ]


# ── OpenAI ────────────────────────────────────────────────────────────────────

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
        messages=_messages(transcript),
        temperature=0.2,
    )
    return _parse_response(resp.choices[0].message.content)


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _compress_anthropic(transcript: str, model: str) -> CompressedSummary:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if base_url:
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
        )
    else:
        client = anthropic.Anthropic(api_key=api_key)

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Compress this transcript:\n\n{transcript}"}],
    )
    return _parse_response(resp.content[0].text)


# ── Groq ──────────────────────────────────────────────────────────────────────

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
        messages=_messages(transcript),
        temperature=0.2,
    )
    return _parse_response(resp.choices[0].message.content)


# ── Gemini ────────────────────────────────────────────────────────────────────

def _compress_gemini(transcript: str, model: str) -> CompressedSummary:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )
    resp = gemini.generate_content(
        f"Compress this transcript:\n\n{transcript}",
        generation_config={"temperature": 0.2},
    )
    return _parse_response(resp.text)


# ── Grok (xAI) ────────────────────────────────────────────────────────────────

def _compress_grok(transcript: str, model: str) -> CompressedSummary:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        raise EnvironmentError("XAI_API_KEY or GROK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=_messages(transcript),
        temperature=0.2,
    )
    return _parse_response(resp.choices[0].message.content)


# ── Mistral ───────────────────────────────────────────────────────────────────

def _compress_mistral(transcript: str, model: str) -> CompressedSummary:
    try:
        from mistralai import Mistral
    except ImportError:
        raise ImportError("Run: pip install mistralai")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not set")

    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=model,
        messages=_messages(transcript),
        temperature=0.2,
    )
    return _parse_response(resp.choices[0].message.content)


# ── Ollama (local) ────────────────────────────────────────────────────────────

def _compress_ollama(transcript: str, model: str) -> CompressedSummary:
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": _messages(transcript),
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    return _parse_response(resp.json()["message"]["content"])


# ── DeepSeek ──────────────────────────────────────────────────────────────────

def _compress_deepseek(transcript: str, model: str) -> CompressedSummary:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=_messages(transcript),
        temperature=0.2,
    )
    return _parse_response(resp.choices[0].message.content)
