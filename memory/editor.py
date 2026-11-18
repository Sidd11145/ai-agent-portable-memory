import os
import subprocess
import tempfile
from pathlib import Path


def open_in_editor(content: str) -> str:
    """Open content in the user's $EDITOR and return the edited result."""
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        subprocess.run([editor, tmp_path], check=True)
        return Path(tmp_path).read_text()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
