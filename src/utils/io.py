"""Small JSON and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file."""
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)
