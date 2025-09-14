"""Config loader utilities."""  # Author: Team DocuRay | Generated: TDD impl | Version: 0.1.0 | Modified: 2025-09-14

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # Fallback if PyYAML not installed
    yaml = None  # type: ignore


def load_yaml(path: str) -> Dict[str, Any]:  # Simple loader
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

