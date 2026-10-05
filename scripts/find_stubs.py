import os
from pathlib import Path

ROOT = Path(".")

IGNORE_DIRS = {".venv", ".venv_gdf", "data", "notebooks", "results", 
               "__pycache__", ".git", ".pytest_cache", "nai.egg-info"}

for p in sorted(ROOT.rglob("*.py")):
    if any(part in IGNORE_DIRS for part in p.parts):
        continue
    size = p.stat().st_size
    if size < 50:
        content = p.read_text(encoding="utf-8", errors="ignore").strip()
        print(f"{size:>5} B  {p}")
        if content and len(content) < 100:
            print(f"        └─ {content[:80]}")
