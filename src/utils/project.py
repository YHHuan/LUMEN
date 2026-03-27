"""
Multi-Project Support — LUMEN v2
==================================
Each project lives under data/<project_name>/ with its own pipeline artifacts.
"""

import json
import os
from pathlib import Path
from typing import Optional

_DATA_ROOT = Path("data")
_ACTIVE_PROJECT_FILE = _DATA_ROOT / ".active_project"

_active_data_dir: str = "data"


def get_data_dir() -> str:
    """Return the active project's data directory (e.g. 'data/PSY_MCI_rTMS')."""
    return _active_data_dir


def set_data_dir(project_dir: str) -> None:
    """Set the active project data directory globally."""
    global _active_data_dir
    _active_data_dir = project_dir


def _discover_projects() -> list[str]:
    """Find all project directories under data/ (those containing input/pico.yaml)."""
    if not _DATA_ROOT.exists():
        return []
    projects = []
    for d in sorted(_DATA_ROOT.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and (d / "input" / "pico.yaml").exists():
            projects.append(d.name)
    return projects


def select_project(skip_prompt: bool = False) -> str:
    """
    Interactive project selector. Called at the start of every script.
    Returns the project data directory path (e.g. 'data/PSY_MCI_rTMS').
    """
    global _active_data_dir

    projects = _discover_projects()

    # Environment variable override — enables non-interactive / background execution
    env_project = os.environ.get("LUMEN_PROJECT", "").strip()
    if env_project and env_project in projects:
        _active_data_dir = str(_DATA_ROOT / env_project)
        _ACTIVE_PROJECT_FILE.write_text(env_project, encoding="utf-8")
        print(f"\n  Active project (env): {env_project}")
        print(f"  Data directory: {_active_data_dir}/\n")
        return _active_data_dir

    last_active: Optional[str] = None
    if _ACTIVE_PROJECT_FILE.exists():
        last_active = _ACTIVE_PROJECT_FILE.read_text(encoding="utf-8").strip()

    if skip_prompt and last_active and last_active in projects:
        _active_data_dir = str(_DATA_ROOT / last_active)
        return _active_data_dir

    print("\n" + "=" * 50)
    print("  LUMEN v2 — Project Selector")
    print("=" * 50)

    if projects:
        for i, name in enumerate(projects, 1):
            marker = " (last active)" if name == last_active else ""
            print(f"  [{i}] {name}{marker}")
        print(f"  [N] Create new project")
    else:
        print("  No existing projects found.")

    print()

    while True:
        default_hint = ""
        if last_active and last_active in projects:
            idx = projects.index(last_active) + 1
            default_hint = f" [default: {idx}={last_active}]"

        choice = input(f"Select project{default_hint}: ").strip()

        if not choice and last_active and last_active in projects:
            chosen = last_active
            break

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(projects):
                chosen = projects[idx - 1]
                break
            else:
                print(f"  Invalid number. Enter 1-{len(projects)} or N.")
                continue

        if choice.upper() == "N" or not projects:
            name = input("  New project name (e.g. PSY_MCI_rTMS): ").strip()
            if not name:
                print("  Name cannot be empty.")
                continue
            name = name.replace(" ", "_")
            project_dir = _DATA_ROOT / name / "input"
            project_dir.mkdir(parents=True, exist_ok=True)
            template = project_dir / "pico.yaml"
            if not template.exists():
                template.write_text(
                    "# PICO Framework — edit before running Phase 1\n"
                    "pico:\n"
                    "  population: \"\"\n"
                    "  intervention: \"\"\n"
                    "  comparison: \"\"\n"
                    "  outcome: \"\"\n"
                    "  study_design: \"randomized controlled trials\"\n",
                    encoding="utf-8",
                )
                print(f"  Created {template} — edit this before running Phase 1.")
            chosen = name
            break
        else:
            matches = [p for p in projects if p.lower() == choice.lower()]
            if matches:
                chosen = matches[0]
                break
            print(f"  Unknown choice '{choice}'. Try again.")

    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _ACTIVE_PROJECT_FILE.write_text(chosen, encoding="utf-8")

    _active_data_dir = str(_DATA_ROOT / chosen)
    print(f"\n  Active project: {chosen}")
    print(f"  Data directory: {_active_data_dir}/")
    print()
    return _active_data_dir
