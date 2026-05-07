"""Runtime helpers for local wrappers around the installed mmseg repo."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_mmseg_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    env_root = os.environ.get('MMSEG_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    import mmseg

    return Path(mmseg.__file__).resolve().parents[1]


def build_subprocess_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    current = env.get('PYTHONPATH', '')
    root = str(project_root())
    env['PYTHONPATH'] = (
        root if not current else f'{root}{os.pathsep}{current}'
    )
    if extra:
        env.update(extra)
    return env


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def add_project_root_to_syspath() -> None:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)

