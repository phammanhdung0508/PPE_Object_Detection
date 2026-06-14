import os
import site
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def add_nvidia_dll_directories() -> None:
    if os.name != "nt":
        return

    candidate_roots = []
    for path in site.getsitepackages():
        candidate_roots.append(Path(path))
    candidate_roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[Path] = set()
    path_entries = []
    for site_packages in candidate_roots:
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue

        for bin_dir in nvidia_root.glob("*/bin"):
            resolved = bin_dir.resolve()
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            path_entries.append(str(resolved))
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(resolved)))

    if path_entries:
        current_path = os.environ.get("PATH", "")
        current_entries = {entry.casefold() for entry in current_path.split(os.pathsep)}
        missing_entries = [
            entry for entry in path_entries if entry.casefold() not in current_entries
        ]
        if missing_entries:
            os.environ["PATH"] = os.pathsep.join(missing_entries + [current_path])
