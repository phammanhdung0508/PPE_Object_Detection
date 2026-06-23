import os
import site
import sys
import ctypes
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_LOADED_SHARED_LIBS = []


def add_nvidia_dll_directories() -> None:
    if os.name == "nt":
        add_windows_nvidia_dll_directories()
        return

    add_linux_nvidia_shared_libraries()


def add_windows_nvidia_dll_directories() -> None:
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


def add_linux_nvidia_shared_libraries() -> None:
    candidate_roots = [Path(path) for path in site.getsitepackages()]
    candidate_roots.append(Path(sys.prefix) / "lib" / "python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

    lib_dirs: list[Path] = []
    for site_packages in candidate_roots:
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue
        lib_dirs.extend(path.resolve() for path in nvidia_root.glob("*/lib") if path.exists())

    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_entries = {Path(entry).resolve() for entry in existing.split(os.pathsep) if entry}
    missing_dirs = [path for path in lib_dirs if path not in existing_entries]
    if missing_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(path) for path in missing_dirs] + ([existing] if existing else [])
        )

    for library_name in (
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
    ):
        for lib_dir in lib_dirs:
            library_path = lib_dir / library_name
            if not library_path.exists():
                continue
            _LOADED_SHARED_LIBS.append(ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL))
            break
