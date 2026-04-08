"""One-click patch/unpatch tool for SGLang + FlexKV integration.

Applies (or reverts) a small unified diff to the installed SGLang source
so that FlexKV can be used as a HiCacheStorage backend.

Usage::

    # Apply patch (auto-detects SGLang location)
    flexkv-patch-sglang

    # Check whether patch is already applied
    flexkv-patch-sglang --check

    # Revert (undo) the patch
    flexkv-patch-sglang --revert

    # Manually specify SGLang source root
    flexkv-patch-sglang --sglang-path /path/to/sglang
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# Patch file bundled alongside this module
_PATCH_DIR = Path(__file__).parent / "patches"
_PATCH_FILE = _PATCH_DIR / "sglang_flexkv.patch"

# File used as a quick "is patched?" probe
_PROBE_REL = Path("python/sglang/srt/mem_cache/storage/backend_factory.py")
_PROBE_MARKER = '"flexkv"'


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_sglang_root(manual_path: str | None = None) -> Path:
    """Locate the SGLang source root directory.

    Resolution order:
    1. ``--sglang-path`` CLI argument
    2. ``import sglang`` and walk up to find the repo root
       (the directory containing ``python/sglang/``)
    """
    if manual_path:
        p = Path(manual_path).resolve()
        if not p.is_dir():
            _die(f"Provided path does not exist: {p}")
        return p

    try:
        mod = importlib.import_module("sglang")
    except ImportError:
        _die("sglang is not installed.  Install it first or use --sglang-path.")

    # sglang.__file__ is typically <root>/python/sglang/__init__.py
    sglang_init = Path(mod.__file__).resolve()
    # Walk up looking for the repo root that contains python/sglang/
    for ancestor in sglang_init.parents:
        if (ancestor / "python" / "sglang").is_dir():
            return ancestor
        # editable install: sglang package might be directly under site-packages
        # In that case ancestor == site-packages/sglang → go one more up
        if (ancestor / "srt" / "managers" / "cache_controller.py").is_file():
            # We're inside the sglang package directly (no python/ prefix)
            return ancestor.parent

    # Fallback: assume sglang_init's grandparent (python/sglang/ → ../../)
    fallback = sglang_init.parent.parent.parent
    if (fallback / _PROBE_REL).is_file():
        return fallback

    _die(
        f"Could not determine SGLang source root from {sglang_init}.\n"
        "Use --sglang-path to specify it manually."
    )


def _is_patched(sglang_root: Path) -> bool:
    """Return True if the FlexKV patch appears to be applied."""
    probe = sglang_root / _PROBE_REL
    if not probe.is_file():
        # Try without the python/ prefix (pip install layout)
        probe = sglang_root / _PROBE_REL.relative_to("python")
    if not probe.is_file():
        return False
    return _PROBE_MARKER in probe.read_text()


def _resolve_strip_level(sglang_root: Path) -> int:
    """Determine the correct -p (strip) level for ``patch``/``git apply``.

    The patch has paths like ``a/python/sglang/...``.
    - If sglang_root contains ``python/sglang/``, strip = 1 (standard git layout)
    - If sglang_root directly contains ``sglang/``, strip = 2 (installed into site-packages parent)
    """
    if (sglang_root / "python" / "sglang").is_dir():
        return 1
    if (sglang_root / "sglang").is_dir():
        return 2
    return 1  # default


def _has_git(sglang_root: Path) -> bool:
    """Check if sglang_root is a git repo."""
    return (sglang_root / ".git").exists()


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDOUT: {result.stdout.strip()}")
        print(f"  STDERR: {result.stderr.strip()}")
    return result


def _die(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------
# Actions
# ------------------------------------------------------------------

def do_check(sglang_root: Path) -> None:
    """Check if patch is already applied."""
    if _is_patched(sglang_root):
        print(f"FlexKV patch is APPLIED  (SGLang root: {sglang_root})")
    else:
        print(f"FlexKV patch is NOT applied  (SGLang root: {sglang_root})")


def do_apply(sglang_root: Path) -> None:
    """Apply the FlexKV patch to SGLang."""
    if _is_patched(sglang_root):
        print("Patch is already applied. Nothing to do.")
        return

    if not _PATCH_FILE.is_file():
        _die(f"Patch file not found: {_PATCH_FILE}")

    strip = _resolve_strip_level(sglang_root)
    patch_path = str(_PATCH_FILE.resolve())

    # Try git apply first (better error messages, atomic)
    if _has_git(sglang_root):
        result = _run(
            ["git", "apply", f"-p{strip}", "--check", patch_path],
            cwd=sglang_root, check=False,
        )
        if result.returncode == 0:
            result = _run(
                ["git", "apply", f"-p{strip}", patch_path],
                cwd=sglang_root,
            )
            if result.returncode == 0:
                print("Patch applied successfully via git apply.")
                return
            else:
                _die(f"git apply failed:\n{result.stderr}")

        # git apply --check failed; fall through to patch(1)
        print("  git apply --check failed, trying patch(1) ...")

    # Fallback: patch(1)
    result = _run(
        ["patch", f"-p{strip}", "--dry-run", "-i", patch_path],
        cwd=sglang_root, check=False,
    )
    if result.returncode != 0:
        _die(
            f"Patch cannot be applied cleanly.\n"
            f"  {result.stdout.strip()}\n"
            f"Your SGLang version may be incompatible.  "
            f"The patch was generated against SGLang v0.5.6 (tag: v0.5.6)."
        )

    result = _run(
        ["patch", f"-p{strip}", "-i", patch_path],
        cwd=sglang_root,
    )
    if result.returncode == 0:
        print("Patch applied successfully via patch(1).")
    else:
        _die(f"patch(1) failed:\n{result.stderr}")


def do_revert(sglang_root: Path) -> None:
    """Revert (undo) the FlexKV patch."""
    if not _is_patched(sglang_root):
        print("Patch is not applied. Nothing to revert.")
        return

    if not _PATCH_FILE.is_file():
        _die(f"Patch file not found: {_PATCH_FILE}")

    strip = _resolve_strip_level(sglang_root)
    patch_path = str(_PATCH_FILE.resolve())

    if _has_git(sglang_root):
        result = _run(
            ["git", "apply", f"-p{strip}", "--reverse", patch_path],
            cwd=sglang_root, check=False,
        )
        if result.returncode == 0:
            print("Patch reverted successfully via git apply --reverse.")
            return

    # Fallback
    result = _run(
        ["patch", f"-p{strip}", "-R", "-i", patch_path],
        cwd=sglang_root, check=False,
    )
    if result.returncode == 0:
        print("Patch reverted successfully via patch -R.")
    else:
        _die(f"Revert failed:\n{result.stdout}\n{result.stderr}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="flexkv-patch-sglang",
        description="Apply or revert the FlexKV integration patch for SGLang.",
    )
    parser.add_argument(
        "--sglang-path",
        default=None,
        help="Path to SGLang source root (auto-detected if omitted).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Check whether the patch is already applied.",
    )
    group.add_argument(
        "--revert",
        action="store_true",
        help="Revert (undo) the patch.",
    )
    args = parser.parse_args()

    sglang_root = _find_sglang_root(args.sglang_path)
    print(f"SGLang root: {sglang_root}")

    if args.check:
        do_check(sglang_root)
    elif args.revert:
        do_revert(sglang_root)
    else:
        do_apply(sglang_root)


if __name__ == "__main__":
    main()
