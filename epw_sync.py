#!/usr/bin/env python3
"""
What it does:
- Downloads EPWgen if needed, then keeps it up to date.
- Finds all data branches (except any you exclude).
- Makes all .epw outputs accessible in one place by creating symlinks.
- Links are created in DEFAULT_EPW_ROOT.

How to use:
1) Edit CONFIG as needed.
2) Run: python3 epw_sync.py
"""
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_REPO_URL = "git@github.com:NatLabRockies/EPWgen.git" # HTTPS alternative: https://github.com/NatLabRockies/EPWgen.git
DEFAULT_MODE = "symlink"
_BASE_ROOT = Path("/projects/geohc/EPW")
DEFAULT_REPO_ROOT = _BASE_ROOT / "EPWgen"
DEFAULT_WORKTREE_ROOT = _BASE_ROOT / "epwgen_branches"
DEFAULT_EPW_ROOT = _BASE_ROOT / "epw_symlinks"

CONFIG = {
    "repo_url": DEFAULT_REPO_URL,
    "repo_root": DEFAULT_REPO_ROOT,
    "worktree_root": DEFAULT_WORKTREE_ROOT,
    "epw_root": DEFAULT_EPW_ROOT,
    "exclude_branches": ["main"],
    "mode": DEFAULT_MODE,  # "symlink" or "copy"
}


def _run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)

def _capture(cmd, cwd=None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True)


def _ensure_repo(repo_url: str, repo_root: Path) -> None:
    if repo_root.exists():
        return
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repo_url, str(repo_root)])

def _fetch_all(repo_root: Path) -> None:
    _run(["git", "-C", str(repo_root), "fetch", "--all", "--prune"])

def _list_remote_branches(repo_root: Path, exclude=None):
    exclude = set(exclude or [])
    output = _capture(
        ["git", "-C", str(repo_root), "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"]
    )
    branches = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line == "origin/HEAD":
            continue
        if line.startswith("origin/"):
            line = line[len("origin/") :]
        if line in {"HEAD", "origin"}:
            continue
        if line in exclude:
            continue
        branches.append(line)
    return branches


def _ensure_worktree(repo_root: Path, worktree_root: Path, branch: str) -> Path:
    wt_path = worktree_root / branch
    if wt_path.exists():
        _run(["git", "-C", str(wt_path), "pull", "--ff-only", "origin", branch])
        return wt_path
    worktree_root.mkdir(parents=True, exist_ok=True)
    _run(["git", "-C", str(repo_root), "fetch", "origin", branch])
    _run(["git", "-C", str(repo_root), "worktree", "add", str(wt_path), f"origin/{branch}"])
    return wt_path


def _sync_outputs(outputs_dir: Path, epw_root: Path, mode: str) -> None:
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue
        dest = epw_root / child.name
        if dest.exists():
            if dest.is_symlink():
                dest.unlink()
            else:
                print(f"Skipping {dest}: exists and is not a symlink")
                continue
        if mode == "symlink":
            os.symlink(child, dest)
            print(f"Symlinked {dest} -> {child}")
        else:
            shutil.copytree(child, dest)


def main() -> None:
    repo_root = Path(CONFIG["repo_root"]).expanduser().resolve()
    worktree_root = Path(CONFIG["worktree_root"]).expanduser().resolve()
    epw_root = Path(CONFIG["epw_root"]).expanduser().resolve()
    epw_root.mkdir(parents=True, exist_ok=True)

    mode = CONFIG["mode"]
    if mode not in {"copy", "symlink"}:
        raise ValueError("CONFIG['mode'] must be 'copy' or 'symlink'")

    _ensure_repo(CONFIG["repo_url"], repo_root)
    _fetch_all(repo_root)

    exclude_cfg = CONFIG.get("exclude_branches") or []
    exclude = {str(b).strip() for b in exclude_cfg if str(b).strip()}
    branches = _list_remote_branches(repo_root, exclude=exclude)
    print(f"Discovered branches: {', '.join(branches)}")
    if worktree_root.exists():
        for child in worktree_root.iterdir():
            if not child.is_dir():
                continue
            if child.name not in branches:
                print(f"Warning: stale worktree {child} (branch not on origin)")

    for branch in branches:
        wt = _ensure_worktree(repo_root, worktree_root, branch)
        outputs_dir = wt / "epwgen" / "outputs"
        if not outputs_dir.exists():
            print(f"Skipping {branch}: no outputs dir at {outputs_dir}")
            continue
        _sync_outputs(outputs_dir, epw_root, mode)


if __name__ == "__main__":
    main()
