"""Commit this repo and push it to GitHub using credentials from .env.

.env needs GITHUB_USER, GITHUB_TOKEN (a personal access token) and optionally GITHUB_REPO
(defaults to this folder's name). The token is handed to git only for the push, through a
temporary GIT_ASKPASS helper: it is never written to .git/config, the remote URL, or the
macOS keychain.

Usage:
  .venv/bin/python tools/github_push.py --check              # show what would be committed; no changes
  .venv/bin/python tools/github_push.py -m "Initial commit"   # create the repo if needed (private), commit, push
  .venv/bin/python tools/github_push.py -m "..." --public     # create it public instead (first run only)

Safety checks before anything is committed:
  - .env must be ignored and not tracked
  - no value from .env (keys, passwords, token) may appear in any file about to be committed
  - no file over 50 MB (GitHub warns at 50 MB and rejects at 100 MB)
"""

import argparse
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"
MAX_MB = 50


def git(*args: str, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout


def fail(msg: str) -> None:
    sys.exit(f"Stopped: {msg}")


def safety_checks(secrets: list[str]) -> list[str]:
    """Stage everything and check it; returns the staged file list. Leaves the index staged."""
    if git("ls-files", "--", ".env").strip():
        fail(".env is tracked by git. Run `git rm --cached .env` first.")
    if subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode != 0:
        fail(".env is not covered by .gitignore.")
    git("add", "-A")
    staged = [f for f in git("diff", "--cached", "--name-only", "-z").split("\0") if f]
    for value in secrets:
        hits = git("grep", "--cached", "-l", "-F", "-e", value, check=False).split()
        if hits:
            git("reset", "-q")
            fail(f"a value from .env appears in {hits}. Remove it (use os.environ) and try again.")
    big = [f for f in git("ls-files", "-z").split("\0")
           if f and (ROOT / f).is_file() and (ROOT / f).stat().st_size > MAX_MB * 1e6]
    if big:
        git("reset", "-q")
        fail(f"files over {MAX_MB} MB: {big}. Add them to .gitignore.")
    return staged


def ensure_repo(user: str, repo: str, token: str, public: bool) -> None:
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"{API}/repos/{user}/{repo}", headers=h, timeout=30)
    if r.status_code == 200:
        print(f"GitHub repo {user}/{repo} exists ({'private' if r.json().get('private') else 'public'}).")
        return
    if r.status_code == 401:
        fail("GitHub rejected the token (401). Check GITHUB_TOKEN in .env.")
    if r.status_code != 404:
        fail(f"GitHub returned {r.status_code}: {r.text[:200]}")
    r = requests.post(f"{API}/user/repos", headers=h, timeout=30,
                      json={"name": repo, "private": not public,
                            "description": "ERCOT generation surprises vs Henry Hub natural gas futures"})
    if r.status_code not in (200, 201):
        fail(f"could not create {user}/{repo} ({r.status_code}): {r.text[:300]}\n"
             "The token needs Administration = Read and write to create repositories.")
    print(f"Created {'public' if public else 'private'} repo {r.json()['html_url']}")


def push(user: str, token: str) -> None:
    fd, helper = tempfile.mkstemp(suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write('#!/bin/sh\ncase "$1" in Username*) echo "$GH_PUSH_USER" ;; *) echo "$GH_PUSH_TOKEN" ;; esac\n')
        os.chmod(helper, stat.S_IRWXU)
        env = {**os.environ, "GIT_ASKPASS": helper, "GIT_TERMINAL_PROMPT": "0",
               "GH_PUSH_USER": user, "GH_PUSH_TOKEN": token}
        branch = git("branch", "--show-current").strip() or "main"
        # credential.helper= : don't let the macOS keychain store the token
        r = subprocess.run(["git", "-c", "credential.helper=", "push", "-u", "origin", branch],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        print(r.stdout + r.stderr.replace(token, "***"))
        if r.returncode != 0:
            fail("push failed (see above).")
    finally:
        os.remove(helper)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--message", help="commit message (needed when there are changes)")
    ap.add_argument("--check", action="store_true", help="show what would be committed, change nothing")
    ap.add_argument("--public", action="store_true", help="create the GitHub repo as public (first run)")
    args = ap.parse_args()

    cfg = dotenv_values(ROOT / ".env")
    user, token = cfg.get("GITHUB_USER"), cfg.get("GITHUB_TOKEN")
    repo = cfg.get("GITHUB_REPO") or ROOT.name
    secrets = [v for v in cfg.values() if v and len(v) >= 8]

    if not (ROOT / ".git").exists():
        git("init", "-b", "main")
        print("Initialized a git repository (branch main).")

    staged = safety_checks(secrets)
    if args.check:
        git("reset", "-q")
        print(f"{len(staged)} files would be committed (safety checks passed):")
        print("\n".join(f"  {f}" for f in staged))
        print(f"\nTarget: github.com/{user or '<GITHUB_USER>'}/{repo}")
        return

    if not user or not token:
        git("reset", "-q")
        fail("set GITHUB_USER and GITHUB_TOKEN in .env.")
    if staged:
        if not args.message:
            git("reset", "-q")
            fail(f"{len(staged)} changed files; pass a commit message with -m.")
        git("commit", "-q", "-m", args.message)
        print(f"Committed {len(staged)} files: {args.message}")
    elif not git("rev-parse", "--verify", "-q", "HEAD", check=False).strip():
        fail("nothing to commit yet.")

    ensure_repo(user, repo, token, args.public)
    url = f"https://github.com/{user}/{repo}.git"
    if git("remote", check=False).split().count("origin"):
        git("remote", "set-url", "origin", url)
    else:
        git("remote", "add", "origin", url)
    push(user, token)
    print(f"Done: https://github.com/{user}/{repo}")


if __name__ == "__main__":
    main()
