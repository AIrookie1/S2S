#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:?Usage: bash tools/publish_anonymous.sh https://github.com/ANONYMOUS_ACCOUNT/REPOSITORY.git}"
[[ ! -e .git ]] || { echo "Refusing a tree with existing Git history. Use the generated release directory." >&2; exit 2; }
"$PYTHON_BIN" - "$TARGET" <<'CHECK'
import re,sys
if not re.fullmatch(r"https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+\.git",sys.argv[1]):
    raise SystemExit("Use a token-free HTTPS URL for your empty anonymous GitHub repository")
CHECK
"$PYTHON_BIN" -m reproducibility.verify_release .
git init --initial-branch=main
git config --local user.name "Anonymous Authors"
git config --local user.email "anonymous@example.invalid"
git config --local commit.gpgsign false
git config --local tag.gpgsign false
git -c core.hooksPath=/dev/null add -- .
GIT_AUTHOR_NAME="Anonymous Authors" GIT_AUTHOR_EMAIL="anonymous@example.invalid" \
GIT_COMMITTER_NAME="Anonymous Authors" GIT_COMMITTER_EMAIL="anonymous@example.invalid" \
  git -c core.hooksPath=/dev/null -c commit.gpgsign=false commit -m "Anonymous reproducibility release"
git remote add origin "$TARGET"
git push --set-upstream origin main
