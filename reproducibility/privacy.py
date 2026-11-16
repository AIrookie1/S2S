"""Fail closed on identity-bearing paths, credentials, and unexpected release files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

# Patterns are deliberately expressed without embedding a real user identity.
RULES = {
    "personal absolute path": re.compile(r"(?<![\w.])/(?:root|home|Users|mnt)/[^\s\"'`<>),;]+"),
    "local file URL": re.compile("file" + r":/+[^\s\"'<>]+"),
    "email address": re.compile(r"\b[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "access token": re.compile(r"\b(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "URL credentials": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    "credential assignment": re.compile(r"""(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*["'][A-Za-z0-9_+/=-]{16,}["']"""),
}
EMAIL_ALLOWLIST = {"anonymous@example.invalid"}
UNSAFE_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".ipynb", ".png", ".pdf", ".svg", ".log", ".pem", ".key"}
UNSAFE_NAMES = {".env", ".netrc", ".git-credentials", "credentials.json", "id_rsa", "id_ed25519"}
SKIP = {"__pycache__", ".pytest_cache", ".venv", "venv"}


def audit_tree(root, deny_terms=(), allow_git=False):
    root = Path(root)
    findings = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP for part in rel.parts):
            continue
        if ".git" in rel.parts:
            if not allow_git and rel.parts == (".git",):
                findings.append(dict(file=".git", line=0, rule="Git history must not be exported"))
            continue
        if path.is_symlink():
            findings.append(dict(file=str(rel), line=0, rule="symlink is not allowed"))
            continue
        if not path.is_file():
            continue
        if path.name in UNSAFE_NAMES or path.suffix.lower() in UNSAFE_SUFFIXES:
            findings.append(dict(file=str(rel), line=0, rule="unreviewed binary, metadata, or secret file"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            findings.append(dict(file=str(rel), line=0, rule="non-text file requires explicit review"))
            continue
        for term in deny_terms:
            if len(term.strip()) >= 3 and term.casefold() in (str(rel) + "\n" + text).casefold():
                findings.append(dict(file=str(rel), line=0, rule="private deny-list match"))
        for name, pattern in RULES.items():
            for match in pattern.finditer(text):
                if name == "email address" and match.group() in EMAIL_ALLOWLIST:
                    continue
                findings.append(dict(file=str(rel), line=text.count("\n", 0, match.start()) + 1, rule=name))
    # Values are never echoed: the report itself must not reveal a credential.
    return findings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--deny-term", action="append", default=[])
    p.add_argument("--allow-git", action="store_true", help="Scan a clean initialized release; ignore .git internals")
    a = p.parse_args()
    findings = audit_tree(a.directory, a.deny_term, a.allow_git)
    print(json.dumps(dict(passed=not findings, finding_count=len(findings), findings=findings), indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
