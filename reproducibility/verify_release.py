"""Verify release-source hashes while separating local runtime artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from reproducibility.privacy import audit_tree

RUNTIME_DIRS = {"models", "features", "checkpoints", "prototypes", "dist", "build", ".venv", ".venv-scaling"}
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def _runtime_path(relative):
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in RUNTIME_DIRS or parts[0].startswith(".venv"):
        return True
    if parts[0] == "results" and len(parts) > 1 and parts[1] not in ("reference", "README.md"):
        return True
    return parts[0] == "data" and len(parts) > 1 and parts[1] not in ("truthfulqa", "README.md")


def verify(root, allow_git=False, strict=False):
    root = Path(root)
    manifest_path = root / "reports/release_manifest.json"
    if not manifest_path.is_file():
        return [{"file": "reports/release_manifest.json", "rule": "missing release manifest"}]
    expected = json.loads(manifest_path.read_text())["files"]
    findings = []
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            findings.append({"file": name, "rule": "missing or changed release file"})
    allowed = set(expected) | {"reports/release_manifest.json", "reports/privacy.json"}
    if (root / ".git").exists() and not allow_git:
        findings.append({"file": ".git", "rule": "Git history is not part of a clean anonymous export"})
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in CACHE_DIRS for part in rel.parts) or (not strict and _runtime_path(rel)):
            continue
        if path.is_symlink():
            findings.append({"file": str(rel), "rule": "unexpected symlink"})
        elif path.is_file() and str(rel) not in allowed:
            findings.append({"file": str(rel), "rule": "file added after source audit"})
    if strict:
        findings += audit_tree(root, allow_git=allow_git)
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--allow-git", action="store_true")
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    findings = verify(arguments.directory, arguments.allow_git, arguments.strict)
    print(json.dumps({"passed": not findings, "findings": findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
