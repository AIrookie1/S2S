"""Verify anonymous export hashes and rescan its complete text content."""
import argparse
import hashlib
import json
from pathlib import Path

from reproducibility.privacy import audit_tree


def verify(root, allow_git=False):
    root = Path(root)
    manifest = json.loads((root / "release_manifest.json").read_text())
    expected = manifest["files"]
    failures = []
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            failures.append(dict(file=name, rule="missing or changed release file"))
    allowed = set(expected) | {"release_manifest.json", "privacy_report.json"}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(x in (".git", "__pycache__", ".pytest_cache") for x in rel.parts):
            continue
        if path.is_file() and str(rel) not in allowed:
            failures.append(dict(file=str(rel), rule="file added after release audit"))
    failures += audit_tree(root, allow_git=allow_git)
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, nargs="?", default=Path.cwd())
    p.add_argument("--allow-git", action="store_true")
    a = p.parse_args()
    failures = verify(a.directory, a.allow_git)
    print(json.dumps(dict(passed=not failures, findings=failures), indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
