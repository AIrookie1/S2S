"""Create a clean, scanned review artifact without Git history or runtime outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from reproducibility.privacy import audit_tree

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "methods", "scripts", "configs", "generic", "experiments", "tools", "reproducibility",
    "tests", "docs", "requirements", "third_party", "truthfulqa", ".github",
)
ROOT_FILES = (
    "README.md", "requirements.txt", "pyproject.toml", "conftest.py", "run_method.py",
    "THIRD_PARTY_NOTICES.md", ".gitignore", ".editorconfig",
)
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".yml", ".yaml", ".toml", ".ini", ".csv", ".txt"}
IGNORED = {"__pycache__", ".pytest_cache", ".git", "results", "features", "checkpoints", "figures"}
LOCAL_PATH = re.compile(r"(?<![\w.])/(?:root|home|Users|mnt)/[^\s\"'`<>),;}$\]]+")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sanitize(text, source_root):
    text = text.replace(str(source_root), ".")
    def replace(match):
        value = match.group().rstrip(".")
        if value.endswith(("/bin/python", "/bin/python3")):
            return "python"
        if "/huggingface/datasets/" in value:
            return "data/" + value.split("/huggingface/datasets/", 1)[1]
        return "models/" + value.rsplit("/", 1)[-1]
    return LOCAL_PATH.sub(replace, text)


def local_deny_terms(source):
    terms = set()
    match = re.match(r"/(?:root|home|Users)/([^/]+)/", str(source))
    if match and len(match.group(1)) >= 3:
        terms.add(match.group(1))
    result = subprocess.run(["git", "log", "--format=%an%n%ae%n%cn%n%ce"], cwd=source,
                            capture_output=True, text=True, check=False)
    for value in result.stdout.splitlines() if result.returncode == 0 else []:
        if value.strip().lower() not in ("root", "anonymous", "anonymous authors", "anonymous@example.invalid") and len(value.strip()) >= 3:
            terms.add(value.strip())
    return sorted(terms)


def source_files(source, include_results=True):
    source = Path(source)
    for name in ROOT_FILES:
        if (source / name).is_file():
            yield source / name
    for folder in SOURCE_DIRS:
        for path in sorted((source / folder).rglob("*")):
            rel = path.relative_to(source / folder)
            if any(part in IGNORED for part in rel.parts) or not path.is_file():
                continue
            if path.suffix in TEXT_SUFFIXES or path.name.endswith("LICENSE"):
                yield path
    for name in ("data/README.md", "data/truthfulqa/README.md", "data/truthfulqa/TruthfulQA.csv",
                 "models/README.md", "results/README.md", "reports/validation.json", "reports/cleanup.json"):
        if (source / name).is_file():
            yield source / name
    if include_results:
        for path in sorted((source / "results/reference").glob("*")):
            if path.is_file() and path.suffix in (".json", ".csv", ".md"):
                yield path


def write_manifest(root):
    root = Path(root)
    excluded = {"reports/release_manifest.json", "reports/privacy.json"}
    files = {str(path.relative_to(root)): sha(path.read_bytes()) for path in source_files(root)
             if path.is_file() and str(path.relative_to(root)) not in excluded}
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/release_manifest.json").write_text(json.dumps({"format_version": 2, "files": files}, indent=2) + "\n")
    return files


def build(source, destination, include_results=True, deny_terms=()):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source.is_relative_to(destination):
        raise ValueError("The export must not contain the source repository")
    if destination.exists():
        raise ValueError("The export destination exists; choose a new path to preserve its files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anonymous-build-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for original in source_files(source, include_results):
            if original.is_symlink():
                raise ValueError("Refusing a source symlink")
            target = staging / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            # The public benchmark snapshot is copied byte for byte.
            if original == source / "data/truthfulqa/TruthfulQA.csv" or original.name.endswith("LICENSE"):
                target.write_bytes(original.read_bytes())
            else:
                target.write_text(sanitize(original.read_text(), source), encoding="utf-8")
            target.chmod(0o755 if original.stat().st_mode & 0o111 else 0o644)
        findings = audit_tree(staging, sorted(set(local_deny_terms(source)) | set(deny_terms)))
        if findings:
            raise ValueError("Privacy scan failed: " + json.dumps(findings[:20]))
        manifest = write_manifest(staging)
        (staging / "reports/privacy.json").write_text(json.dumps({
            "passed": True, "scanned_files": len(manifest), "git_history_exported": False,
            "runtime_outputs_exported": False,
            "scope": "Source-content scan; account metadata and recognizable code remain outside this check.",
        }, indent=2) + "\n")
        shutil.copytree(staging, destination)
    return {"destination": str(destination), "files": len(manifest), "privacy_scan": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/s2s-anonymous")
    parser.add_argument("--include-results", action="store_true", default=True)
    parser.add_argument("--deny-term", action="append", default=[])
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.source, arguments.output, arguments.include_results, arguments.deny_term), indent=2))


if __name__ == "__main__":
    main()
