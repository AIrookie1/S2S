"""Create a reviewed text-only source tree without repository history or local artifacts."""
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
SOURCE_DIRS = ("method", "script", "hyperparams", "generic", "experiments", "tools", "reproducibility", "docs", ".github")
SOURCE_SUFFIXES = {".py", ".sh", ".md", ".yml", ".yaml", ".toml", ".ini"}
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", "results", "result", "figures", "checkpoints", "features", ".git"}
ROOT_FILES = ("README.md", "requirements.txt", "requirements-scaling.txt", "pytest.ini", "test.py", "THIRD_PARTY_NOTICES.md", ".gitignore")
TRUTHFULQA_FILES = ("__init__.py", "models.py", "utilities.py", "configs.py", "presets.py")
LOCAL_PATH = re.compile(r"(?<![\w.])/(?:root|home|Users|mnt)/[^\s\"'`<>),;}$\]]+")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sanitize(text, source_root):
    # Convert repository paths first, keeping checkpoint/dataset suffixes intact.
    text = text.replace(str(source_root), ".")
    def replace(match):
        value = match.group().rstrip(".")
        if value.endswith("/bin/python") or value.endswith("/bin/python3"):
            return "python"
        if "/huggingface/datasets/" in value:
            return "data/" + value.split("/huggingface/datasets/", 1)[1]
        leaf = value.rsplit("/", 1)[-1]
        return "models/" + leaf
    return LOCAL_PATH.sub(replace, text)


def local_deny_terms(source):
    terms = set()
    # Learn local usernames from paths, never hard-code them into a public file.
    match = re.match(r"/(?:root|home|Users)/([^/]+)/", str(source))
    if match and len(match.group(1)) >= 3:
        terms.add(match.group(1))
    result = subprocess.run(["git", "log", "--format=%an%n%ae%n%cn%n%ce"], cwd=source,
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        for value in result.stdout.splitlines():
            if value.strip().lower() not in ("root", "anonymous", "anonymous authors", "anonymous@example.invalid") and len(value.strip()) >= 3:
                terms.add(value.strip())
    return sorted(terms)


def source_files(source):
    for name in ROOT_FILES:
        p = source / name
        if p.is_file():
            yield p
    for folder in SOURCE_DIRS:
        for p in sorted((source / folder).rglob("*")):
            rel = p.relative_to(source / folder)
            if folder == "tools" and (rel.as_posix() in ("normalize_release_names.py", "curate_review_release.py") or rel.parts[0] == "review_release_assets"):
                continue
            if any(x in EXCLUDED_DIRS for x in rel.parts) or not p.is_file():
                continue
            if p.suffix in SOURCE_SUFFIXES or p.name.startswith("requirements") or p.name.endswith("LICENSE"):
                yield p
    for name in TRUTHFULQA_FILES:
        p = source / "truthfulqa" / name
        if p.is_file():
            yield p
    for name in ("LICENSE", "NOTICE", "data/v1/TruthfulQA.csv"):
        p = source / "TruthfulQA" / name
        if p.is_file():
            yield p


def build(source, destination, include_results=False, deny_terms=()):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source.is_relative_to(destination):
        raise ValueError("Release destination must not contain the source repository")
    if destination.exists():
        raise ValueError("Destination already exists; choose a new empty path to preserve edits")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anonymous-build-", dir=destination.parent) as staging_name:
        staging = Path(staging_name)
        files = list(source_files(source))
        for original in files:
            if original.is_symlink():
                raise ValueError("Refusing to follow source symlink")
            target = staging / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            text = original.read_text(encoding="utf-8")
            # Required third-party copyright notices remain verbatim.
            if original.name not in ("LICENSE", "NOTICE", "BIPO_LICENSE"):
                text = sanitize(text, source)
            if original.name == ".gitignore":
                text += "\n# Reproduction and release outputs\nresults/\nfeatures/\ncheckpoints/\nprototypes/\ndata/\nmodels/\n.venv*/\n.env*\n*.log\n*.zip\nanonymous_release*/\n"
                # The vendored dataset and library must remain tracked.
                text = "\n".join(x for x in text.splitlines() if x.strip() not in ("TruthfulQA", "truthfulqa")) + "\n"
            target.write_text(text, encoding="utf-8")
            target.chmod(0o755 if original.stat().st_mode & 0o111 else 0o644)
        library_init = staging / "truthfulqa/__init__.py"
        if not library_init.exists():
            library_init.parent.mkdir(parents=True, exist_ok=True)
            library_init.write_text('"""Bundled local TruthfulQA evaluation dependency; see third-party notices."""\n')
        if include_results:
            from reproducibility.audit_results import write_audit
            write_audit(source, staging / "artifacts", sanitizer=lambda s: sanitize(s, source))
        naming_adapter = source / "tools/normalize_release_names.py"
        if naming_adapter.is_file():
            import runpy
            adapter = runpy.run_path(str(naming_adapter))
            adapter["normalize_tree"](staging)
        review_adapter = source / "tools/curate_review_release.py"
        if review_adapter.is_file():
            import runpy
            runpy.run_path(str(review_adapter))["curate_tree"](staging)
        deny = sorted(set(local_deny_terms(source)) | set(deny_terms))
        findings = audit_tree(staging, deny)
        if findings:
            raise ValueError("Release blocked by privacy scan: " + json.dumps(findings[:30]))
        manifest = {str(p.relative_to(staging)): sha(p.read_bytes()) for p in sorted(staging.rglob("*")) if p.is_file()}
        (staging / "release_manifest.json").write_text(json.dumps(dict(format_version=1, files=manifest), indent=2) + "\n")
        (staging / "privacy_report.json").write_text(json.dumps(dict(passed=True, scanned_files=len(manifest),
            history_exported=False, binary_artifacts_exported=False,
            limitation="Content scan only; account ownership, platform metadata, and recognizable code can still identify authors."), indent=2) + "\n")
        # Copy only after the complete staging tree passes the audit.
        shutil.copytree(staging, destination)
    return dict(destination=str(destination), files=len(manifest), privacy_scan="passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=ROOT / "anonymous_release")
    p.add_argument("--include-results", action="store_true", help="Include compact sanitized evidence, not logs/checkpoints")
    p.add_argument("--deny-term", action="append", default=[])
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output, a.include_results, a.deny_term), indent=2))


if __name__ == "__main__":
    main()
