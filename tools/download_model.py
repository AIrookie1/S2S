"""Download an official model snapshot and record its resolved commit."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--revision", help="Exact commit SHA; omitted: resolve current official snapshot once")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from huggingface_hub import model_info, snapshot_download
    revision = a.revision or model_info(a.repo_id).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        p.error("revision must be a full 40-character model commit SHA")
    receipt = a.output / "download_receipt.json"
    identity = dict(repo_id=a.repo_id, revision=revision)
    if a.output.exists() and any(a.output.iterdir()):
        if not receipt.exists() or any(json.loads(receipt.read_text()).get(k) != v for k, v in identity.items()):
            raise ValueError("Nonempty model directory without a matching pinned receipt; use a new directory")
    a.output.mkdir(parents=True, exist_ok=True)
    # Write the pinned identity before transfer so interrupted downloads can resume.
    receipt.write_text(json.dumps(identity, indent=2) + "\n")
    snapshot_download(a.repo_id, revision=revision, local_dir=a.output)
    index = {}
    for f in sorted(a.output.iterdir()):
        if f.is_file() and f.name != receipt.name:
            item = dict(size=f.stat().st_size)
            if f.suffix in (".json", ".model", ".jinja"):
                item["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
            index[f.name] = item
    receipt.write_text(json.dumps(dict(**identity, files=index), indent=2) + "\n")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
