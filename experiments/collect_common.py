from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def load_mc1(path: Path) -> tuple[float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(payload["metrics"]["MC1"]), payload
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric metrics.MC1 in {path}") from exc


def checkpoint_scalar(checkpoint: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in checkpoint.files:
        if default is not None:
            return default
        raise KeyError(f"checkpoint missing {key}")
    return np.asarray(checkpoint[key]).item()


def update_summary(output: Path, metadata: dict, record: dict, identity_keys: tuple[str, ...], sort_keys: tuple[str, ...]):
    if output.exists():
        summary = json.loads(output.read_text(encoding="utf-8"))
        for key in ("experiment", "model", "fold_idx"):
            if key in metadata and summary.get(key) != metadata[key]:
                raise ValueError(f"Summary metadata mismatch for {key} in {output}")
    else:
        summary = {**metadata, "results": []}
    def same(row):
        return all(row.get(k) == record.get(k) for k in identity_keys)
    rows = [row for row in summary.get("results", []) if not same(row)]
    rows.append(record)
    summary["results"] = sorted(rows, key=lambda row: tuple(row[k] for k in sort_keys))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
