#!/usr/bin/env python3
"""Validate metadata for a cached S2S_R evaluation JSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--eval-split", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weight-kappa", type=float, required=True)
    parser.add_argument("--margin", type=float, required=True)
    args = parser.parse_args()

    try:
        report = json.loads(args.evaluation.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        for key in ("MC1", "MC2", "MC3"):
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"{key} is non-finite: {value}")
        parameters = report["parameters"]
    except Exception as exc:
        print(f"invalid evaluation {args.evaluation}: {exc}")
        raise SystemExit(1)

    mismatches: list[str] = []
    if str(report.get("model_name")) != args.model:
        mismatches.append(
            f"model_name: cached={report.get('model_name')!r}, requested={args.model!r}"
        )
    if int(report.get("layer", -1)) != args.layer:
        mismatches.append(
            f"layer: cached={report.get('layer')!r}, requested={args.layer!r}"
        )
    if str(report.get("eval_split")) != args.eval_split:
        mismatches.append(
            f"eval_split: cached={report.get('eval_split')!r}, requested={args.eval_split!r}"
        )

    try:
        cached_checkpoint = Path(str(report.get("checkpoint_path"))).expanduser().resolve()
        requested_checkpoint = args.checkpoint.expanduser().resolve()
        if cached_checkpoint != requested_checkpoint:
            mismatches.append(
                f"checkpoint_path: cached={cached_checkpoint}, requested={requested_checkpoint}"
            )
    except Exception as exc:
        mismatches.append(f"checkpoint_path invalid: {exc}")

    for key, wanted in (("weight_kappa", args.weight_kappa), ("margin", args.margin)):
        try:
            got = float(parameters[key])
            if not math.isclose(got, wanted, rel_tol=1e-6, abs_tol=1e-8):
                mismatches.append(f"{key}: cached={got!r}, requested={wanted!r}")
        except Exception:
            mismatches.append(f"{key}: cached={parameters.get(key)!r}, requested={wanted!r}")

    if mismatches:
        print(f"stale evaluation {args.evaluation}: " + "; ".join(mismatches))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
