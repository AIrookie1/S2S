#!/usr/bin/env python3
"""Collect one WEIGHT_KAPPA evaluation into the Fig. 5 summary JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    payload = json.loads(a.eval_json.read_text(encoding="utf-8"))
    mc1 = float(payload["metrics"]["MC1"])
    with np.load(a.checkpoint, allow_pickle=False) as data:
        rank = int(np.asarray(data["rank"]).item())
    summary = json.loads(a.output.read_text(encoding="utf-8")) if a.output.exists() else {
        "experiment": "fig5_weight_kappa_mc1",
        "method": "Similarity-Graph Adaptive 3D Spherical Steering",
        "x_axis": "WEIGHT_KAPPA", "y_axis": "MC1", "results": []}
    record = {"weight_kappa": a.kappa, "rank": rank, "MC1": mc1,
              "checkpoint": str(a.checkpoint), "evaluation": str(a.eval_json)}
    records = [x for x in summary.get("results", []) if float(x["weight_kappa"]) != a.kappa]
    records.append(record)
    summary["results"] = sorted(records, key=lambda x: float(x["weight_kappa"]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(a.output)
    print(f"WEIGHT_KAPPA={a.kappa:g} | rank={rank} | MC1={mc1:.6f}")

if __name__ == "__main__":
    main()
