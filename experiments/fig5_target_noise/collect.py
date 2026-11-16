#!/usr/bin/env python3
"""Collect per-noise TruthfulQA MC1 results from the four steering methods."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

METHODS = ("L2S-Linear", "L2S-SLERP", "S2S_L_Static", "S2S_R")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--noise_levels", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()

def tag(sigma: float) -> str:
    return f"{sigma:g}".replace(".", "p")

def read_mc1(path: Path, method: str) -> float:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if method in {"L2S-Linear", "L2S-SLERP"}:
        wanted = "l2s_linear" if method == "L2S-Linear" else "l2s_slerp"
        for row in obj["results"]:
            if row["method"] == wanted:
                return float(row["MC1"])
        raise KeyError(f"{wanted} missing from {path}")
    return float(obj["metrics"]["MC1"])

def main():
    a = parse_args()
    rows = []
    for sigma in a.noise_levels:
        d = a.input_dir / f"noise_{tag(sigma)}"
        l2s = d / "l2s.json"
        paths = {
            "L2S-Linear": l2s,
            "L2S-SLERP": l2s,
            "S2S_L_Static": d / "s2s_l_static.json",
            "S2S_R": d / "s2s_r.json",
        }
        for method, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({"noise_sigma": float(sigma), "method": method, "MC1": read_mc1(path, method)})
    out = a.output or (a.input_dir / "mc1_by_noise.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["noise_sigma", "method", "MC1"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
