"""Export compact numeric evidence and distinguish fold means from pooled scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("MC1", "MC2", "MC3")
GROUPS = ("model_scaling_ablation", "component_ablation", "bipo_truthfulqa",
          "l2s_noise_robustness", "margin_ablation", "margin_gate_correctness", "gate_correctness")
CLAIMS = (
    ("qwen3_s2sl_table_a", "Qwen2.5-3B", "s2s_l", 32.68),
    ("qwen3_s2sl_table_b", "Qwen2.5-3B", "s2s_l", 36.68),
    ("qwen14_s2sl_table_a", "Qwen2.5-14B", "s2s_l", 44.68),
    ("qwen14_s2sl_table_b", "Qwen2.5-14B", "s2s_l", 48.29),
    ("llama_noise_s2sl", "llama3.1-8B", "s2s_l", 49.02),
    ("q2_fold1_mean_control", "llama3.1-8B", "vavg_aw_gate", 38.88),
    ("review_bipo", "llama", "bipo", 60.18),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fold_id(path):
    match = re.search(r"fold[_-]?(\d)", str(path))
    return int(match.group(1)) if match else None


def method_id(value):
    value = value.lower().replace("-", "_")
    if value in ("s2s_l_static", "s2s_l", "dyn2d") or "dyn2d" in value:
        return "s2s_l"
    if value in ("dyn3d_sga", "s2s_r") or "similarity_graph" in value:
        return "s2s_r"
    if value == "l2s_linear":
        return "l2s"
    if value == "spherical_steering":
        return "spherical"
    return value


def numeric_metrics(value):
    return {key: float(value[key]) for key in METRICS
            if key in value and isinstance(value[key], (float, int)) and math.isfinite(value[key])}


def question_rows(data, csv_path=None):
    values = data.get("results", data.get("details", [])) if isinstance(data, dict) else data
    if not isinstance(values, list):
        values = []
    if not values and csv_path and csv_path.exists():
        with csv_path.open() as stream:
            values = list(csv.DictReader(stream))
    rows = []
    for item in values:
        if not isinstance(item, dict):
            continue
        q = item.get("hf_idx", item.get("q_idx", item.get("question_id")))
        if q is None:
            continue
        row = {"question_id": int(q)}
        for key in METRICS:
            candidates = [v for k, v in item.items() if k == key or k.endswith(" " + key)]
            if len(candidates) == 1 and candidates[0] not in (None, ""):
                row[key] = float(candidates[0])
        rows.append(row)
    return rows


def context_for(path, source):
    for folder in [path.parent, *list(path.parents)[:3]]:
        config = folder / "run_config.json"
        if config.is_file():
            d = json.loads(config.read_text())
            return {k: d[k] for k in ("repo_id", "revision", "dtype", "load_in_4bit", "layer", "seed", "selection", "prompt", "s2s_l_variant") if k in d}
    return {}


def extract(path, source):
    d = json.loads(path.read_text())
    rel = str(path.relative_to(source))
    parts = path.relative_to(source).parts
    raw_method = parts[2] if parts[:2] == ("results", "truthfulqa") else "unknown"
    model = parts[3] if len(parts) > 3 and parts[1] == "truthfulqa" else "unknown"
    context = context_for(path, source)
    base = dict(source=rel, source_sha256=sha(path), model=context.get("repo_id", model),
                method=method_id(raw_method), native_fold=fold_id(path), context=context)
    records = []

    def append(metrics, row=None, suffix="", qrows=None):
        row = row or {}
        values = numeric_metrics(metrics)
        if not values:
            return
        record = dict(base)
        record.update(metrics=values, method=method_id(str(row.get("method", row.get("variant", record["method"])))),
                      parameters=row.get("parameters", {k:row[k] for k in ("weight_kappa", "margin", "strength", "slerp_t") if k in row}), n_questions=row.get("n_questions"))
        f = row.get("fold", row.get("fold_idx", record["native_fold"]))
        record["native_fold"] = f
        record["fold"] = 1 - f if record["method"] == "spherical" and "results/truthfulqa/" in rel and f is not None else f
        if qrows:
            ids = [r["question_id"] for r in qrows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Repeated questions in {rel}")
            record["n_questions"] = len(ids)
            record["question_ids"] = sorted(ids)
            record["question_metrics"] = qrows
            for key, value in values.items():
                if all(key in q for q in qrows) and abs(sum(q[key] for q in qrows) / len(qrows) - value) > 1e-7:
                    raise ValueError(f"Stored mean does not match question scores: {rel} {key}")
        record["id"] = hashlib.sha256((rel + "#" + suffix).encode()).hexdigest()[:16]
        records.append(record)

    if isinstance(d, list):
        if d and all(isinstance(x, dict) and "method" in x for x in d):
            for index, row in enumerate(d):
                base["model"] = "llama3.1-8B-Instruct"
                append(row, row, str(index))
        elif d and isinstance(d[0], dict) and "q_idx" in d[0]:
            qrows = question_rows(d)
            if qrows and all(all(k in q for k in METRICS) for q in qrows):
                method = path.stem.removeprefix("test_").removeprefix("test_retuned_")
                base.update(model="llama3.1-8B-Instruct", method=method_id(method))
                selection = path.parent / ("selection_" + method + ".json")
                parameters = json.loads(selection.read_text()).get("chosen", {}).get("parameters", {}) if selection.exists() else {}
                append({k: sum(q[k] for q in qrows) / len(qrows) for k in METRICS},
                       {"parameters": parameters}, qrows=qrows)
        return records
    if not isinstance(d, dict):
        return []
    meta = d.get("metadata", d.get("context", {}))
    meta = meta.get("context", meta)
    base["model"] = d.get("model_name", d.get("model", meta.get("model_name", base["model"])))
    if not isinstance(base["model"], str):
        base["model"] = context.get("repo_id", "unknown")
    if "fold_idx" in d or "fold_idx" in meta:
        base["native_fold"] = d.get("fold_idx", meta.get("fold_idx"))
    if "parameters" in d:
        base["context"] = {**context, "parameters": d["parameters"]}
    for key in ("layer", "eval_split", "definition", "definitions", "eval_mode", "seed", "kappa", "alpha", "beta", "no_intervention"):
        if key in d or key in meta:
            base["context"][key] = d.get(key, meta.get(key))
    qrows = question_rows(d, path.with_name(path.stem + "_details.csv"))
    if "metrics" in d:
        if base["method"] == "unknown" and "method" in d:
            base["method"] = method_id(str(d["method"]))
        append(d["metrics"], {**d, "parameters": d.get("parameters", {})}, qrows=qrows)
    if isinstance(d.get("row"), dict):
        append(d["row"], d["row"], "fixed_grid_cell")
        if records:
            records[-1]["question_ids"] = meta.get("question_ids", [])
            records[-1]["n_questions"] = len(meta.get("question_ids", [])) or None
    for index, row in enumerate(d.get("rows", [])):
        if isinstance(row, dict):
            append(row, row, "row" + str(index))
    for index, row in enumerate(d.get("folds", [])):
        if isinstance(row, dict) and "metrics" in row:
            qs = question_rows(row)
            append(row["metrics"], {**row, "n_questions": len(row.get("question_ids", []))},
                   "fold" + str(index), qrows=qs)
            if row.get("question_ids"):
                records[-1]["question_ids"] = sorted(row["question_ids"])
    if isinstance(d.get("best"), dict):
        append(d["best"], d["best"], "test_grid_best")
        if records:
            records[-1]["context"]["selection_warning"] = "Best observed test-grid result; not an independent held-out estimate"
            records[-1]["context"]["complete"] = d.get("metadata", {}).get("complete")
    for margin, summaries in d.get("summaries", {}).items():
        if not isinstance(summaries, dict):
            continue
        for scope, row in summaries.items():
            if isinstance(row, dict) and "MC1" in row:
                append(row, {"method": "gate_diagnostic", "fold": fold_id(scope),
                             "parameters": {"margin": float(margin)}}, f"margin{margin}-{scope}")
                records[-1]["gate_rates"] = {k: v for k, v in row.items() if "rate" in k}
    return records


def aggregate_folds(records):
    if len(records) != 2 or {r["fold"] for r in records} != {0, 1}:
        raise ValueError("Exactly folds 0 and 1 are required")
    a, b = sorted(records, key=lambda r: r["fold"])
    if not a.get("n_questions") or not b.get("n_questions"):
        raise ValueError("Need per-fold question counts")
    if a.get("question_ids") and b.get("question_ids") and set(a["question_ids"]) & set(b["question_ids"]):
        raise ValueError("Test folds overlap")
    n = a["n_questions"] + b["n_questions"]
    return {**{f"macro_{k}": (a["metrics"][k] + b["metrics"][k]) / 2 for k in METRICS},
            **{f"pooled_{k}": (a["metrics"][k] * a["n_questions"] + b["metrics"][k] * b["n_questions"]) / n for k in METRICS},
            "n_questions": n}


def sources(source):
    for folder in [source / "results/truthfulqa", *[source / "results/experiment" / g for g in GROUPS]]:
        for p in sorted(folder.rglob("*.json")):
            rel = str(p.relative_to(folder))
            if any(x in rel.lower() for x in ("smoke", "launcher", "status", "download", "vmf_grid", "llm_judge")):
                continue
            if "cells" in p.parts and p.name not in ("kappa3.0_margin-0.7.json", "kappa2.5_margin0.965.json"):
                continue
            if any(x in p.parts for x in ("dev", "checkpoints", "features", "offload")) or p.name.startswith("dev_"):
                continue
            if p.name.endswith("selection.json") or p.name.startswith("selection_"):
                continue
            yield p


def write_audit(source, output, sanitizer=lambda s: s):
    source, output = Path(source), Path(output)
    records, errors = [], []
    for path in sources(source):
        try:
            records.extend(extract(path, source))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(dict(source=str(path.relative_to(source)), issue=type(exc).__name__, message=str(exc)))
    # An export must not include machine paths, even when present in diagnostics.
    records = json.loads(sanitizer(json.dumps(records, ensure_ascii=False)))
    errors = json.loads(sanitizer(json.dumps(errors, ensure_ascii=False)))
    write_json(output / "evidence.json", dict(units="MC fractions in [0,1]", records=records, validation_errors=errors))
    per_fold, averages = [], []
    for model in ("llama3.1-8B-Instruct", "Qwen2.5-7B-Instruct"):
        paths = dict(l2s="l2s_linear", spherical="spherical_steering", s2s_r="dyn3d_sga",
                     s2s_l="S2S_L_Static" if model.startswith("llama") else "dyn2d")
        for method, folder in paths.items():
            chosen = [r for r in records if r["source"].startswith(f"results/truthfulqa/{folder}/{model}/")]
            if len(chosen) != 2:
                per_fold.append(dict(model=model, method=method, status="missing_or_ambiguous", source=""))
                continue
            for r in sorted(chosen, key=lambda x: x["fold"]):
                per_fold.append(dict(model=model, method=method, fold=r["fold"], native_fold=r["native_fold"],
                                     n_questions=r["n_questions"], **r["metrics"], status="archived", source=r["source"]))
            try:
                averages.append(dict(model=model, method=method, **aggregate_folds(chosen)))
            except ValueError as exc:
                errors.append(dict(source=f"main/{model}/{method}", issue=str(exc)))
        for method in ("baseline", "additive"):
            per_fold.append(dict(model=model, method=method, status="original_artifact_not_found; run main suite", source=""))
    write_csv(output / "main_per_fold.csv", per_fold,
              ["model", "method", "fold", "native_fold", "n_questions", *METRICS, "status", "source"])
    write_csv(output / "main_averages.csv", averages,
              ["model", "method", "n_questions", *["macro_" + k for k in METRICS], *["pooled_" + k for k in METRICS]])
    claims = []
    for key, model, method, percent in CLAIMS:
        matches = [dict(evidence_id=r["id"], source=r["source"], model=r["model"], method=r["method"],
                        fold=r["fold"], context=r["context"], parameters=r["parameters"])
                   for r in records if model.casefold() in r["model"].casefold() and r["method"] == method
                   and abs(r["metrics"].get("MC1", -100) * 100 - percent) <= 0.005001]
        claims.append(dict(claim_id=key, claimed_MC1_percent=percent,
                           status="matching_artifact_found" if matches else "unresolved_no_matching_audited_artifact",
                           matches=matches))
    write_json(output / "number_provenance.json", dict(claims=claims,
        note="A numerical match is evidence of a configuration, not proof of which response table was intended."))
    report = ["# Archived result audit", "", "All MC values in CSV/JSON are fractions, not percentages.",
              "This audit reads existing files; it does not rerun models or certify the paper tables.", "",
              "| Model | Method | Fold 0 MC1 (%) | Fold 1 MC1 (%) | Macro MC1 (%) | Pooled MC1 (%) |",
              "|---|---|---:|---:|---:|---:|"]
    for a in averages:
        rs = sorted([r for r in per_fold if r.get("model") == a["model"] and r.get("method") == a["method"] and "MC1" in r], key=lambda r:r["fold"])
        report.append(f"| {a['model']} | {a['method']} | {rs[0]['MC1']*100:.2f} | {rs[1]['MC1']*100:.2f} | {a['macro_MC1']*100:.2f} | {a['pooled_MC1']*100:.2f} |")
    report += ["", "## Reviewer-number provenance", ""]
    for claim in claims:
        report.append(f"- `{claim['claim_id']}` ({claim['claimed_MC1_percent']:.2f}%): {claim['status']}; {len(claim['matches'])} matching record(s).")
    report += ["", "## Limitations", "", "- Baseline/additive original main-table files are absent from the audited paths; the main suite provides both-fold reruns.",
               "- Historical Spherical fold labels are inverted in the aligned table, using its opposite test-half convention.",
               "- Main-table aggregation is an arithmetic fold mean; scaling and BiPO use question-weighted pooling. Both are retained.",
               "- The archived Llama S2S-L fold-0 value is 49.02%; its two-fold macro mean is 43.95%.",
               "- BiPO files currently present may differ from the 60.18% review snapshot. Do not substitute one snapshot silently.",
               "- MM safe dataset, original safety judge, and original generation records were not found; the new generation runner cannot establish the historical 79%/31% claim.",
               "- Test-grid-selected component controls are labeled exploratory and kept separate from fixed-parameter controls.",
               f"- Artifact validation errors: {len(errors)}. See evidence.json; invalid files are excluded."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    write_json(output / "audit_status.json", dict(record_count=len(records), validation_errors=errors,
        model_execution="not_performed", unresolved_claims=[c["claim_id"] for c in claims if not c["matches"]]))
    return dict(records=len(records), errors=len(errors), unresolved=[c["claim_id"] for c in claims if not c["matches"]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts")
    a = p.parse_args()
    from reproducibility.build_release import sanitize
    print(json.dumps(write_audit(a.source, a.output, lambda s: sanitize(s, a.source.resolve())), indent=2))


if __name__ == "__main__":
    main()
