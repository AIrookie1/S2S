"""Validate complete matched test coverage and collect newly executed main results."""
import argparse
from pathlib import Path

from reproducibility.audit_results import METRICS, aggregate_folds, extract, write_csv, write_json

METHODS = ("baseline", "additive", "l2s", "spherical", "s2s_l", "s2s_r")


def collect(work_dir, output, models, expected_questions=817):
    work_dir, output = Path(work_dir).resolve(), Path(output)
    per_fold, averages, records = [], [], []
    for model in models:
        matched = {}
        for method in METHODS:
            pair = []
            for fold in (0, 1):
                path = work_dir / "main" / model / f"fold{fold}" / f"{method}.json"
                if not path.is_file():
                    raise ValueError("Missing main result: " + str(path))
                found = extract(path, work_dir)
                if len(found) != 1:
                    raise ValueError("Expected exactly one aggregate per main result")
                row = found[0]
                row.update(model=model, method=method, fold=fold)
                # New output paths already use aligned folds. Native Spherical's
                # checkpoint label remains recoverable from the reproduction plan.
                row["native_fold"] = 1-fold if method == "spherical" else fold
                if not row.get("question_ids") or not all(k in row["metrics"] for k in METRICS):
                    raise ValueError("Missing question identities/metrics: " + str(path))
                ids = set(row["question_ids"])
                if fold in matched and matched[fold] != ids:
                    raise ValueError("Methods evaluated different held-out questions")
                matched[fold] = ids
                pair.append(row)
                records.append(row)
                per_fold.append(dict(model=model, method=method, fold=fold, native_fold=row["native_fold"],
                                     n_questions=row["n_questions"], **row["metrics"]))
            aggregate = aggregate_folds(pair)
            if aggregate["n_questions"] != expected_questions:
                raise ValueError("Incomplete benchmark coverage for " + model + "/" + method)
            averages.append(dict(model=model, method=method, **aggregate))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "main_per_fold.csv", per_fold, ["model", "method", "fold", "native_fold", "n_questions", *METRICS])
    write_csv(output / "main_averages.csv", averages, ["model", "method", "n_questions", *["macro_"+k for k in METRICS], *["pooled_"+k for k in METRICS]])
    write_json(output / "main_evidence.json", dict(units="MC fractions", records=records))
    return dict(models=models, method_pairs=len(averages), expected_questions=expected_questions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, default=Path("results/reproduction"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--expected-questions", type=int, default=817)
    a = p.parse_args()
    print(collect(a.work_dir, a.output, a.models, a.expected_questions))


if __name__ == "__main__":
    main()
