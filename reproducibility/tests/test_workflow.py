"""No-model tests for plans, evidence validation, anonymization, and safety metrics."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reproducibility.run import build_plan
from reproducibility.audit_results import aggregate_folds, extract
from reproducibility.build_release import sanitize
from reproducibility.generation import read_prompts
from reproducibility.privacy import audit_tree
from reproducibility.safety import summarize

ROOT = Path(__file__).resolve().parents[2]


def args(suite="main", **kwargs):
    values = dict(suite=suite, model="llama", model_dir="models/Llama", model_root=None,
                  work_dir=Path("results/test-plan"), folds=[0, 1], gpu="0", device="cuda",
                  dtype="float16", epochs=2000, bipo_epochs=20, seed=42, prompts=None,
                  scaling_models=["qwen05", "qwen15", "qwen3", "qwen14"],
                  datasets=["boolq", "copa", "storycloze", "mmlu_global", "winogrande"])
    values.update(kwargs)
    return argparse.Namespace(**values)


class ReproductionTests(unittest.TestCase):
    def test_all_main_methods_cover_both_folds(self):
        jobs = build_plan(args())
        main = [j for j in jobs if j["name"].startswith("main-")]
        self.assertEqual(len(main), 12)
        self.assertEqual(len({j["name"] for j in jobs}), len(jobs))
        for method in ("baseline", "additive", "l2s", "spherical", "s2s_l", "s2s_r"):
            self.assertEqual(sum(j["name"].endswith("-" + method) for j in main), 2)

    def test_native_spherical_folds_are_reversed(self):
        for job in build_plan(args()):
            if job["name"].endswith("-spherical"):
                fold = 0 if "-0-spherical" in job["name"] else 1
                i = job["command"].index("--prototype_path")
                self.assertTrue(job["command"][i + 1].endswith(f"_fold{1-fold}.npz"))

    def test_q2_includes_defaults_and_the_exact_review_cell_on_both_folds(self):
        jobs = build_plan(args("component"))
        fixed = [j for j in jobs if j["name"].startswith("component-")]
        review = [j for j in jobs if j["name"].startswith("review-control-")]
        self.assertEqual(len(fixed), 2)
        self.assertEqual(len(review), 2)
        for job in fixed:
            self.assertIn("vavg_aw_gate", job["command"])
            self.assertEqual(job["command"][job["command"].index("--margin") + 1], "-0.7")
        for job in review:
            command = job["command"]
            self.assertEqual(command[command.index("--weight_kappa") + 1], "2.5")
            self.assertEqual(command[command.index("--margin") + 1], "0.965")

    def test_margin_includes_default_and_gate_closed(self):
        jobs = build_plan(args("margin"))
        values = {j["command"][j["command"].index("--margin") + 1]
                  for j in jobs if j["name"].startswith("margin-") and "--margin" in j["command"]}
        self.assertIn("-0.7", values)
        self.assertIn("1.0", values)

    def test_noise_includes_clean_targets_on_both_folds(self):
        jobs = [j for j in build_plan(args("noise")) if j["name"].startswith("noise-")]
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(j["env"]["NOISE_LEVELS"].split()[0] == "0" for j in jobs))

    def test_scaling_precision_is_not_silently_mixed(self):
        jobs = build_plan(args("scaling"))
        for job in jobs:
            self.assertEqual("--load-in-4bit" in job["command"], job["name"] == "scaling-qwen14")

    def test_generic_includes_five_datasets_and_baselines(self):
        jobs = build_plan(args("generic"))
        self.assertEqual(len(jobs), 25)
        self.assertEqual(sum("--disable_steering" in j["command"] for j in jobs), 5)

    def test_dry_plan_needs_no_ml_dependencies(self):
        result = subprocess.run([sys.executable, "-m", "reproducibility.run", "main"],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["mode"], "plan_only")

    def test_weighted_pooling_and_overlap_rejection(self):
        rows = [dict(fold=f, n_questions=n, question_ids=ids, metrics={k:v for k in ("MC1", "MC2", "MC3")})
                for f,n,ids,v in [(0,2,[0,1],1), (1,1,[2],0)]]
        report = aggregate_folds(rows)
        self.assertAlmostEqual(report["macro_MC1"], .5)
        self.assertAlmostEqual(report["pooled_MC1"], 2/3)
        rows[1]["question_ids"] = [1]
        with self.assertRaises(ValueError):
            aggregate_folds(rows)

    def test_extraction_rejects_inconsistent_saved_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "results/truthfulqa/dyn3d_sga/model/fold0.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(metrics=dict(MC1=.2), results=[dict(hf_idx=0, MC1=1)])))
            with self.assertRaises(ValueError):
                extract(path, root)

    def test_collector_rejects_mismatched_method_question_sets(self):
        from reproducibility.collect import collect, METHODS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in METHODS:
                for fold, ids in ((0,[0,1]),(1,[2,3])):
                    path = root / "main/toy" / f"fold{fold}" / f"{method}.json"
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(dict(metrics={k:.5 for k in ("MC1","MC2","MC3")},
                        results=[dict(hf_idx=q,MC1=.5,MC2=.5,MC3=.5) for q in ids])))
            self.assertEqual(collect(root,root/"report",["toy"],4)["method_pairs"],6)
            path=root/"main/toy/fold0/l2s.json"
            data=json.loads(path.read_text());data["results"][0]["hf_idx"]=10;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                collect(root,root/"report2",["toy"],4)

    def test_personal_paths_are_sanitized_and_no_secret_value_is_echoed(self):
        private = "/" + "home" + "/" + "private-user" + "/models/Model"
        clean = sanitize(private, ROOT)
        self.assertEqual(clean, "models/Model")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.txt"
            path.write_text(private + "\n" + "hf_" + "z" * 32)
            findings = audit_tree(Path(temporary))
            self.assertEqual(len(findings), 2)
            self.assertNotIn("private-user", json.dumps(findings))
            self.assertNotIn("z" * 32, json.dumps(findings))

    def test_unsafe_release_payloads_and_added_git_history_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            (root / "weights.npz").write_bytes(b"test")
            self.assertEqual(len(audit_tree(root)), 2)

    def test_prompt_ids_must_be_unique_and_multimodal_input_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text('{"id":"a","prompt":"x"}\n{"id":"a","prompt":"y"}\n')
            with self.assertRaises(ValueError):
                read_prompts(path)
            path.write_text('{"id":"a","prompt":"x","image":"image.png"}\n')
            with self.assertRaises(ValueError):
                read_prompts(path)

    def test_safety_threshold_equality_and_length_accounting(self):
        rows = [dict(id=str(i), method=m, generated_tokens=200 if i else 12, hit_token_limit=bool(i))
                for m in ("baseline", "s2s_r") for i in range(2)]
        scores = [dict(id=r["id"], method=r["method"], unsafe_probability=.5 if r["id"] == "1" else .1) for r in rows]
        result = summarize(rows, scores, [.5])
        self.assertTrue(all(r["safe_rate"] == .5 and r["token_limit_rate"] == .5 for r in result))
        with self.assertRaises(ValueError):
            summarize(rows, scores[:-1], [.5])


if __name__ == "__main__":
    unittest.main()
