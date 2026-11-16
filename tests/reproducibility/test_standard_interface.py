"""Public-interface checks independent of model downloads or benchmark scores."""
import argparse
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reproducibility.cli import execute_plan, main, make_plan
from reproducibility.config import ROOT, load_profile, load_registry, main_parameters
from reproducibility.build_release import build
from reproducibility.verify_release import verify


def cli_plan(*arguments):
    stream = io.StringIO()
    with redirect_stdout(stream):
        main(["run", *arguments, "--dry-run"])
    return json.loads(stream.getvalue())


class PublicInterfaceTests(unittest.TestCase):
    def test_catalog_keeps_the_requested_main_and_rebuttal_scope(self):
        entries = load_registry()
        expected = {"table1", "table1-generation", "table2", "table3", "table4", "fig2a", "fig2bc",
                    "fig3a", "fig3b-hidden", "fig3b-fraction", "fig3c-margin", "fig3c-kappa",
                    "fig3d", "fig4ab", "fig5-noise", "r1-noise", "r1-scales", "r1-models",
                    "r2-layers", "r2-bipo", "r2-bipo-calibration", "r3-mean-control", "r3-gates",
                    "r3-margin", "mmsafe-generate", "mmsafe-score", "supp-transfer",
                    "supp-prediction-errors", "supp-vector-metrics", "supp-geometry"}
        self.assertTrue(expected <= entries.keys())
        self.assertEqual(entries["external-vectors"]["backend"], "unavailable")

    def test_all_available_routes_can_be_previewed_without_model_dependencies(self):
        for entry in load_registry().values():
            if entry["backend"] == "unavailable":
                continue
            with self.subTest(entry=entry["id"]):
                plan = cli_plan(entry["id"])
                self.assertTrue(plan["dry_run"])
                self.assertTrue(plan["commands"])
                for command in plan["commands"]:
                    argv = command["argv"]
                    if argv[1] != "-m":
                        self.assertTrue((ROOT / argv[1]).is_file())

    def test_preview_creates_no_runtime_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            plan = cli_plan("table1", "--model", "qwen", "--work-dir", str(output))
            self.assertEqual(plan["model"], "qwen")
            self.assertFalse(output.exists())

    def test_config_rejects_a_conflicting_model_family(self):
        with self.assertRaises(ValueError):
            load_profile(ROOT / "configs/reproduction/qwen.json", "llama")

    def test_profiles_preserve_fold_specific_reported_parameters(self):
        llama = load_profile(model="llama")
        qwen = load_profile(model="qwen")
        self.assertEqual(llama["training"]["epochs"], 2000)
        self.assertEqual(main_parameters(llama, 0)["s2s_r"]["margin"], -.7)
        self.assertEqual(main_parameters(qwen, 0)["s2s_r"]["margin"], -.3)
        self.assertEqual(main_parameters(qwen, 1)["s2s_r"]["margin"], -.7)
        self.assertEqual(main_parameters(qwen, 1)["l2s_alpha"], 1.4)

    def test_missing_judges_are_shown_before_execution(self):
        plan = cli_plan("table1-generation")
        self.assertEqual(set(plan["missing_inputs"]), {"truth_judge_dir", "info_judge_dir"})
        with self.assertRaises(ValueError):
            execute_plan(plan)

    def test_inference_noise_and_target_noise_route_to_distinct_programs(self):
        inference = cli_plan("r1-noise")
        target = cli_plan("fig5-noise")
        self.assertNotEqual(inference["commands"][0]["argv"], target["commands"][0]["argv"])
        self.assertEqual(inference["runtime"]["folds"], [0])
        self.assertEqual(target["runtime"]["folds"], [0, 1])

    def test_fraction_study_and_native_fold_environment_agree(self):
        plan = cli_plan("fig3b-fraction", "--folds", "0", "1")
        self.assertEqual([command["env"]["FOLD_IDX"] for command in plan["commands"]], ["0", "1"])
        self.assertEqual(cli_plan("fig3d")["commands"][0]["env"]["WEIGHT_KAPPAS"], "1 5 10 20")

    def test_layer_grid_uses_the_explicit_profile(self):
        self.assertEqual(cli_plan("r2-layers", "--search-profile", "full")["commands"][0]["env"]["SEARCH_PROFILE"], "full")

    def test_followup_sweeps_use_single_model_workers_on_the_requested_gpu(self):
        for identifier in ("r2-spherical-grid", "r2-fixed-t"):
            plan = cli_plan(identifier, "--gpu", "4")
            self.assertEqual(len(plan["commands"]), 7)
            for command in plan["commands"]:
                self.assertEqual(command["env"]["CUDA_VISIBLE_DEVICES"], "4")
                self.assertTrue("--model" in command["argv"] or "--summarize" in command["argv"])

    def test_geometry_sampling_seed_matches_the_native_protocol(self):
        plan = cli_plan("supp-geometry")
        self.assertEqual(plan["runtime"]["seed"], 20261111)
        command = plan["commands"][0]["argv"]
        self.assertEqual(command[command.index("--seed") + 1], "20261111")

    def test_fraction_study_preserves_its_multiple_training_seeds(self):
        plan = cli_plan("fig3b-fraction")
        self.assertEqual(plan["native_seed_grid"], [42,123,0])
        self.assertEqual(plan["commands"][0]["env"]["SEEDS"], "42 123 0")

    def test_paired_figure_plotting_is_a_separate_model_free_stage(self):
        for identifier in ("fig3a", "fig3b-hidden", "fig3b-fraction", "fig3c-margin", "fig3c-kappa"):
            plan = cli_plan(identifier)
            self.assertEqual(plan["commands"][0]["env"]["AUTO_PLOT"], "0")
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["plot", identifier, "--dry-run"])
            plotted = json.loads(stream.getvalue())
            self.assertFalse(plotted["needs_model"])
            self.assertTrue((ROOT / plotted["commands"][0]["argv"][1]).is_file())
            self.assertEqual(plotted["commands"][0]["env"]["PYTHON"], sys.executable)

    def test_review_scores_remain_explicit_external_inputs(self):
        plan = cli_plan("mmsafe-score")
        self.assertFalse(plan["needs_model"])
        self.assertEqual(set(plan["missing_inputs"]), {"generations", "scores", "judge_config"})

    def test_bundled_data_is_pinned_and_not_ignored_by_git(self):
        path = ROOT / "data/truthfulqa/TruthfulQA.csv"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "f9bd9e859cc102cb1f647f1064da7e009be752c416845cf9fa56e6eaae403a7d")
        with path.open() as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 817)
        if (ROOT / ".git").exists():
            result = subprocess.run(["git", "check-ignore", "-q", str(path.relative_to(ROOT))], cwd=ROOT)
            self.assertEqual(result.returncode, 1)

    def test_resume_flag_does_not_change_the_experiment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"experiment": "test", "model": "llama", "runtime": {"work_dir": directory, "model_dir": "unused", "seed": 42},
                    "missing_inputs": [], "required_files": [], "needs_model": False,
                    "commands": [{"argv": [sys.executable, "native.py"], "env": {"REUSE_EXISTING": "0"}}]}
            with patch("reproducibility.cli.subprocess.run") as run:
                execute_plan(plan)
                second = copy.deepcopy(plan)
                second["commands"][0]["argv"].append("--resume")
                second["commands"][0]["env"]["REUSE_EXISTING"] = "1"
                execute_plan(second)
                self.assertEqual(run.call_count, 2)
                changed = copy.deepcopy(second)
                changed["runtime"]["seed"] = 7
                with self.assertRaises(ValueError):
                    execute_plan(changed)

    def test_export_keeps_sources_and_references_but_not_history_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            payloads = {"README.md": "# Anonymous fixture\n", "methods/S2S_L/__init__.py": "",
                        "data/truthfulqa/TruthfulQA.csv": "Question\nexample\n",
                        "results/reference/evidence.json": '{"records": []}\n',
                        "models/weights.bin": "private runtime data", ".git/config": "local history"}
            for name, body in payloads.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
            target = Path(directory) / "export"
            build(source, target)
            self.assertFalse((target / ".git").exists())
            self.assertFalse((target / "models/weights.bin").exists())
            self.assertTrue((target / "results/reference/evidence.json").is_file())
            self.assertEqual((target / "data/truthfulqa/TruthfulQA.csv").read_bytes(), (source / "data/truthfulqa/TruthfulQA.csv").read_bytes())
            self.assertFalse(verify(target, strict=True))
            (target / "models").mkdir(exist_ok=True)
            (target / "models/local.bin").write_bytes(b"runtime")
            self.assertFalse(verify(target))
            self.assertTrue(verify(target, strict=True))


if __name__ == "__main__":
    unittest.main()
