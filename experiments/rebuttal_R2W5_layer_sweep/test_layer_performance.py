"""CPU tests for selection, cache safety and existing Llama steering integration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from results import SELECTION, aggregate, cells, valid_cell, write_json, write_summaries


def fixture_reports():
    reports = []
    # Fold-specific winners differ. The shared second tuple wins on fold mean.
    for strength, scores in ((0.5, (0.95, 0.25)), (1.0, (0.7, 0.7))):
        for fold, mc1 in enumerate(scores):
            reports.append({"selection": SELECTION, "context": {"layer": 14, "fold": fold},
                            "method": "S2S-L", "parameters": {"strength": strength, "slerp_t": 0.2},
                            "test_q_indices": [fold], "n_questions": 1,
                            "metrics": {"MC1": mc1, "MC2": mc1 / 2, "MC3": 0.3}})
    return reports


class SelectionTests(unittest.TestCase):
    def test_shared_tuple_after_fold_mean(self):
        rows, best = aggregate(fixture_reports())
        self.assertEqual(len(rows), 2)
        self.assertEqual(best[0]["strength"], 1)
        self.assertEqual(best[0]["MC1"], 0.7)
        self.assertEqual(best[0]["MC2"], 0.35)
        self.assertEqual(best[0]["selection"], SELECTION)

    def test_invalid_partitions_and_metrics(self):
        reports = fixture_reports()
        for invalid in (reports[:-1], reports + [reports[0]]):
            with self.assertRaises(ValueError):
                aggregate(invalid)
        for ids in ([0], [1, 1]):
            bad = copy.deepcopy(reports)
            bad[1]["test_q_indices"] = ids
            bad[1]["n_questions"] = len(ids)
            with self.assertRaises(ValueError):
                aggregate(bad)
        bad = copy.deepcopy(reports)
        bad[0]["metrics"]["MC1"] = float("nan")
        with self.assertRaises(ValueError):
            aggregate(bad)

    def test_resume_matches_context_and_parameters(self):
        row = fixture_reports()[0]
        self.assertTrue(valid_cell(row, row["context"], row["method"], row["parameters"]))
        self.assertFalse(valid_cell(row, {**row["context"], "layer": 31}, row["method"], row["parameters"]))
        self.assertFalse(valid_cell(row, row["context"], row["method"], {"strength": 2, "slerp_t": 0.2}))
        with tempfile.TemporaryDirectory() as directory:
            write_summaries(directory, fixture_reports(), {})
            report = json.loads((Path(directory) / "summaries/best_per_layer.json").read_text())
            self.assertEqual(report["rows"][0]["strength"], 1)
            self.assertTrue((Path(directory) / "summaries/all_grid_fold_mean.csv").exists())

    def test_default_count_and_no_write_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "uncreated"
            command = [sys.executable, HERE / "rebuttal_R2W5_run.py", "--dry-run", "--output-dir", output]
            result = subprocess.run(list(map(str, command)), capture_output=True, text=True, check=True)
            self.assertIn("fold-specific evaluations=4608", result.stdout)
            self.assertFalse(output.exists())
            invalid = subprocess.run(list(map(str, command + ["--layers", "32"])), capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)

    def test_complete_resume_without_model_import(self):
        import evaluate_grid
        config = {"model_name": "tiny", "model_dir": "tiny", "device": "cpu", "dtype": "float32",
                  "seed": 42, "code": "fixture", "methods": ["S2S-L"],
                  "grid": {"strength": [1], "slerp_t": [0.2]}}
        from results import file_identity
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoints = [output / f"fold{x}.npz" for x in (0, 1)]
            for p in checkpoints:
                p.write_bytes(b"fixture")
            config_path = output / "config.json"
            write_json(config_path, config)
            for method, params, fold, relative in cells(config, 14):
                context = {"layer": 14, "fold": fold, "checkpoint": file_identity(checkpoints[fold]),
                           **{k: config[k] for k in ("model_name", "model_dir", "device", "dtype", "seed", "code")},
                           "eval_split": "test", "prompt": "zero_shot_instruction"}
                write_json(output / relative, {"selection": SELECTION, "context": context, "method": method,
                           "parameters": params, "test_q_indices": [fold], "n_questions": 1,
                           "metrics": {"MC1": 0.5, "MC2": 0.5, "MC3": 0.5}})
            argv = ["evaluate_grid.py", "--config", str(config_path), "--layer", "14", "--checkpoint0", str(checkpoints[0]),
                    "--checkpoint1", str(checkpoints[1]), "--output", str(output), "--resume"]
            # Invalid checkpoint bytes would fail immediately if not fully reused.
            with patch.object(sys, "argv", argv):
                evaluate_grid.main()


class TinyLlamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from method.S2S_R.predictor import DirectVectorMLP
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.torch = torch
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                              num_hidden_layers=2, num_attention_heads=2,
                                              num_key_value_heads=2, max_position_embeddings=256)).eval()
        cls.predictor = DirectVectorMLP(16, 8, 16).eval()

    def test_extraction_training_and_grid_pipeline(self):
        import numpy as np
        import pandas as pd
        import evaluate_grid
        import extract_features
        from method.S2S_R.train_predictor import main as train
        torch = self.torch

        class Inputs(dict):
            def to(self, device):
                return Inputs({key: value.to(device) for key, value in self.items()})

        class Tokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"
            pad_token_id = 0

            def __call__(self, text, **kwargs):
                ids = torch.tensor([[1] + [3 + ord(c) % 28 for c in text]])
                result = Inputs(input_ids=ids)
                result.input_ids = ids
                return result

        dataset = {"prompts": np.array([f"Q: Q{q}? A: {a}" for q in range(4) for a in ("Yes.", "No.")]),
                   "labels": np.tile([1, 0], 4), "q_indices": np.repeat(np.arange(4), 2),
                   "choice_indices": np.tile([0, 1], 4), "choices": np.tile(["Yes.", "No."], 4)}
        frame = pd.DataFrame([{"Question": f"Q{q}?", "Correct Answers": "Yes.", "Incorrect Answers": "No.",
                               "Best Answer": "Yes.", "hf_idx": q} for q in range(4)])
        config = {"model_name": "llama3.1-8B-Instruct", "model_dir": "fixture", "device": "cpu",
                  "dtype": "float32", "seed": 42, "code": "fixture", "methods": ["S2S-L", "S2S-R"],
                  "grid": {"strength": [1.0], "slerp_t": [0.2], "weight_kappa": [3.0], "margin": [-0.7]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config_path = output / "config.json"
            feature = output / "feature.npz"
            checkpoints = [output / f"fold{fold}.npz" for fold in (0, 1)]
            write_json(config_path, config)
            with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=self.model), patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                with patch("method.S2S_R.data.get_truthfulqa_data", return_value=dataset), patch.object(
                        sys, "argv", ["extract", "--config", str(config_path), "--layer", "1", "--output", str(feature)]):
                    extract_features.main()
                with np.load(feature) as values:
                    self.assertEqual(values["activations"].shape, (8, 16))
                    self.assertEqual(str(values["extraction"]), "block_output_before_final_norm")
                for fold, checkpoint in enumerate(checkpoints):
                    argv = ["train", "--feature_path", str(feature), "--output_path", str(checkpoint),
                            "--fold_idx", str(fold), "--epochs", "1", "--mlp_hidden", "8", "--device", "cpu"]
                    with patch.object(sys, "argv", argv):
                        train()
                argv = ["grid", "--config", str(config_path), "--layer", "1", "--checkpoint0", str(checkpoints[0]),
                        "--checkpoint1", str(checkpoints[1]), "--output", str(output)]
                with patch("method.S2S_R.eval_utils.load_truthfulqa_mc_dataframe", return_value=(frame, range(4))), patch.object(sys, "argv", argv):
                    evaluate_grid.main()
            reports = [json.loads((output / relative).read_text()) for _, _, _, relative in cells(config, 1)]
            rows, best = aggregate(reports)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["method"] for row in best}, {"S2S-L", "S2S-R"})

    def test_real_mc_scoring_for_both_methods_and_last_layer(self):
        import pandas as pd
        import evaluate_grid
        from method.S2S_L.steering import get_s2s_l_intervention
        from method.S2S_R.evaluate_S2S_R import get_s2s_r_intervention
        torch = self.torch

        class Tokenizer:
            def __call__(self, text, **kwargs):
                # Sufficient vocabulary for deterministic toy answer likelihoods.
                from types import SimpleNamespace
                return SimpleNamespace(input_ids=torch.tensor([[1] + [3 + ord(c) % 28 for c in text]]))

        frame = pd.DataFrame([{"Question": "Q?", "Correct Answers": "Yes.",
                               "Incorrect Answers": "No.", "Best Answer": "Yes."}])
        config = {"model_name": "llama3.1-8B-Instruct", "device": "cpu"}
        for factory, params in ((get_s2s_l_intervention, {"strength": 1, "slerp_t": 0.2}),
                                (get_s2s_r_intervention, {"weight_kappa": 3, "margin": -0.7})):
            stats = {}
            hook = factory(torch.ones(16), self.predictor, stats=stats, **params)
            metrics = evaluate_grid.evaluate_one(self.model, Tokenizer(), frame, config, 1, hook)
            self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))
            self.assertGreater(stats["total"], 0)
        # Verify why the extractor must capture block output before final norm.
        captured = []
        handle = self.model.model.layers[-1].register_forward_hook(
            lambda _m, _i, out: captured.append((out[0] if isinstance(out, tuple) else out).detach()))
        try:
            with torch.inference_mode():
                output = self.model(torch.tensor([[1, 2, 3]]), output_hidden_states=True)
        finally:
            handle.remove()
        self.assertFalse(torch.allclose(captured[0], output.hidden_states[-1]))


if __name__ == "__main__":
    unittest.main()
