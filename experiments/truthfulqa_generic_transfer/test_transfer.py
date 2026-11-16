"""CPU checks for source isolation, TruthfulQA mean preservation, and fold averaging."""
from argparse import Namespace
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.truthfulqa_generic_transfer.collect import summarize, write_outputs
from experiments.truthfulqa_generic_transfer.prepare_checkpoint import prepare, validate_source

HERE = Path(__file__).resolve().parent
MODEL = "llama3.1-8B-Instruct"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("transfer_training")
    rng = np.random.default_rng(17)
    feature = directory / "truthful_qa.npz"
    # Two positives and two negatives exercise mean-positive pairing.
    np.savez(feature, activations=rng.normal(size=(32, 4)).astype(np.float32),
             labels=np.tile([1, 1, 0, 0], 8), q_indices=np.repeat(np.arange(8), 4),
             model_name=np.asarray(MODEL), layer=np.asarray(12), dataset_name=np.asarray("truthful_qa"))
    checkpoints = []
    for fold in (0, 1):
        name = f"source{fold}"
        output = directory / f"{name}.npz"
        subprocess.run([sys.executable, str(ROOT / "method/dyn3d_sga/train_dyn_mlp.py"),
                        "--feature_path", str(feature), "--output_path", str(output),
                        "--num_folds", "2", "--fold_idx", str(fold),
                        "--epochs", "1", "--mlp_hidden", "3", "--batch_size", "4",
                        "--device", "cpu", "--seed", "42"], check=True, capture_output=True, text=True)
        checkpoints.append(output)
    return feature, checkpoints


def transfer_args(trained, output_dir, fold):
    feature, checkpoints = trained
    return Namespace(truthfulqa_feature=feature, source_checkpoint=checkpoints[fold],
                     model=MODEL, layer=12, source_fold=fold, dataset="boolq",
                     output=output_dir / f"transfer{fold}.npz")


def test_transfer_preserves_source_weights_and_fold_specific_mean(trained, tmp_path):
    means, source_questions = [], []
    for fold in (0, 1):
        args = transfer_args(trained, tmp_path, fold)
        before = args.source_checkpoint.read_bytes()
        prepare(args)
        assert args.source_checkpoint.read_bytes() == before
        with np.load(args.output) as transfer, np.load(args.source_checkpoint) as source:
            for key in ("mlp_fc1.weight", "mlp_fc1.bias", "mlp_fc2.weight", "mlp_fc2.bias",
                        "v_avg", "mean_train_q_indices", "mlp_train_q_indices"):
                np.testing.assert_array_equal(transfer[key], source[key])
            assert str(transfer["v_avg_source_dataset"].item()) == "truthful_qa"
            assert str(transfer["mlp_source_dataset"].item()) == "truthful_qa"
            assert str(transfer["target_dataset"].item()) == "boolq"
            assert not any(key.startswith("generic_") for key in transfer.files)
            means.append(transfer["v_avg"].copy())
            source_questions.append(set(transfer["outer_train_q_indices"].tolist()))
    assert source_questions[0].isdisjoint(source_questions[1])
    assert not np.array_equal(means[0], means[1])


def test_incompatible_layer_and_fold_are_rejected(trained, tmp_path):
    with pytest.raises(ValueError, match="model/layer"):
        validate_source(trained[0], MODEL, 16)
    args = transfer_args(trained, tmp_path, 0)
    args.source_fold = 1
    with pytest.raises(ValueError, match="requested TruthfulQA fold"):
        prepare(args)


def test_fold_summary_requires_both_folds_and_reports_mean(trained, tmp_path):
    results = []
    for fold, score in ((0, 0.6), (1, 0.8)):
        args = transfer_args(trained, tmp_path, fold)
        prepare(args)
        path = tmp_path / f"eval{fold}.json"
        path.write_text(json.dumps({"MC1": score, "steering_enabled": True,
                                   "checkpoint_path": str(args.output), "model_name": MODEL,
                                   "dataset": "boolq", "layer": 12,
                                   "parameters": {"weight_kappa": 7, "margin": 0.9, "seed": 42}}))
        results.append(path)
    summary = summarize(results[::-1])
    assert summary["v_avg_source_dataset"] == "truthful_qa"
    assert summary["MC1_mean"] == pytest.approx(0.7)
    assert summary["MC1_std"] == pytest.approx(np.std([0.6, 0.8], ddof=1))
    assert [row["source_fold"] for row in summary["folds"]] == [0, 1]
    write_outputs([summary], tmp_path / "summary")
    assert (tmp_path / "summary/summary.csv").read_text().count("\n") == 2
    with pytest.raises(ValueError, match="Exactly one result"):
        summarize([results[0], results[0]])
    # Reject any mean that no longer matches the original TruthfulQA checkpoint.
    with np.load(tmp_path / "transfer1.npz") as data:
        payload = {key: data[key].copy() for key in data.files}
    payload["v_avg"] += 1
    np.savez(tmp_path / "transfer1.npz", **payload)
    with pytest.raises(ValueError, match="changed source TruthfulQA v_avg"):
        summarize(results)


def test_launcher_keeps_dataset_configs_independent(tmp_path):
    env = dict(os.environ, DRY_RUN="1", RUN_EXTRACT="auto", PYTHON_BIN=sys.executable,
               FEATURE_DIR=str(tmp_path / "features"), CHECKPOINT_ROOT=str(tmp_path / "checkpoints"),
               PROTOTYPE_DIR=str(tmp_path / "prototypes"), RESULT_ROOT=str(tmp_path / "results"))
    for key in ("DATASETS", "LAYER", "MARGIN", "WEIGHT_KAPPA", "NUM_FOLDS", "FOLD_IDX", "MODELS"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(HERE / "run.sh")], env=env, check=True,
                            capture_output=True, text=True)
    assert "get_activations_generic.py" not in result.stdout
    assert "--generic-feature" not in result.stdout
    commands = [shlex.split(line)[1:] for line in result.stdout.splitlines() if line.startswith("[dry-run]")]
    training = [cmd for cmd in commands if "method/dyn3d_sga/train_dyn_mlp.py" in cmd]
    evaluation = [cmd for cmd in commands if "method/dyn3d_sga/generic/evaluate_generic.py" in cmd]
    assert len(training) == len(evaluation) == 10
    expected = {"boolq": (12, 7, 0.9), "copa": (16, 10, 0), "storycloze": (27, 2, -0.5),
                "mmlu_global": (28, 3, 0.3), "winogrande": (16, 3, -0.8)}
    for cmd in training:
        assert cmd[cmd.index("--num_folds") + 1] == "2"
        assert "generic" not in Path(cmd[cmd.index("--feature_path") + 1]).name
    for cmd in evaluation:
        dataset = cmd[cmd.index("--dataset") + 1]
        actual = tuple(float(cmd[cmd.index(flag) + 1]) for flag in ("--layer", "--weight_kappa", "--margin"))
        assert actual == expected[dataset]
