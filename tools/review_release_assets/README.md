# S2S anonymous reproduction

Commands run from the repository root. Python launchers print a plan unless `--execute` is present.
Paper/rebuttal mapping and missing external inputs: [`EXPERIMENT_INDEX.csv`](EXPERIMENT_INDEX.csv).

Verify the clean checkout before creating model/output files:

```bash
python -m reproducibility.verify_release . --allow-git
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/download_model.py --repo-id meta-llama/Llama-3.1-8B-Instruct --output models/Llama
python tools/download_model.py --repo-id Qwen/Qwen2.5-7B-Instruct --output models/Qwen

export PYTHON_BIN=python GPU=0
export LLAMA_MODEL=llama3.1-8B-Instruct
export LLAMA_MODEL_DIR="$PWD/models/Llama" QWEN_MODEL_DIR="$PWD/models/Qwen"
export FEATURE_DIR="$PWD/results/reproduction/features"
export CHECKPOINT_ROOT="$PWD/results/reproduction/checkpoints"
export PROTOTYPE_DIR="$PWD/results/reproduction/prototypes"
export RESULT_ROOT="$PWD/results"
```

## Table 1 — TruthfulQA, both folds

```bash
python -m reproducibility.run main --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume
python -m reproducibility.run main --model qwen --model-dir "$QWEN_MODEL_DIR" --execute --resume

# Supply the original local TruthfulQA truth/info judge models.
export TRUTH_JUDGE_DIR="$PWD/models/truthfulqa-truth-judge-llama2-7B"
export INFO_JUDGE_DIR="$PWD/models/truthfulqa-info-judge-llama2-7B"
python experiments/table1_truthfulqa/table1_run_generation.py --model llama --model-dir "$LLAMA_MODEL_DIR" --truth-judge-dir "$TRUTH_JUDGE_DIR" --info-judge-dir "$INFO_JUDGE_DIR" --execute
python experiments/table1_truthfulqa/table1_run_generation.py --model qwen --model-dir "$QWEN_MODEL_DIR" --truth-judge-dir "$TRUTH_JUDGE_DIR" --info-judge-dir "$INFO_JUDGE_DIR" --execute
```

## Table 2 — Task-specific generic training

```bash
python -m reproducibility.run generic --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume
# Optional Qwen comparison:
python -m reproducibility.run generic --model qwen --model-dir "$QWEN_MODEL_DIR" --execute --resume
```

## Tables 3–4 — Components and throughput

```bash
# Includes both folds and the review's kappa=2.5, margin=0.965 mean-only control.
python -m reproducibility.run component --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume

S2SL_CHECKPOINT="$CHECKPOINT_ROOT/S2S_R/truthfulqa/$LLAMA_MODEL/${LLAMA_MODEL}_layer14_predictor_fold0_mean.npz" \
S2SR_CHECKPOINT="$CHECKPOINT_ROOT/S2S_R/truthfulqa/$LLAMA_MODEL/${LLAMA_MODEL}_layer14_predictor_fold0_mean.npz" \
SPHERICAL_PROTOTYPE="$PROTOTYPE_DIR/$LLAMA_MODEL/${LLAMA_MODEL}_layer14_fold1.npz" \
  bash experiments/table4_generation_efficiency/run.sh
```

## Figure 2 — Oracle reference and learned-vector errors

```bash
# Label-informed train-fold diagnostic; not a deployable method or certified historical curve.
python experiments/fig2a_oracle_vectors/run_paper.py --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume
python experiments/fig2bc_l2s_reliability/run_paper.py --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume
```

## Figure 3 — Hyperparameters and weights

```bash
AUTO_PLOT=1 bash experiments/fig3a_s2s_l_hyperparameters/run.sh
RUN_EXTRACT=1 AUTO_PLOT=1 bash experiments/fig3b_hidden_size/run.sh
RUN_EXTRACT=1 AUTO_PLOT=1 bash experiments/fig3b_predictor_fraction/run.sh
RUN_EXTRACT=1 AUTO_PLOT=1 bash experiments/fig3c_gate_margin/run.sh
RUN_EXTRACT=1 AUTO_PLOT=1 bash experiments/fig3c_weight_sharpness/run.sh
WEIGHT_KAPPAS="1 5 10 20" bash experiments/fig3d_adaptive_weights/plot.sh
```

## Figure 4(a,b) — Residual transfer

```bash
python experiments/fig4ab_residual_transfer/run_paper.py --model llama --model-dir "$LLAMA_MODEL_DIR" --execute
python experiments/fig4ab_residual_transfer/run_paper.py --model qwen --model-dir "$QWEN_MODEL_DIR" --execute
```

## Figure 5 middle — Training-target noise

```bash
python -m reproducibility.run noise --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume
```

## Rebuttal R1 W1 — Inference residual noise

```bash
# Historical response table: fold 0. Use --folds 0 1 for both folds.
python experiments/rebuttal_R1W1_inference_noise/run.py --model llama --model-dir "$LLAMA_MODEL_DIR" --folds 0 --execute --resume
```

## Rebuttal R1 W2 / R2 W4,W6 — Models, scales, and fixed t

```bash
python -m venv .venv-scaling
.venv-scaling/bin/pip install -r requirements-scaling.txt
export SCALING_PY="$PWD/.venv-scaling/bin/python"

for model in qwen05 qwen15 qwen3; do
  "$SCALING_PY" experiments/rebuttal_R1W2_model_scaling/run.py --model "$model"
done
"$SCALING_PY" experiments/rebuttal_R1W2_model_scaling/run.py --model qwen14 --load-in-4bit
"$SCALING_PY" experiments/rebuttal_R1W2_model_scaling/run.py --model qwen35 --epochs 250 --output-dir results/experiment/rebuttal_R1W2_model_scaling/Qwen3.5-9B/bfloat16_fullgpu_e250
"$SCALING_PY" experiments/rebuttal_R1W2_model_scaling/run.py --model nemotron_nano8b --epochs 250 --output-dir results/experiment/rebuttal_R1W2_model_scaling/Llama-3.1-Nemotron-Nano-8B-v1/bfloat16_fullgpu_e250

# After the source runs (and Figure 3a for the 7B/8B comparisons):
"$SCALING_PY" experiments/rebuttal_R1W2_model_scaling/retune_spherical.py
"$SCALING_PY" experiments/rebuttal_R1W2_model_scaling/compare_slerp.py
```

## Rebuttal R2 W5 — Layer sweep

```bash
GPU=0 SEARCH_PROFILE=fast bash experiments/rebuttal_R2W5_layer_sweep/run_llama.sh
# Larger exploratory test grid:
GPU=0 SEARCH_PROFILE=full bash experiments/rebuttal_R2W5_layer_sweep/run_llama.sh
```

## Rebuttal BiPO — Standalone baseline and learned-anchor calibration

```bash
python experiments/rebuttal_R2Q4_bipo/run.py --model-dir "$LLAMA_MODEL_DIR" --feature "$FEATURE_DIR/${LLAMA_MODEL}_layer14.npz"
python experiments/rebuttal_R2Q4_bipo/analyze.py
python experiments/rebuttal_R2Q4_bipo/run_mlp_s2s.py --model-dir "$LLAMA_MODEL_DIR" --feature "$FEATURE_DIR/${LLAMA_MODEL}_layer14.npz"
```

## Rebuttal R3 Q3 — Correct/incorrect candidate gates

```bash
python -m reproducibility.run gate --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume
python -m reproducibility.run margin --model llama --model-dir "$LLAMA_MODEL_DIR" --execute --resume
```

## Rebuttal MMsafe — Generation, truncation, and safety

```bash
# Requires original text prompts and original judge scores/configuration.
python -m reproducibility.run generation --model llama --model-dir "$LLAMA_MODEL_DIR" --prompts data/mmsafe.jsonl --execute --resume
for fold in 0 1; do
  python -m reproducibility.safety \
    --generations "results/reproduction/generation/$LLAMA_MODEL/fold${fold}/generations.jsonl" \
    --scores "data/mmsafe_scores_fold${fold}.jsonl" --judge-config data/mmsafe_judge.json \
    --thresholds 0.5 0.7 0.9 --output-dir "results/mmsafe/fold${fold}"
done
```

MMsafe file fields: prompts = `id,prompt`; judge scores = `id,method,unsafe_probability`; judge config = `judge_id,revision,input_format,unsafe_score_definition`.

## Additional analyses listed in the supplied configuration supplement

```bash
# Requires Table 2 training artifacts first.
RUN_EXTRACT=1 bash experiments/rebuttal_R2Q2_cross_task_transfer/run.sh
python experiments/rebuttal_prediction_error/run_official.py
python experiments/rebuttal_prediction_error/run_reliability.py
python experiments/rebuttal_geometry_diagnostics/run.py --model-dir "$LLAMA_MODEL_DIR"
```

## Checks

```bash
python -m unittest discover -s reproducibility/tests -v
```

Original MMsafe inputs/judge, external ITI/SADI/LoFiT/LoReFT implementations, and original Figure 4(c)/Figure 5-left artifacts are not bundled. Schematics require no model run. Archived scores are not a new GPU execution; unresolved records remain identified in `artifacts/`.
