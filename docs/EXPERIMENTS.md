# Supported experiment commands and output contracts

All commands below run from the repository root. A command without `--execute`
prints its complete plan, including arguments, expected outputs, and explicit
environment overrides. Runtime dependencies are imported by the child evaluators,
not by the planner.

## Main experiments

```bash
python -m reproducibility.run main --model llama --model-dir models/Llama --execute --resume
python -m reproducibility.run main --model qwen --model-dir models/Qwen --execute --resume
python -m reproducibility.run generic --model llama --model-dir models/Llama --execute --resume
python -m reproducibility.run generic --model qwen --model-dir models/Qwen --execute --resume
```

`main` extracts one answer-level feature artifact per model; fits independent
fold predictors for L2S and S2S; computes native Spherical prototypes; and evaluates
six methods on the held-out questions in both folds. S2S-L and S2S-R deliberately
share the same direct-vector predictor under this fixed training protocol. Their
intervention geometries remain different. Additive mean uses the raw class-centroid
difference with alpha=1.0; L2S uses alpha=1.0. S2S-L is explicitly `S2S_L_Static`,
strength=3.0, t=0.2. S2S-R uses kappa=3, margin=-0.7. No test-set hyperparameter
search is performed by the main runner.

Main outputs: `results/reproduction/main/<model>/fold{0,1}/{method}.json`, plus
native detailed CSVs where supported. When both folds finish, the runner automatically
validates matched question IDs across all six methods and writes `reports/<model>/main_per_fold.csv`,
`main_averages.csv` (macro and pooled), and compact evidence. Native Spherical's fold label is reversed
when choosing the prototype. The actual test question sets, not filenames alone,
should define per-fold comparisons. Its historical evaluator is FP16-only, so the
main suite rejects a mixed-precision request instead of silently mixing dtypes.

`generic` preserves each method's per-dataset layer and hyperparameters. It
extracts/trains/evaluates from scratch and includes a no-steering reference using
the Dyn3D generic evaluator. Use `--datasets boolq copa` for a subset. These tasks
use their own official train/evaluation split; they are not TruthfulQA two-fold runs.

## Q2: mean + AW + Gate, on BOTH folds

```bash
python -m reproducibility.run component --model llama --model-dir models/Llama --execute --resume
```

For each fold this writes two independent outputs:

1. `component/<model>/foldN/`: the six fixed ablation variants, all with
   kappa=3 and margin=-0.7. The mean-only variant reuses `(X,V,V)` weights,
   merges the duplicate V weights, applies single SLERP, and has no residual.
2. `component_review_control/<model>/foldN/`: only mean + AW + Gate at
   **kappa=2.5, margin=0.965**, the archived configuration producing fold-1
   MC1=38.8753%. The previously missing fold-0 counterpart is now runnable.

The second setting was located in the historical fold-1 **test** grid; it is a
fixed-setting comparison here, not independent development-selected evidence.
Do not replace it with the best test-grid score from another setting. The
archive contains other tuned controls, including controls exceeding the cited
38.88%, so a claim that the residual is necessary needs the complete configuration
and selection rule. The merged-mean control's default gate can be all-pass;
report its actual steering rate with the score.

## Q3: gate behavior, margins, and model scale

```bash
python -m reproducibility.run gate --model llama --model-dir models/Llama --execute --resume
python -m reproducibility.run margin --model llama --model-dir models/Llama --execute --resume
```

The gate suite preserves both apply and observe-only modes. The margin suite first
evaluates both folds for every declared margin, including -0.7 and 1.0, then supplies
those matching MC artifacts to the label-conditioned gate-rate recorder. It does
not require unpublished precomputed margin results.

```bash
.venv-scaling/bin/python -m reproducibility.run scaling --execute --resume
```

Scaling retains its existing pinned model revisions and per-fold 20% dev holdout.
0.5B/1.5B/3B run in BF16; 14B runs in NF4 with BF16 compute. Precision is in the
path and metadata; those rows must remain labeled. `--scaling-models qwen05 qwen3`
selects a subset; `--model-root models/scaling` uses the existing download helper's
model directory layout. The suite is separate from main because it uses the
existing modern Transformers environment and completion-logprob scorer.

## Noise configuration

```bash
python -m reproducibility.run noise --model llama --model-dir models/Llama --execute --resume
```

This explicitly invokes the current **training-target** noise experiment,
including sigma=0, for both folds. Target perturbations are sampled once and fixed
across training. Inference residual noise is pinned to zero. Do not label an old
inference-noise artifact as this experiment. A sigma=0 single-fold value remains
a single-fold value; it is not directly interchangeable with the main macro mean.

## BiPO alone and calibrated BiPO

```bash
python -m reproducibility.run bipo --model llama --model-dir models/Llama --execute --resume
```

BiPO's adaptation is currently Llama-3.1-8B-Instruct layer 14, FP16, 20 epochs,
seed 42, beta=.1, LR=5e-4, batch size 2. It uses local model files, separate
question-level training/dev/test partitions, bidirectional-dev-loss checkpoint
selection, then dev-MC1 parameter selection. **BiPO alone is an explicit baseline**;
BiPO+S2S-L and BiPO+S2S-R are separate methods. The all-token intervention scope
is preserved; it differs from some main S2S evaluation scopes. Main-table and
BiPO result tables therefore need their protocol labels, not a silent merge.
Original tests, split records, development selections, and upstream vendor license
are retained in `experiments/bipo_truthfulqa/`.

## Free generation / MMsafe prerequisite

The original MMsafe prompts, original classifier configuration, and original
200-token generation records are absent. Supply the exact original data and judge
to reproduce that historical evaluation. The added adapter accepts **text** JSONL:

```json
{"id": "prompt-0001", "prompt": "The exact original evaluation prompt"}
```

It rejects image/audio/video fields so it cannot silently turn a multimodal
benchmark into a text-only evaluation.

```bash
python -m reproducibility.run generation --model llama --model-dir models/Llama \
  --prompts data/mmsafe.jsonl --execute --resume
```

For each checkpoint fold it generates baseline/L2S/S2S-L/S2S-R answers with greedy
decoding, no forced minimum length, and max_new_tokens=200. It records token IDs,
actual emitted-token counts, EOS versus length finish reason, and paired prompt IDs.
S2S defaults to the existing fixed-probe three-pass generation semantics: baseline
answer -> Q+baseline probe -> fresh steered generation with a frozen prediction.
It never uses reference answers. `python reproducibility/generation.py --help`
also exposes online prediction and explicit as-is/QA/chat prompt formatting.
Prompt formatting, predictor mode, EOS rules and data/checkpoint/source hashes are
saved; changing them requires a new output directory.

Safety scoring requires a JSON metadata file containing `judge_id`, `revision`,
`input_format`, `unsafe_score_definition`. Use the original judge's export:

```bash
python -m reproducibility.safety \
  --generations results/reproduction/generation/llama3.1-8B-Instruct/fold0/generations.jsonl \
  --scores data/original_judge_scores.jsonl --judge-config data/original_judge_config.json \
  --thresholds 0.5 --output-dir results/safety/fold0
```

Each score row must contain `id`, `method`, `unsafe_probability`. Coverage must
match exactly. Safety is `unsafe_probability < threshold`; equality is unsafe.
A positive-safe score must be converted with the original judge's declared score
semantics before using this interface. The script can optionally run a Hugging Face
sequence classifier with `--classifier`, `--unsafe-label`, `--judge-input` and an
explicit revision in the judge config; choosing a substitute classifier creates a
new experiment and does not validate the historical safety numbers. Input truncation
is off unless explicitly configured. `summary.csv` reports safety, limit-hit rate,
and mean generated length on the same prompts.

## Resuming, provenance, and tests

Default output root: `results/reproduction`. Use a new `--work-dir` when changing
model snapshots, source code, package versions, settings, or selection protocols.
Receipts compare source/package/model-config fingerprints and output hashes.
Unverified pre-existing final outputs are not silently reused. Interrupted stages
without a receipt may require a fresh work directory; native evaluators retain
their own documented resume semantics.

```bash
python -m unittest discover -s reproducibility/tests -v
python -m pytest -q reproducibility/tests experiments/component_ablation/test_interventions.py \
  experiments/component_ablation/test_vavg_grid.py experiments/bipo_truthfulqa/test_bipo.py \
  experiments/gate_correctness/test_experiment.py experiments/l2s_noise_robustness/test_noise.py
.venv-scaling/bin/python -m pytest -q experiments/model_scaling_ablation/test_ablation.py \
  experiments/model_scaling_ablation/test_retune_spherical.py
```

No full GPU rerun is claimed by the source release. CPU tests check numerical
invariants and orchestration; they do not establish final benchmark accuracy.

Known legacy-only entry points: `experiments/l2s_reliability/evaluate.py` and
several `experiments/cosine_similarity` scripts still import the removed
`method.l2s` package. They are retained as historical analysis sources and are not
used by the main/review runners. Do not substitute a differently defined L2S
implementation merely to make those old analyses import.
