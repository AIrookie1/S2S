# BIPO on TruthfulQA with S2S calibration

Run `bash experiments/bipo_truthfulqa/run.sh` from the repository root.
The default uses GPU 1, `/root/miniconda3/envs/py310/bin/python`, local
Llama-3.1-8B-Instruct, and zero-indexed block 14. Override `CUDA_VISIBLE_DEVICES`
and `PYTHON_BIN` as needed. Outputs live in `results/experiment/bipo_truthfulqa`.
`--smoke` writes to a separate directory and is never a full-data result.

## Upstream and adaptation

Official source: https://github.com/CaoYuanpu/BiPO at commit
`cef1d00ab108d0e265578589e6b344b200ea3db8`. The MIT license and the two
relevant source files are preserved in `vendor/` for inspection; the vendored
trainer is a reference, not a dependency on its old Transformers/TRL stack.

The base model is frozen. A single zero-initialized vector `v` is optimized;
its sign `s` is sampled uniformly from {-1,+1} per batch. The intervention is
`h <- h + s*v` at **every token** of block 14, matching the upstream wrapper.
The objective is exactly its default sigmoid loss:

```
-log sigmoid(beta*s*((log pi_s(y+)-log pi_s(y-))
                     -(log pi_0(y+)-log pi_0(y-))))
```

Only answer-token log probabilities contribute to the sum. The reference is
the same frozen model with zero steering. Cached block outputs and reference
log probabilities save recomputation without training the base model. An
on-device numerical audit checks full-model versus prefix+tail execution.

Explicit adaptations: Llama-3.1 replaces the original Llama-2/Mistral;
repository zero-shot truthful instructions replace FastChat formatting;
FP16 runs on a 24GB GPU; batches contain two pairs; 20 epochs, seed 42,
learning rate 5e-4, beta .1, cosine schedule, 100 warmup steps, gradient clip 1.
As in the *actual* upstream `DPOConfig` invocation, weight decay is zero
(the upstream CLI's .05 value is not forwarded to its trainer).
One uniformly sampled positive/negative answer pair per training question per
epoch provides question-balanced stochastic Cartesian pairing. This is an
adaptation, not an exact reproduction of the upstream persona data protocol.
Use `--epochs 100` with a fresh `--output` for a longer training budget.

## Held-out protocol and S2S definitions

The 817 HF question IDs are split into two sorted halves. For each fold,
one half supplies outer training, the other supplies held-out testing.
20% of outer training is reserved as development data (seed 42+fold).
Only the remaining questions optimize BIPO **and** estimate the calibration
anchor `v_avg = mean(h_pos_raw-h_neg_raw)` over Cartesian answer pairs.
The model, layer and feature artifact hash are recorded in `config.json`.
No development/test activations enter the mean anchor.

The BIPO checkpoint minimizes **bidirectional development loss**. Then each
method's parameters maximize development MC1, breaking ties by MC2 and grid
order. Test results never select checkpoints or parameters.

- **BIPO**: all-token additive vector; strengths 1, 2, 3.
- **BIPO+S2S-L**: existing `get_2d_dynamic_intervention`, static interpolation
  variant (`S2S_L_Static`), `V=normalize(v_avg)`, `R=normalize(v-v_avg)`,
  `D=SLERP(V,R,0.2)`, `h'=h+strength*||v||*D`; strengths 1, 2, 3.
- **BIPO+S2S-R**: existing `get_3d_dynamic_intervention`; symmetric composition
  of normalized h, V, R, adaptive weighting kappa=3, restoring `||h||`;
  margin in {-0.7,-0.3,0}.

Both BIPO and S2S intervene at **all tokens**, matching the original BIPO
training/inference scope and isolating the geometric calibration. The S2S
geometry itself is imported unchanged; its hooks receive `start_idx=0`.
Neither S2S wrapper retrains BIPO. A constant predictor adapter returns `v`
to existing hooks; there is no MLP substituted for BIPO.

MC1/MC2/MC3 use repository `truthfulqa.models.MC_calcs` and its CSV/HF question
mapping, raw summed answer log probabilities, and no answer-length normalization.
The main report pools question-level results across the two disjoint test folds
(817 questions), and preserves fold-level summaries and per-answer scores.

## Reliability

Uses `experiments/l2s_reliability/metrics.py` directly, with the same raw
Cartesian pair targets `v_T=h_pos_raw-h_neg_raw`. The existing reliability
entry point imports a removed `method.l2s` module, so it cannot be directly run
for BIPO; this independent adapter reuses its metric definitions and grouping.

For BIPO, the prediction is its shared learned vector. For the reference Mean
source, it is the training-only average vector. Report:

- cosine similarity to `v_T`;
- angle `acos(cosine)` in degrees;
- relative norm error `abs(||v||-||v_T||)/||v_T||`;
- actual predicted and target norms.

Metrics are averaged over pairs **within each question first**, then over
questions. Every fold writes `diagnostics_question.csv` and
`diagnostics_summary.json` for train/dev/test. Train/test differences are an
empirical outcome, not a required positive result: a constant vector can have
similar aggregate error in both splits and still disagree substantially with
individual target vectors. A BIPO preference optimum need not equal an
activation difference; these diagnostics measure that geometric discrepancy,
not a supervised vector-regression training loss.

## Audits and artifacts

`python -m pytest -q experiments/bipo_truthfulqa/test_bipo.py` checks bidirectional
loss orientation, all-token BIPO intervention, S2S-R norm preservation, and
completion-only masked log probabilities. `--smoke` exercises actual model
cache equivalence, vector training, selection, and both MC folds end to end.
Each fold saves the selected vector, last epoch, training history, split,
mean anchor, dev parameter grid, per-question test scores, and diagnostics.
Partial MC evaluation resumes from completed questions. Completed BIPO training
is reused; a configuration mismatch rejects artifact reuse.

After the full two-fold run, execute:

```bash
/root/miniconda3/envs/py310/bin/python experiments/bipo_truthfulqa/analyze.py
```

This creates pooled reliability tables, explicit per-fold test-minus-train gaps,
and `bipo_reliability.png/.pdf/.svg`. `REPORT.md` and `performance.json` are
written by the main run itself. Geometric errors are descriptive; they do not
by themselves establish a causal explanation for an MC performance change.

## BiPO as the S2S mean anchor

After the BiPO experiment completes, run:

```bash
bash experiments/bipo_truthfulqa/run_mlp_s2s.sh
```

This is a separate comparison under `results/experiment/bipo_truthfulqa/mlp_s2s_retuned`.
For each fold, it trains the repository Dyn3D direct-vector MLP using only
that fold's existing BiPO optimization questions and Cartesian raw targets
`h_pos_raw-h_neg_raw`. It then uses the fold's learned BiPO vector as `v_avg`
and the MLP prediction as `v_hat` in the existing S2S-L/S2S-R geometry.
It reselects each fold's settings using only that fold's development questions.
S2S-L sweeps `strength ∈ {0.5, 1, 2, 3}` and `slerp_t ∈ {0, 0.2, 0.5, 0.8, 1}`;
S2S-R sweeps `weight_kappa ∈ {1, 3, 5}` and `margin ∈ {-0.7, -0.3, 0}`.
Each selected setting is evaluated on that fold's held-out test questions.
The earlier constant-predictor and fixed-setting MLP-S2S results remain
separate. The 2000-epoch MLP checkpoints are saved beside each fold's BiPO
artifacts as `mlp_predictor.npz`.

Optional complete-versus-cached numerical audit after evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/py310/bin/python experiments/bipo_truthfulqa/audit_cache.py
```

It checks both freshly captured prefix states at identical batch shape and the
stored prefix cache (which can have different FP16 padding/GEMM shapes).
The resulting numerical differences are recorded in `steered_cache_audit.json`.

MLP inference trims each answer to its actual nonpadding length before applying the S2S hook. This prevents a short answer from supplying a padding activation to the predictor. Historical fixed-setting MLP-S2S scores are superseded because the old batched evaluator could use padding for this probe. The original constant-predictor BIPO experiment is unaffected.

## Same inference hyperparameters in both folds

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/miniconda3/envs/py310/bin/python experiments/bipo_truthfulqa/run_shared_fold0.py
```

This user-requested comparison uses the corrected Fold 0 validation selection
for both folds: BIPO strength=1; MLP-S2S-L strength=1, t=0.2;
MLP-S2S-R kappa=1, margin=-0.3. It reuses matching verified test scores and
re-evaluates Fold 1's two MLP-S2S conditions. Each fold retains its independently
trained BIPO and MLP checkpoints. Outputs go to `mlp_s2s_shared_fold0/`.

Because Fold 0 development questions overlap Fold 1 test questions, the pooled
scores describe a common-setting comparison and are not a new fully independent
nested evaluation of parameter selection. This limitation is recorded in the
report; it does not mean the test labels were used to train either checkpoint.

## MLP controls without BiPO

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/miniconda3/envs/py310/bin/python experiments/bipo_truthfulqa/run_mlp_controls.py
```

Outputs: `mlp_controls_shared/`. Both folds use identical inference settings:

- `mlp_mean_s2s_l`: training-only `mean.pt` as anchor, the same trained MLP,
  strength=1 and t=0.2. The anchor is the raw Cartesian answer-difference mean.
- `mlp_additive`: standard additive `h_new=h+alpha*v_hat`, alpha=1.

Both controls use `normalize(h)` at the actual last nonpadding answer token,
one prediction per answer, broadcast across all tokens, preserving the raw
predicted magnitude. Neither loads a BiPO vector. Existing fold-specific MLP
weights are reused without refitting or test-set tuning. Standard additive
steering here is not convex interpolation between hidden state and vector.
