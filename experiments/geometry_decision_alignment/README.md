# Geometry to decision alignment

This mechanism study tests three links: whether a learned correction aligns
with task-relevant hidden-state directions, whether the released S2S-R spherical
fusion preserves useful directions, and whether adaptive geometric weights
favor directions that improve the answer score margin.

Run a reproducible pilot on 128 sampled questions per benchmark (COPA uses all
100 validation questions):

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /root/miniconda3/envs/py310/bin/python experiments/geometry_decision_alignment/run.py
```

Use `--max-questions 0` for full splits, `--datasets ...` for a subset,
`--device cpu` when a GPU is unavailable, or `--output PATH` for a separate
run. Questions are selected with seed 20261111. Per-question JSON records are
written atomically and can be resumed when the manifest signature matches.

## Full-set supervised run

```bash
/root/miniconda3/envs/py310/bin/python experiments/geometry_decision_alignment/run_full.py --gpus 1 2
```

Splits each task's questions into disjoint GPU shards. The default output is
`results/experiment/geometry_decision_alignment_full/`. COPA reuses its completed
100-question run; the other tasks evaluate their entire sets: TruthfulQA 817,
StoryCloze 1511, MMLU balanced test 2000, Winogrande 1267, BoolQ 3270.
The supervisor writes `run_status.json`, per-GPU logs, and per-question records.
Only when both workers finish successfully does it generate `final_report.md`
and `final_results.csv/json`. The report checks all expected question IDs and
record configuration signatures, includes corrected/broken counts, 10000-sample
paired question-bootstrap confidence intervals and exact McNemar p values.
These are exploratory per-task tests; p values have no multiplicity adjustment.

To resume after an interrupted or failed run, rerun the same command/output.
Completed questions are skipped. `worker_failed` or `report_failed` in the
status file means completion must not be claimed; inspect the associated log.

The run uses the repository's official generic prompt/scoring definitions,
generic task-specific S2S-R and Spherical Steering hyperparameters, released
checkpoints, and train-derived spherical prototypes. TruthfulQA uses its
existing held-out MLP and spherical prototype fold per question. All six
predeclared static three-way weight settings keep the corresponding adaptive
gate fixed. Per-question label-selected settings are explicitly oracle
diagnostics and are not deployable benchmark results.

For each question, candidate answer scores are replayed through the exact model
suffix after the intervention layer. The pilot checks that baseline,
Spherical-Steering, and adaptive-S2S scores match full-model hook evaluation
within 0.02 score units. It records correct-versus-strongest-incorrect margin,
the identity of that competitor, predicted and nearest-truthful oracle
residuals, adaptive weights and gate, actual spherical rotation, and per-token
gradient contributions. Gradient margin changes are checked against matched
small positive/negative finite differences.

Equal-norm radial/tangential probes decompose the MLP vector relative to each
hidden state. Mean, residual, and oracle tangential probes use the same 1% of
hidden norm, so their local score effects compare direction while controlling
edit size. Full-vector probes provide the radial-inclusive comparison. The
actual S2S and Spherical Steering edits are also rescored, providing nonlinear
margin changes rather than relying only on local gradients.

The fixed weight grid selects one setting per question using answer labels.
That estimate is reported only as a diagnostic upper bound. It does not tune
the primary S2S result. Report the sampled pilot separately from full benchmark
accuracy and use paired question bootstrap / McNemar analysis before making
significance claims.

End-to-end GPU smoke tests now run on the local Llama shards. TruthfulQA MLP
and spherical-prototype banks are routed independently by their recorded
held-out question IDs; their numeric fold labels need not match.

Generate execution/numerical verification summaries with:

```bash
/root/miniconda3/envs/py310/bin/python experiments/geometry_decision_alignment/summarize.py \
  results/experiment/geometry_decision_alignment_multitask_smoke
```

FP32 tiny-model tests verify exact suffix/full-model scores and analytic
gradients against central finite differences. In full Llama FP16 inference,
small finite differences can be dominated by rounding: they are reported as
diagnostics, not a passed gradient validation gate. Higher-precision numerical
checks are still required before claiming gradient-based mechanism findings.
