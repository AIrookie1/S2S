# L2S-LINEAR

L2S-LINEAR learns the raw vector predictor

```text
MLP(z_neg_raw) -> v_hat ~= v_T = z_pos_raw - z_neg_raw
```

Training uses ordinary MSE with raw hidden states and raw target vectors.
Inference is the standard additive steering rule, with no spherical
normalization and no vMF gate:

```text
h_new = h + alpha * v_hat
```

Only `alpha` controls the inference-time steering strength.

## Generic multiple-choice tasks

L2S-LINEAR also supports the repository's five generic MC datasets:
`boolq`, `copa`, `storycloze`, `mmlu_global`, and `winogrande`.

The generic path reuses the same answer-level hidden-state features, prompts,
option scoring, and MC1 evaluation logic as S2S_R. The existing L2S loader
reconstructs raw positive/negative pairs from `activations`, `labels`, and
`q_indices`, and training keeps the same question-level fold/within-fold split
convention as S2S_R.

Direct launchers are under:

```text
scripts/l2s_linear/LLAMA/generic/
scripts/l2s_linear/QWEN/generic/
```

For example:

```bash
bash scripts/l2s_linear/LLAMA/generic/boolq.sh
bash scripts/l2s_linear/QWEN/generic/mmlu_global.sh
```

`RUN_EXTRACT=auto` is the default: existing canonical S2S_R generic features
are reused when available, otherwise they are extracted automatically.
Override the initial steering strength at launch time, for example
`ALPHA=0.8 bash ...`.
