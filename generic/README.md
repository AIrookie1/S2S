# Generic multiple-choice evaluation

This directory keeps the historical generic dataset entry points. The
maintained implementations live under `method/*/generic/`.

Supported datasets are:

- `boolq`
- `copa`
- `storycloze`
- `mmlu_global`
- `winogrande`

Dyn3D uses a direct MLP predictor with the existing adaptive spherical
steering rule. L2S-LINEAR now uses the same generic hidden-state features and
MC1 scoring pipeline, but applies its standard additive rule:

```text
h_new = h + alpha * L2S(h)
```

The compatibility launcher `generic/script/run_generic_dataset.sh` supports
LLAMA and QWEN. Examples:

```bash
METHOD=S2S_R MODEL_FAMILY=LLAMA bash generic/script/run_generic_dataset.sh boolq
METHOD=L2S_LINEAR MODEL_FAMILY=LLAMA bash generic/script/run_generic_dataset.sh boolq
METHOD=L2S_LINEAR MODEL_FAMILY=QWEN bash generic/script/run_generic_dataset.sh mmlu_global
```

For L2S-LINEAR, results are written by default to
`results/generic/<dataset>/l2s_linear/`.
