# Scripts

Scripts are grouped by method and model family. The main experiment launchers
include:

- `script/dyn2d/LLAMA|QWEN/`
- `script/dyn3d/LLAMA|QWEN/`
- `script/spherical_steering/LLAMA|QWEN/`
- `script/l2s_linear/LLAMA|QWEN/` (standard additive L2S)
- `script/l2s_slerp/LLAMA|QWEN/` (vMF-gated spherical L2S)

L2S-LINEAR now has direct generic launchers for BoolQ, COPA, StoryCloze,
MMLU-Global, and Winogrande:

```bash
bash script/l2s_linear/LLAMA/generic/boolq.sh
bash script/l2s_linear/LLAMA/generic/run_all.sh
bash script/l2s_linear/QWEN/generic/copa.sh
```

The generic L2S launcher defaults to `RUN_EXTRACT=auto`, `RUN_TRAIN=1`, and
`RUN_EVAL=1`. Existing Dyn3D generic feature files are reused by default so
both methods consume identical hidden-state inputs. Set `RUN_EXTRACT=0` to
require a pre-existing feature file, or `RUN_EXTRACT=1` to force extraction.

Generic L2S outputs are placed under
`results/generic/<dataset>/l2s_linear/` by default.
