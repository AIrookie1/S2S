# Dyn2D launchers

Dyn2D mirrors the Dyn3D launcher layout for both `LLAMA` and `QWEN`:

- `run_2d_extract.sh`
- `run_2d_train.sh`
- `run_2d_eval.sh`
- `run_2d_dyn.sh`
- `run_llm_judge.sh`
- `run_llm_judge_2d.sh`
- `generic/run_dataset.sh` plus per-dataset wrappers

The requested ablation is separate and fixed to Llama/fold0:

```bash
bash experiments/dyn2d_ablation/run.sh
```
