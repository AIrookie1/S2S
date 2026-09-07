# Runnable pipeline scripts

Each script resolves the repository root before running, uses
`/root/miniconda3/envs/py310/bin/python` when available, and accepts environment
variables in the same style as the reference `NegSteer` scripts.

Run individual stages:

```bash
MODEL_DIR=/path/to/model GPU=0 ./script/spherical/extract_features.sh
./script/spherical/compute_prototypes.sh
FOLD_IDX=0 KAPPA=20 ALPHA=0.7 BETA=-0.15 ./script/spherical/evaluate_mc.sh
FOLD_IDX=0 ./script/spherical/evaluate_llm_judge.sh
```

The `run_mc_all.sh` and `run_llm_judge_all.sh` wrappers run both folds. The
top-level `quickstart_llama.sh` and `quickstart_qwen.sh` call all stages in
order; override `MODEL_DIR`, `GPU`, `PYTHON_BIN`, `LAYER`, `KAPPA`, `ALPHA`, or
`BETA` as needed.
