# Generic Dyn3D tasks

Five datasets use separate configs under `hyperparams/generic/<dataset>/config.sh`: BoolQ, COPA, StoryCloze, MMLU Global, and Winogrande. Shared method defaults are energy=0.99, router Top-K=40, and load-balancing loss weight=0.02. The main LLAMA evaluation/judge default is kappa=8 and margin=-0.3, while generic configs intentionally preserve their dataset-specific tuned kappa/margin and other existing variants.

Run one dataset with `bash generic/script/<dataset>_generic.sh`, or all five with `bash generic/script/quick_set_generic.sh`. Feature extraction is opt-in (`RUN_EXTRACT=1`); training and evaluation are on by default. `DRY_RUN=1` prints the exact commands without executing model/data workloads.
