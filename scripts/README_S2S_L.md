# S2S_L launchers

S2S_L mirrors the S2S_R launcher layout for both `LLAMA` and `QWEN`:

- `run_S2S_L_extract.sh`
- `run_S2S_L_train.sh`
- `run_S2S_L_eval.sh`
- `run_S2S_L.sh`
- `run_llm_judge.sh`
- `run_llm_judge_S2S_L.sh`
- `generic/run_dataset.sh` plus per-dataset wrappers

The requested ablation is separate and fixed to Llama/fold0:

```bash
bash experiments/fig3a_s2s_l_hyperparameters/fig3a_run.sh
```
