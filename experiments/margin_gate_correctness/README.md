# Margin × TruthfulQA 候选类别门控触发率与 MC1

这个实验修正 `margin_ablation` 只报告整体 token 触发率的问题：对每个
`margin`，分别统计 TruthfulQA `Correct Answers` 与 `Incorrect Answers` 的门控触发率，
并关联同一 margin 的 MC1。

主触发率是：每题内同一类别候选等权平均，再对题目等权平均；`summary.csv` 同时保存
token-weighted 触发率。MC1 直接复用 `results/experiment/margin_ablation/evaluations` 中
同一模型、fold 和 margin 的已有评测，避免重复做 steering 评分。

运行 Llama：

```bash
GPU=1 bash experiments/margin_gate_correctness/run.sh
```

运行 Qwen：

```bash
GPU=2 OUTPUT_DIR=results/experiment/margin_gate_correctness/Qwen2.5-7B-Instruct \
  bash experiments/margin_gate_correctness/run.sh \
  --model_name Qwen2.5-7B-Instruct \
  --model_dir /mnt/nas-public-bing/pinxuanli11/Qwen2.5-7B-Instruct \
  --checkpoint_dir checkpoints/dyn3d_sga/truthfulqa/Qwen2.5-7B-Instruct \
  --layer 19
```

结果包含逐候选 `candidates.jsonl`、按 margin/fold 的 `summary.csv` 与 `summary.json`、
中文 `REPORT.md` 和图表。当前 sweep 为 `-0.8,-0.6,...,1.0`；上一轮 `margin=-0.7`
不在该 sweep 中。
