# TruthfulQA 正确/错误候选答案的门控触发频率

研究问题：在现有 S2S-R 候选答案评分流程中，正确答案和错误答案的门控触发率是否不同？

## 固定设置

- Llama-3.1-8B-Instruct，第 14 层，κ=3，margin=-0.7，seed=42，FP16。
- 默认 GPU 1，使用项目 py310 环境与本地模型、缓存的 TruthfulQA 数据。
- 复用已有 direct-MLP mean-positive 检查点，fold0/fold1 各自只评估 test_q_indices。
- 检查所有训练子集与测试集不重叠，两折测试题不重叠且完整覆盖 817 道题。
- 沿用生产 zero-shot prompt 和默认 instruction。使用 CSV 的所有 Correct Answers / Incorrect Answers，以原有 split_multi_answer 规范化答案。
- 不根据这次结果调参，不重新训练。

## 两种模式

`apply` 调用原始 get_3d_dynamic_intervention 并返回干预输出。
`observe` 也计算、记录同一个门控，但返回未修改的层输出。

两种模式各做一次真实前向计算。每次均用原始 hook 的 token 计数和权重和核对记录器，并逐候选检查两模式的完整门控 mask 相同。
当前单层干预发生在门控计算之后，所以同一个候选答案的两模式门控预计相同；两模式的答案对数概率、MC1/MC2/MC3 可以不同。

## 触发率与位置边界

门控条件严格沿用 `max(beta, gamma) > alpha + margin`。
零样本评分的干预区间包含“最后一个 prompt token + N 个答案 token”。为避免边界混淆，分别输出：

1. **answer_tokens（主指标）**：N 个答案 token 的隐藏状态位置，排除最后一个 prompt token。
2. scored_answer_predictors：预测 N 个被评分答案 token 的位置，包含最后一个 prompt token，排除最后一个答案隐藏位置。
3. legacy_intervention_segment：原始 hook 的全部 N+1 个位置，用于对齐原评测日志。

每个候选先计算触发 token 数 / 对应区间 token 数。主统计先在每题内对每类候选等权取均值，再对题目等权取均值；主差值为错误减正确。
报告 20,000 次按题配对、按折分层 bootstrap 的 95% 百分位置信区间，固定已训练检查点。CI 反映题目抽样不确定性，不包含训练随机性。
同时报告每折结果、总 token 加权率、候选等权平均率、至少一次触发的候选比例及候选平均长度。

## 运行

```bash
GPU=1 bash experiments/gate_correctness/run.sh
```

默认结果目录：`results/experiment/gate_correctness/llama3.1-8B-Instruct`。
已有结果时程序拒绝覆盖；重新运行请设 `OUTPUT_DIR` 为新目录。

```bash
GPU=1 OUTPUT_DIR=results/experiment/gate_correctness/smoke \
  bash experiments/gate_correctness/run.sh --max_questions_per_fold 2 --bootstrap_repeats 1000
```

可通过 `PYTHON_BIN`、`OUTPUT_DIR`、`GPU` 环境变量和 Python 参数覆盖路径或配置。
`run.sh` 使用离线 Hugging Face 缓存，避免评测时额外下载。

输出：

- `config.json`：实际参数、检查点/源码/数据 SHA256、GPU 与依赖版本。
- `status.json`：运行状态与已完成题目数。
- `candidates.jsonl`：每候选、每模式的得分、三种窗口的计数、逐 token 门控 mask、答案 token ID、权重均值、门控差值。
- `questions.csv`：逐题配对触发率与 MC 指标。
- `summary.json` / `summary.csv`：两折、汇总、两模式的完整统计与一致性检查。
- `REPORT.md`：可阅读的中文结果报告。
- `gate_correctness.png` / `.pdf` / `.svg`：主触发率及按题 bootstrap 置信区间。

仅重算统计与绘图：

```bash
/root/miniconda3/envs/py310/bin/python experiments/gate_correctness/analyze.py \
  results/experiment/gate_correctness/llama3.1-8B-Instruct
```

## 结论范围

严格保留生产实现：使用整个候选最后一个隐藏状态预测 residual，并广播到整个干预段。这是 **teacher forcing 候选评分的答案条件化分析**，不等价于逐步自由生成的门控行为。
正确/错误标签由 TruthfulQA 标注给定，不由模型预测是否答对决定。
问题等权可减少不同候选数量与长度的加权影响，无法因果排除长度、内容措辞等混杂；即使存在显著差异，也不能单独证明门控能检测错误。
