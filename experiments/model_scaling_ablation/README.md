# 不同规模模型消融

比较同一 Qwen2.5-Instruct 系列的不同规模，并保留已有的新架构和大模型对照：

| 设置 | 模型 | 默认干预层（从 0 开始） | 官方权重 revision |
|---|---|---:|---|
| 小模型 | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 16 / 24 层 | `7ae557604adf67be50417f59c2c2f167def9a775` |
| 小模型 | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 20 / 28 层 | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 小模型 | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 22 / 36 层 | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| 稍大模型 | [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 32 / 48 层 | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |
| 新架构 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 21 / 32 层 | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| 更大规模 | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 43 / 64 层 | `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` |

针对 24GB RTX 3090：推荐 `qwen15` + `qwen3` 的 BF16 规模对照；`qwen05` 可作为极小规模锚点，资源占用很低。`qwen14` 用统一 NF4 4-bit 配置在单卡运行，避免 BF16 全量权重超过显存；量化结果需与 BF16 行分开解读。原有 9B 可用 BF16，多卡/空闲显存允许时运行；32B 需要 4-bit 或大量 CPU/disk 卸载，耗时明显，不建议作为这次常规扩展首选。实验室的 GPU 使用率会变化，启动前请以 `nvidia-smi` 实测空闲显存为准。

默认层按约 2/3 网络深度预先指定，**不是已经验证的最佳层**。可用 `--layer` 改变，但不得用测试集挑选层再报告无偏结果。Qwen3.5 使用官方 `Qwen3_5ForConditionalGeneration`，自动定位 `model.language_model.layers`；旧 Qwen 使用 `model.layers`。对两者均直接提取 block output，避免末层 normalization 改变特征定义。

五种方法：

| 输出名称 | 方法 | 实现 |
|---|---|---|
| `baseline` | 原模型 | 无干预 |
| `additive` | 常规静态加法 steering | `h + alpha * (mean(h_true) - mean(h_false))`，保留原向量尺度 |
| `l2s` | L2S | 当前 `method/l2s_linear`，raw hidden -> raw pair vector，再直接加法；按仓库实现使用 Cartesian positive/negative pairs |
| `spherical` | Spherical Steering | 当前 `method/spherical_steering/steering.py`，contrastive antipodal prototypes + vMF gate |
| `s2s` | 我们的 S2S-R | 当前 `get_3d_dynamic_intervention()`，normalized-input MLP + 动态 residual + adaptive weights + margin gate；训练使用当前 mean-positive 定义 |

每个模型重新提取特征、训练 L2S/S2S 的预测器，禁止复用 7B/8B 权重。常规加法与 Spherical Steering 使用同一批 geometry training 问题计算 class-centroid difference。S2S 的 `v_avg` 按现有方法从同一批 geometry 问题估计，保留方法本身的 pair-weighting 语义。

## 数据协议

- TruthfulQA 全部 817 道题，MC1/MC2/MC3；使用仓库已有 CSV/HF 对齐与指标函数。
- 外层按问题编号分两折，每一折只评估另一半。每道题作为 test 恰好出现一次。
- 每个 outer-train 留出 20% 作为 dev，其余按现有训练器规则分为 geometry / predictor 两部分（各 50%）。geometry、predictor、dev、test 四部分两两不相交。
- 默认在 dev 搜索 additive/L2S alpha、Spherical alpha/beta 和 S2S weight_kappa/margin，按 MC1、再 MC2、再网格顺序选择。所有 dev 候选与逐题分数均保留。
- `--no-tune` 使用声明的固定参数；结果中明确标注 `fixed_parameters`，不称为调参最优结果。
- 最终指标按问题数加权汇总两折；408/409 题的两折不会被不加权平均。
- prompt 遵循原仓库：特征为 `Q: question A: choice`；评测加入原 Qwen instruction。没有 chat template、没有生成 thinking trace，指标是答案的 teacher-forced log probability。
- `--smoke` 仅使用前 12 题、训练 2 epochs、单参数设置，输出目录带 `_smoke`；**不作为论文结果**。

## 环境与下载

建议新建环境，CUDA PyTorch 使用主机对应版本。现有依赖可复用，但不要升级原实验环境。

```bash
python -m venv --system-site-packages /path/to/new/venv
/path/to/new/venv/bin/python -m pip install -r experiments/model_scaling_ablation/requirements.txt
/path/to/new/venv/bin/python experiments/model_scaling_ablation/download.py \
  --model-root /path/to/models
```

下载脚本固定 revision、支持 Hugging Face 断点续传、检查所有 index 中的 shard 并写 `s2s_download.json`。本机下载位置是 `/mnt/nas-public-bing/pinxuanli11/s2s_ablation_models`；独立 Python 是 `/root/.cache/s2s-model-ablation/venv/bin/python`。

## 运行

先检查计划（不加载模型、不创建结果）：

```bash
python experiments/model_scaling_ablation/run.py --model qwen15 --dry-run
python experiments/model_scaling_ablation/run.py --model qwen3 --dry-run
python experiments/model_scaling_ablation/run.py --model qwen14 --dry-run
python experiments/model_scaling_ablation/run.py --model qwen35 --dry-run
python experiments/model_scaling_ablation/run.py --model qwen32 --dry-run
```

实际运行时指定下载目录：

```bash
PYTHON_BIN=/path/to/new/venv/bin/python \
MODEL_ROOT=/path/to/models MODEL=qwen35 \
  bash experiments/model_scaling_ablation/run.sh --smoke

PYTHON_BIN=/path/to/new/venv/bin/python \
MODEL_ROOT=/path/to/models MODEL=qwen15 \
  bash experiments/model_scaling_ablation/run.sh --smoke

PYTHON_BIN=/path/to/new/venv/bin/python \
MODEL_ROOT=/path/to/models MODEL=qwen3 \
  bash experiments/model_scaling_ablation/run.sh --smoke

# 14B on a 24GB card: use NF4 for this complete result set.
PYTHON_BIN=/path/to/new/venv/bin/python \
MODEL_ROOT=/path/to/models MODEL=qwen14 \
  bash experiments/model_scaling_ablation/run.sh --load-in-4bit --smoke

PYTHON_BIN=/path/to/new/venv/bin/python \
MODEL_ROOT=/path/to/models MODEL=qwen35 \
  bash experiments/model_scaling_ablation/run.sh

PYTHON_BIN=/path/to/new/venv/bin/python \
MODEL_ROOT=/path/to/models MODEL=qwen32 \
  bash experiments/model_scaling_ablation/run.sh
```

两模型顺序执行 download -> smoke -> full，并持续写 job 状态：

```bash
/path/to/new/venv/bin/python experiments/model_scaling_ablation/run_all.py \
  --model-root /path/to/models \
  --job-dir results/experiment/model_scaling_ablation/job_bfloat16 \
  --dtype bfloat16 --cpu-memory 40GiB
```

本次实际任务按用户选择固定为 **BF16**，后台队列状态保存在 `results/experiment/model_scaling_ablation/job_bfloat16/status.json`；旧 `job_nf4` 已标记为被替换。

默认 BF16，可用多卡和 CPU/disk 卸载；每张卡按当前空闲量预留 3 GiB 给临时张量，不停止其他任务。CPU 卸载的 32B 可能很慢。显存不足可在**两个模型的所有方法**中统一增加 `--load-in-4bit`（NF4 + double quant，BF16 compute）；该设置单独写入 `nf4` 目录，不与 BF16 混报。`CUDA_VISIBLE_DEVICES` 可限制用卡。可选 `--cpu-memory 40GiB`、`--output-dir`、`--epochs`、`--seed`、`--layer`。

每个运行目录固定配置与源码 fingerprint，配置变化要求新目录，避免串用旧模型或旧参数的缓存。中断后相同命令会复用已完成特征、训练 checkpoint、dev 单元和 test 报告；未完成的阶段会重新执行。

## 输出与检查

`results/experiment/model_scaling_ablation/<model>/<bfloat16|nf4>[_smoke]/`：

- `run_config.json`、`runtime.json`、`status.json`：模型 revision、精度、层、源码身份、实际加载类/设备、运行状态。
- `features.npz`：当前模型当前层的原始特征。
- `fold{0,1}/split.json`：每个 geometry / predictor / dev / test 问题 ID。
- `fold{0,1}/l2s.npz`、`s2s.npz`、`geometry.npz` 与训练日志。
- `fold{0,1}/dev/*.json`、`*_selection.json`：验证集选择证据。
- `fold{0,1}/*_test.json`：最终参数、逐题答案 log probability 与 MC 分数。
- `summary.json`、`summary.csv`、`summary.md`：五方法两折汇总；百分数仅用于 Markdown 展示，JSON/CSV 保持 [0,1]。

测试（使用真实架构的随机小网络，不下载大权重，不代替真实评测）：

```bash
/path/to/new/venv/bin/python -m unittest experiments/model_scaling_ablation/test_ablation.py -v
```

另有实际 GPU 检查确认 BF16 加载及 CPU 卸载前后评分一致。

测试覆盖 Qwen3.5 的 linear/full attention 和 Qwen2 的真实 forward、层挂钩作用、答案 log probability 独立对照、五种方法的 segment 边界与 tuple 输出、训练/dev/test 无泄漏、两折加权汇总与重叠拒绝。

## Spherical Steering 扩展调参

`retune_spherical.py` 针对现有六个完整模型结果补扫 κ、α、β，不重新训练模型或改变干预层。每折共 63 个候选，保留原 36 点网格，并扩展 κ 到 100/200/500、α 到 0.9、β 到 0/-0.1，包含 α=0 的无干预候选。完整缓存会复用；Qwen3.5-9B 原目录只有 6 个 dev 候选，需要补齐其余点。

**每折独立在自己的 dev 集按 MC1、MC2、网格顺序选参**，然后只评估该配置的 test；禁止将两折 dev 合并后选参，因为另一折 dev 属于本折 test。旧 `sweep_spherical_kappa.py` 和 `shared_hparam_eval.py` 的 pooled-dev 选择不能作为无泄漏的两折结果。本次新脚本不采用这些结果。

```bash
# 只查看计划，不加载模型、不写入结果。
/root/.cache/s2s-model-ablation/venv/bin/python experiments/model_scaling_ablation/retune_spherical.py --plan

# 三张空闲 GPU 的独立进程队列：0=0.5B/1.5B，1=3B/14B，2=Nemotron8B/9B。
/root/.cache/s2s-model-ablation/venv/bin/python -u experiments/model_scaling_ablation/retune_spherical.py

# 断点续跑单模型，或仅刷新总表。
CUDA_VISIBLE_DEVICES=0 /root/.cache/s2s-model-ablation/venv/bin/python experiments/model_scaling_ablation/retune_spherical.py --model Qwen2.5-0.5B-Instruct
/root/.cache/s2s-model-ablation/venv/bin/python experiments/model_scaling_ablation/retune_spherical.py --summarize
```

新结果位于 `results/experiment/model_scaling_ablation/spherical_retuned_v1/`，原结果不覆盖。每个模型保留输入与源码哈希、每折逐题 dev 缓存、选择依据、逐题 test、运行状态与日志；总表 `comparison.md/json/csv` 随任务进度更新。14B 保留 NF4，其余保持 BF16。报告全部模型的原始及调参结果，不按 test 再选择“更好”的版本；扩大 dev 搜索不保证 test 提升。
