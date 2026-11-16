# Q1 rebuttal：实验配置、逐折结果与数值来源说明

核对日期：2026-11-15。本说明依据当前仓库的原始 JSON、checkpoint 元数据、运行配置和已有审计文件整理；此次没有执行 GPU 实验。分数在最后一步乘100、保留两位小数，所有平均使用未舍入分数。

用途：论文实验设置补充、rebuttal 表注和内部核对的统一依据。当前仓库未包含审稿人所见两份完整回复表或论文源文件，因此用配置 ID 标识实验，不猜测未知表号、历史配置或错误原因。

## 1. 审稿意见对应的核实结论

| 问题 | 已核实事实 | 应采取的修改 |
|---|---|---|
| 主表均值与折间20点差距 | Llama S2S-R两折MC1=67.4019608% / 46.9437653%，算术均值57.1728630%，显示57.17% | 正文说明问题级划分及训练子集；附上主要方法逐折结果 |
| Qwen-3B：32.68 / 36.68；14B：44.68 / 48.29 | 现有scaling记录支持32.68% / 44.68%；另两个数字尚无匹配审计记录 | 给每表绑定配置；对未确认数字回查或明确撤回，不编造差异原因 |
| Llama S2S-L：49.02 / 43.95 | 前者是历史噪声clean条件fold0；后者是两折算术均值 | 噪声表标注fold0、408题、噪声位置；主表标注macro mean |

**尚未完全解决的证据缺口：** 原主表No Steering / Additive的匹配逐折文件缺失；36.68% / 48.29%的来源未确认。本文提供所有已核实配置，但不能仅靠补充文字宣称这些缺口已经闭环。

## 2. 不同表格的配置 ID 与口径

| 配置 ID | 实验 | 模型/层（0-based） | 训练与选参 | 汇总 |
|---|---|---|---|---|
| MAIN-ARCHIVE | 归档TruthfulQA主结果 | Llama-3.1-8B-Instruct /14；Qwen2.5-7B-Instruct /19 | 两折；outer-train内再分predictor/geometry；历史实际参数见第4节 | 两折算术均值macro |
| MAIN-RERUN | 当前统一main入口 | 同上，CUDA FP16 | 显式固定参数；部分Qwen参数与归档不同 | 同时输出macro与pooled |
| SCALE-3B | 产生32.68%的规模实验 | Qwen2.5-3B-Instruct /22，BF16 | 每折20% dev，独立选参 | 817题pooled |
| SCALE-14B | 产生44.68%的规模实验 | Qwen2.5-14B-Instruct /32，NF4、BF16 compute | 同上 | 817题pooled |
| NOISE-LEGACY-F0 | 产生49.02%的历史clean结果 | Llama /14，S2S-L-Static | fold0，inference residual noise σ=0 | 单折408题 |
| NOISE-TARGET-RERUN | 当前训练目标噪声 | Llama/Qwen主模型及主层 | 每个水平重训；统一入口显式两折 | 先按折输出，跨折均值须声明 |
| COMPONENT | 组件消融/Q2对照 | Llama /14 | 默认κ=3, margin=-0.7；Q2指定对照2.5/0.965 | 原单折/test sweep须分别标注 |
| GENERIC | 五个generic任务 | 每任务层及参数见第8节 | 任务自身训练/评估划分 | 任务MC1 |
| BIPO | BiPO及校准变体 | Llama /14，FP16 | 20% dev；dev loss选checkpoint，再dev MC1选参 | 817题pooled |
| GENERATION / EFFICIENCY | 自由生成/吞吐率 | 见第9节 | 自由生成max200；计时强制50 token | 各自生成指标 |

## 3. fold的具体定义与平均公式

### 3.1 外层按问题划分，固定顺序

TruthfulQA `multiple_choice` 的 `validation` 共817题。使用Hugging Face数据顺序的0-based question ID，经 `np.array_split(sorted(unique_question_ids), 2)` 划为前409题和后408题；外层不shuffle。同题的全部答案进入同一集合。

| 对齐fold | S2S/L2S原生fold | Spherical原生fold | outer-train IDs | held-out test IDs | 训练n | 测试n |
|---|---|---|---|---|---:|---:|
| 0 | 0 | 1 | 0–408 | 409–816 | 409 | 408 |
| 1 | 1 | 0 | 409–816 | 0–408 | 408 | 409 |

Spherical原生编号与S2S/L2S相反。逐折比较须以test question IDs对齐，不能直接比较同名fold文件。

### 3.2 主实验outer-train内部划分

使用 `np.random.default_rng(42).permutation(outer_train_ids)`，`n_predictor=round(n_outer_train×0.5)`。打乱序列末尾的n_predictor题训练MLP，剩余题估计mean/geometry。已核对S2S checkpoint中的实际数量：

| fold | outer-train | geometry/mean | predictor | test |
|---|---:|---:|---:|---:|
| 0 | 409 | 205 | 204 | 408 |
| 1 | 408 | 204 | 204 | 409 |

`val_fraction=0.5` 是历史参数名，实际是predictor-training fraction，**不是独立超参数验证集**。不能把主实验描述成采用scaling的20% dev协议。历史Spherical在对应outer-train半区估计prototypes，其训练数据预算应按方法说明。

### 3.3 汇总与折间差异

- 主表：`macro=(fold0+fold1)/2`。
- 问题级合并：`pooled=(408×fold0+409×fold1)/817`。
- 对MC1/MC2/MC3分别应用公式，最后再四舍五入。
- `Avg=(MC1+MC2+MC3)/3` 是跨指标均值，不能与跨fold均值混淆。
- Llama S2S-R：macro=57.1728630%（57.17），pooled=57.1603427%（57.16）。
- Llama S2S-L：macro=43.9474567%（43.95），pooled=43.9412485%（43.94）。

Llama S2S-R折间差距为20.4581955个百分点。测试半区与训练样本同时变化，仅凭两折无法确定差距的因果，也不能据此声称跨划分稳定性。公开逐折基线有助于判断这种差异是否也影响其他方法。

## 4. 主实验实际配置与逐折结果

### 4.1 训练、prompt和scorer

已读取主结果引用的L2S/S2S checkpoint：hidden=256，epochs=2000，batch=256，lr=0.001，seed=42，fraction=0.5。当前trainer使用Adam/MSE；S2S预测器为 `Linear→GELU→Linear`，输入 `normalize(h_neg)`，`positive_mode=mean`，目标为每题正答案激活均值减该负答案激活。L2S使用自身raw hidden / Cartesian positive-negative pair目标，不能混称为同一训练定义。

S2S-L/R采用直接向量预测语义；历史checkpoint分属dyn2d/dyn3d路径，未逐权重核对时不声称它们是同一个二进制文件。特征prompt为 `Q: {question} A: {choice}`，提取目标block的最后非padding答案token激活。

归档结果记录 `few_shot=false` / `eval_mode=zero_shot`。评分使用仓库instruction加Q/A格式，teacher-forced答案token log probability求和，不做答案长度归一化、不使用chat template。`preset=qa` 与 `few_shot=false` 同时出现不表示启用了few-shot QA示例。当前zero-shot分支累计全部答案token；`lp[3:]` 属于few-shot分支，不适用于解释这些zero-shot结果。

当前统一主入口要求CUDA FP16，Spherical evaluator固定FP16。历史结果没有完整保存所有dtype、包版本、模型revision和选参轨迹，不能用当前依赖文件反推每次历史运行环境。

### 4.2 归档结果真正使用的超参数

下表从 `artifacts/main_per_fold.csv` 指向的原JSON读取，其优先级高于当前shell默认值。

| 模型 | 方法 | 对齐fold | 实际参数 |
|---|---|---|---|
| llama3.1-8B-Instruct | L2S-Linear | 0 | alpha=1.0 |
| llama3.1-8B-Instruct | L2S-Linear | 1 | alpha=1.0 |
| llama3.1-8B-Instruct | Spherical Steering | 0 | kappa=20.0, alpha=0.7, beta=-0.15 |
| llama3.1-8B-Instruct | Spherical Steering | 1 | kappa=20.0, alpha=0.7, beta=-0.15 |
| llama3.1-8B-Instruct | S2S-R | 0 | weight_kappa=3.0, margin=-0.7 |
| llama3.1-8B-Instruct | S2S-R | 1 | weight_kappa=3.0, margin=-0.7 |
| llama3.1-8B-Instruct | S2S-L (Static) | 0 | strength=3.0, slerp_t=0.2 |
| llama3.1-8B-Instruct | S2S-L (Static) | 1 | strength=3.0, slerp_t=0.2 |
| Qwen2.5-7B-Instruct | L2S-Linear | 0 | alpha=1.4 |
| Qwen2.5-7B-Instruct | L2S-Linear | 1 | alpha=1.4 |
| Qwen2.5-7B-Instruct | Spherical Steering | 0 | kappa=20.0, alpha=0.6, beta=0.4 |
| Qwen2.5-7B-Instruct | Spherical Steering | 1 | kappa=20.0, alpha=0.6, beta=0.4 |
| Qwen2.5-7B-Instruct | S2S-R | 0 | weight_kappa=3.0, margin=-0.3 |
| Qwen2.5-7B-Instruct | S2S-R | 1 | weight_kappa=3.0, margin=-0.7 |
| Qwen2.5-7B-Instruct | S2S-L (Static) | 0 | strength=3.0, slerp_t=0.2 |
| Qwen2.5-7B-Instruct | S2S-L (Static) | 1 | strength=3.0, slerp_t=0.2 |

**Qwen-7B归档与当前runner存在额外差异，必须披露：**

- 归档L2S两折alpha=1.4；统一main runner两折alpha=1.0。
- 归档S2S-R fold0 margin=-0.3、fold1 margin=-0.7，κ均为3；统一main runner两折margin均为-0.7。
- 所以归档62.55%不能标成“同一margin=-0.7的两折结果”。历史margin如何选出缺完整记录，不能补写为dev-selected。
- 保留归档值时应披露逐折参数及选参证据缺口；若改成共用参数，须完成两折新运行后更新整张对应表。表注修改不能使旧数值变成新配置的结果。

### 4.3 可放入rebuttal的主要方法逐折MC1

下表为可确认归档值，不保证与未提供的论文每一行一致。`—` 表示匹配文件缺失，不是0。

| 模型 | 方法 | fold0 MC1 (%) | fold1 MC1 (%) | macro MC1 (%) | 状态 |
|---|---|---|---|---|---|
| llama3.1-8B-Instruct | No Steering | — | — | — | 待补原始记录 |
| llama3.1-8B-Instruct | Additive Mean | — | — | — | 待补原始记录 |
| llama3.1-8B-Instruct | L2S-Linear | 37.75 | 35.45 | 36.60 | 参数见4.2 |
| llama3.1-8B-Instruct | Spherical Steering | 54.17 | 45.72 | 49.94 | 参数见4.2 |
| llama3.1-8B-Instruct | S2S-L (Static) | 49.02 | 38.88 | 43.95 | 参数见4.2 |
| llama3.1-8B-Instruct | S2S-R | 67.40 | 46.94 | 57.17 | 参数见4.2 |
| Qwen2.5-7B-Instruct | No Steering | — | — | — | 待补原始记录 |
| Qwen2.5-7B-Instruct | Additive Mean | — | — | — | 待补原始记录 |
| Qwen2.5-7B-Instruct | L2S-Linear | 37.50 | 34.96 | 36.23 | 参数见4.2 |
| Qwen2.5-7B-Instruct | Spherical Steering | 48.77 | 48.41 | 48.59 | 参数见4.2 |
| Qwen2.5-7B-Instruct | S2S-L (Static) | 46.32 | 43.77 | 45.04 | 参数见4.2 |
| Qwen2.5-7B-Instruct | S2S-R | 66.18 | 58.92 | 62.55 | 参数见4.2 |

### 4.4 逐折MC1/MC2/MC3完整表

来源：`artifacts/main_per_fold.csv`。对齐fold0测试408题，fold1测试409题。

| 模型 | 方法 | fold | MC1 (%) | MC2 (%) | MC3 (%) |
|---|---|---|---|---|---|
| llama3.1-8B-Instruct | L2S-Linear | 0 | 37.75 | 55.96 | 29.40 |
| llama3.1-8B-Instruct | L2S-Linear | 1 | 35.45 | 56.58 | 28.93 |
| llama3.1-8B-Instruct | Spherical Steering | 0 | 54.17 | 71.37 | 43.89 |
| llama3.1-8B-Instruct | Spherical Steering | 1 | 45.72 | 65.63 | 38.16 |
| llama3.1-8B-Instruct | S2S-R | 0 | 67.40 | 78.46 | 63.19 |
| llama3.1-8B-Instruct | S2S-R | 1 | 46.94 | 65.06 | 39.02 |
| llama3.1-8B-Instruct | S2S-L (Static) | 0 | 49.02 | 67.63 | 41.73 |
| llama3.1-8B-Instruct | S2S-L (Static) | 1 | 38.88 | 61.94 | 34.29 |
| Qwen2.5-7B-Instruct | L2S-Linear | 0 | 37.50 | 57.66 | 28.20 |
| Qwen2.5-7B-Instruct | L2S-Linear | 1 | 34.96 | 55.91 | 27.12 |
| Qwen2.5-7B-Instruct | Spherical Steering | 0 | 48.77 | 65.67 | 38.20 |
| Qwen2.5-7B-Instruct | Spherical Steering | 1 | 48.41 | 68.26 | 40.04 |
| Qwen2.5-7B-Instruct | S2S-R | 0 | 66.18 | 78.26 | 58.35 |
| Qwen2.5-7B-Instruct | S2S-R | 1 | 58.92 | 73.58 | 51.16 |
| Qwen2.5-7B-Instruct | S2S-L (Static) | 0 | 46.32 | 65.84 | 36.94 |
| Qwen2.5-7B-Instruct | S2S-L (Static) | 1 | 43.77 | 65.57 | 36.94 |

No Steering/Additive应先找回匹配原文件；如重跑，保存新model revision、dtype、prompt、参数和逐题ID，标为新结果。不能用其他实验的closed-gate结果替代原主表基线。

## 5. Qwen-3B/14B争议数字的精确配置

### 5.1 模型与训练身份

两个运行均为full data、smoke=false、tune=true、seed42、hidden256、epochs2000、batch256、lr0.001、dev_fraction0.2。共同归档source fingerprint为：

`ffc92975c59a08ac504b50ffd70e0616e5e974fbaa20c7d065f6ddefae6715de`

| ID | 官方模型 | revision | block | 精度 |
|---|---|---|---:|---|
| SCALE-3B | Qwen/Qwen2.5-3B-Instruct | aa8e72537993ba99e69dfaafa59ed015b17504d1 | 22 | BF16，不量化 |
| SCALE-14B | Qwen/Qwen2.5-14B-Instruct | cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 | 32 | NF4 4-bit + double quant，BF16 compute |

每折outer-train留82题dev，余下再分geometry/predictor；实际question IDs保存在对应fold的split.json。

| fold | outer-train | geometry | predictor | dev | test |
|---|---:|---:|---:|---:|---:|
| 0 | 409 | 163 | 164 | 82 | 408 |
| 1 | 408 | 163 | 163 | 82 | 409 |

每折仅在自己的dev上按MC1、MC2、网格顺序选参，再评估本折test。不能合并两折dev后仍称为独立nested test，因为一折dev可能属于另一折test。

prompt标记为 `repository_zero_shot_instruction_no_chat_template`，feature prompt为 `repository_standard_Q_question_A_choice`。使用scaling自己的completion-token边界/scorer，不能仅凭同名MC指标就断言与主表全部运行细节相同。

### 5.2 产生32.68%/44.68%的S2S-L配置

| 模型 | fold | strength | slerp_t | 单折MC1 (%) | 两折pooled MC1 (%) |
|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0 | 2.0 | 0.5 | 32.60 | 32.68 |
| Qwen2.5-3B-Instruct | 1 | 3.0 | 0.0 | 32.76 | 32.68 |
| Qwen2.5-14B-Instruct | 0 | 3.0 | 0.2 | 48.53 | 44.68 |
| Qwen2.5-14B-Instruct | 1 | 3.0 | 1.0 | 40.83 | 44.68 |

- 32.68%来自 `(408×32.5980392%+409×32.7628362%)/817`。
- 44.68%来自 `(408×48.5294118%+409×40.8312958%)/817`。

公式中的折分数是展示近似，实际计算使用JSON全精度分数。14B的48.5294%也不是审稿人所引48.29%，不能据此认定后者是某一折结果。

### 5.3 同次scaling运行的主要基线逐折结果与参数

这些值只适用于对应scaling协议，不能填到Llama/Qwen-7B原主表缺失位置。

| 模型 | 方法 | fold0 (%) | fold1 (%) | pooled (%) | fold0参数 | fold1参数 |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | No Steering | 26.96 | 24.21 | 25.58 | 无干预 | 无干预 |
| Qwen2.5-3B-Instruct | Additive Mean | 30.15 | 29.58 | 29.87 | alpha=2.0 | alpha=2.0 |
| Qwen2.5-3B-Instruct | L2S-Linear | 24.75 | 22.98 | 23.87 | alpha=1.4 | alpha=0.5 |
| Qwen2.5-3B-Instruct | Spherical Steering | 44.36 | 39.61 | 41.98 | kappa=50.0; alpha=0.6; beta=0.1 | kappa=50.0; alpha=0.6; beta=0.1 |
| Qwen2.5-3B-Instruct | S2S-L (Static) | 32.60 | 32.76 | 32.68 | strength=2.0; slerp_t=0.5 | strength=3.0; slerp_t=0.0 |
| Qwen2.5-3B-Instruct | S2S-R | 54.66 | 54.03 | 54.35 | weight_kappa=5.0; margin=-0.7 | weight_kappa=3.0; margin=-0.3 |
| Qwen2.5-14B-Instruct | No Steering | 41.91 | 37.16 | 39.53 | 无干预 | 无干预 |
| Qwen2.5-14B-Instruct | Additive Mean | 42.40 | 39.36 | 40.88 | alpha=1.4 | alpha=1.4 |
| Qwen2.5-14B-Instruct | L2S-Linear | 42.65 | 39.12 | 40.88 | alpha=1.0 | alpha=1.0 |
| Qwen2.5-14B-Instruct | Spherical Steering | 42.89 | 38.39 | 40.64 | kappa=20.0; alpha=0.3; beta=0.1 | kappa=50.0; alpha=0.6; beta=0.1 |
| Qwen2.5-14B-Instruct | S2S-L (Static) | 48.53 | 40.83 | 44.68 | strength=3.0; slerp_t=0.2 | strength=3.0; slerp_t=1.0 |
| Qwen2.5-14B-Instruct | S2S-R | 60.29 | 50.12 | 55.20 | weight_kappa=3.0; margin=-0.7 | weight_kappa=1.0; margin=-0.7 |

### 5.4 该次dev搜索网格

| 方法 | 搜索空间 |
|---|---|
| Additive/L2S | alpha={0.5,1,1.4,2} |
| Spherical | kappa={1,3,5,10,20,50} × alpha={0.15,0.3,0.6} × beta={0.1,0.4}，36组 |
| S2S-L | strength={0.5,1,2,3} × t={0,0.2,0.5,0.8,1}，20组 |
| S2S-R | weight_kappa={1,3,5} × margin={-0.7,-0.3,0}，9组 |
| Baseline | 无搜索 |

后续 `spherical_retuned_v1`、`shared_hparam_eval`、`compare_slerp` 均需各自标明协议，不能将结果静默贴回原表。共享dev/某折选出的公共设置不能混称为独立逐折选参。

### 5.5 36.68%/48.29%的处理原则

`artifacts/number_provenance.json` 中两者状态均为 `unresolved_no_matching_audited_artifact`。这只表示当前已审计记录无法确认来源，不证明历史上不存在该运行。

建议将能够明确绑定SCALE-3B/SCALE-14B的表统一为32.68%/44.68%，并加上以上配置。另一张历史表须回查原JSON、checkpoint、实际fold、precision、strength/t、prompt、选参和均值口径；若无法确认，应明确撤回未核实条目并给出可追溯的新表。**不能未经核实称为笔误，也不能归因于量化或fold差异。**

## 6. 噪声表49.02与主表43.95

### 6.1 原始记录与计算

文件：`results/experiment/l2s_noise_robustness/llama3.1-8B-Instruct/fold0/noise_0/s2s_l_static.json`。

| 字段 | 值 |
|---|---|
| 模型/block | Llama-3.1-8B-Instruct /14 |
| fold/test | 0 /408题，IDs409–816 |
| 方法 | S2S-L-Static |
| strength / slerp_t | 3.0 /0.2 |
| few_shot / preset | false /qa |
| residual_noise_sigma / seed | 0.0 /42 |
| MC1 / MC2 / MC3 | 49.0196078% /67.6289095% /41.7273576% |

干预定义：`V=normalize(v_avg)`，`R=normalize(v_hat-v_avg)`，`D=SLERP(V,R,0.2)`，`h'=h+3×||v_hat||×D`。

主表对应fold1 MC1=38.8753056%；因此 `(49.0196078+38.8753056)/2=43.9474567%`，显示43.95%。差别是单折与双折平均，不是同一汇总条件下的冲突。

### 6.2 历史inference-noise与当前target-noise应分别说明

上述历史clean文件记录的是 `residual_noise_sigma=0`；历史目录还有inference residual-noise非零文件。不能把当前训练目标噪声脚本的定义反套给旧图，也不能从同一个数值推断某噪声条件生效与否。

当前 `experiments/l2s_noise_robustness/run.sh` 只扰动MLP训练target：

`v_tilde_i=v_i+sigma×||v_i||×normalize(epsilon_i), epsilon_i~N(0,I)`。

每个target在训练前采样一次、在全部epoch/batch固定。原输入、geometry mean、held-out test保持clean；inference residual noise强制0。包含L2S-Linear、L2S-SLERP、S2S-L-Static、S2S-R。

| 入口 | noise levels默认值 | fold覆盖 |
|---|---|---|
| 原生run.sh | 1,2,3,4,5，不含clean | 读取模型配置/override；不能默认称两折 |
| 统一 `reproducibility.run noise` | 0,0.1,0.2,0.5,1,2,3,4,5 | 默认显式运行0/1两折 |

新target-noise图须使用新结果、明确训练是否重跑、噪声位置及汇总方式；不能仅改旧图标签。

## 7. 可直接使用的英文表注

**主表（MAIN-ARCHIVE）：**

> TruthfulQA results use question-level two-fold evaluation. Fold 0 tests questions 409–816 (n=408), while fold 1 tests questions 0–408 (n=409). Each learned predictor uses only its outer-training half, partitioned into predictor and geometry subsets with fraction 0.5 and seed 42. Scores are arithmetic means of the two unrounded fold scores. Methods are aligned by held-out question IDs. Method- and fold-specific parameters are reported separately, including the archived Qwen-7B S2S-R margins of −0.3 and −0.7 for folds 0 and 1.

**规模表：**

> Scaling results reserve 20% of each outer-training half for development, select parameters independently within each fold, and pool scores over all 817 held-out questions. Qwen2.5-3B uses BF16 at layer 22; Qwen2.5-14B uses NF4 weights with BF16 computation at layer 32. S2S-L uses the static interpolation implementation with the fold-specific development-selected strength and interpolation coefficient listed in the configuration table.

**历史噪声表：**

> Llama-3.1-8B-Instruct, layer 14, fold 0 only (408 held-out questions), S2S-L-Static with strength 3.0 and interpolation coefficient 0.2. The clean condition uses inference residual-noise sigma 0. Its MC1 of 49.02% is a single-fold score; Table 1 reports the two-fold arithmetic mean of 43.95%.

## 8. 五个Generic任务的所有模型/任务配置

以下为当前 `hyperparams/{method}/{LLAMA,QWEN}/generic/<dataset>.sh` 的原生默认值，**不是历史结果的执行证明**。模型族分别指Llama-3.1-8B-Instruct和Qwen2.5-7B-Instruct。

所有行FP16；学习方法NUM_FOLDS=1，batch256、lr0.001、seed42；在任务训练集按fraction分predictor/geometry，评估用任务自身划分。每行三个学习方法的layer/hidden/epochs/fraction相同；Spherical有自己的prototype估计协议。参数顺序按表头，不能把各方法的κ视为同一含义。

| 模型族 | 任务 | layer | hidden | epochs | predictor fraction | L2S α | Spherical κ/α/β | S2S-L strength/t | S2S-R κ/margin |
|---|---|---|---|---|---|---|---|---|---|
| LLAMA | BoolQ | 12 | 256 | 2000 | 0.5 | 1.0 | 20/0.5/-1 | 0.5/0.07 | 7/0.9 |
| LLAMA | COPA | 16 | 256 | 2000 | 0.5 | 1.0 | 20/0.9/-0.4 | 0.8/0.7 | 10/0 |
| LLAMA | StoryCloze | 27 | 256 | 2000 | 0.5 | 1.0 | 20/0.9/-0.7 | 4.0/0.1 | 2/-0.5 |
| LLAMA | MMLU-Global | 28 | 256 | 2000 | 0.5 | 0.05 | 20/0.8/-0.1 | 0.2/0.7 | 3/0.3 |
| LLAMA | Winogrande | 16 | 256 | 2000 | 0.5 | 1.0 | 20/0.9/-0.8 | 4.4/0.4 | 3/-0.8 |
| QWEN | BoolQ | 12 | 256 | 2000 | 0.8 | 1.4 | 20/0.5/-1 | 3.0/0.2 | 4/-0.8 |
| QWEN | COPA | 16 | 1024 | 400 | 0.9 | 1.4 | 20/0.9/-0.4 | 3.0/0.2 | 10/-0.3 |
| QWEN | StoryCloze | 27 | 256 | 2000 | 0.6 | 1.4 | 20/0.9/-0.7 | 3.0/0.2 | 4/-0.8 |
| QWEN | MMLU-Global | 28 | 256 | 2000 | 0.2 | 1.4 | 20/0.8/-0.1 | 3.0/0.2 | 4/0.3 |
| QWEN | Winogrande | 16 | 256 | 2000 | 0.4 | 1.4 | 20/0.9/-0.8 | 3.0/0.2 | 3/-0.9 |

Generic的prompt、长度归一化和评估子集依任务loader/evaluator定义，不能套用TruthfulQA主表raw summed-answer scorer。MMLU-Global采用仓库balanced subset，不能写成完整MMLU test。层编号保留当前配置原值，实际运行仍需检查各evaluator的层索引约定。

**入口覆盖差异：** 统一generic runner显式传EPOCHS=2000，会覆盖Qwen COPA原生400 epochs。因此上表不能当作统一runner所有运行的最终配置；实际值以打印计划与receipt为准，其他环境override同理。

## 9. 其余实验配置登记

下表补充仓库其余实验。脚本存在/可打印计划不代表已经完成实验；默认配置也不能替代历史图表的实际manifest。

| 实验 | 已确认配置、入口差异与报告口径 |
|---|---|
| component_ablation | 历史论文说明Llama L14 fold0；原生run.sh当前默认fold1；统一入口显式两折。六行含mean、mean+AW+Gate、MLP、固定spherical、adaptive、full；默认κ3/margin-0.7 |
| Q2指定mean+AW+Gate | fold1 38.8753056%来自κ2.5/margin0.965的历史test grid。统一入口提供两折相同参数，但本次未执行。不能把另一个test最优单元替代该指定配置 |
| mean-control grid | 原生run_vavg_sweep.sh默认fold0；κ={0.1,0.3,0.5,1,2,3,5,10} × margin={-0.7,0.34,0.5,0.7,0.9,0.95,0.99,1}；按test MC1选最佳是探索性test sweep |
| gate_correctness | 统一两折S2S-R，κ3/margin-0.7，applied与observe-only分开；按正确/错误candidate记录gate。teacher-forced gate不能证明自由生成安全性 |
| margin_ablation / margin_gate_correctness | 统一margin={-0.8,-0.7,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1}、κ3、两折。原生旧网格漏-0.7且重复1.0，必须注明入口 |
| weight_kappa_ablation | 原生κ={0.5,1,2,4,8,16,32}；Llama/Qwen两折；其余训练/推理取模型配置 |
| mlp_hidden_ablation | README标称16–2048两折；但当前run.sh引用的hyperparams/dyn3d_sga/experiments/mlp_hidden_ablation.sh缺失，MLP_HIDDEN_VALUES定义未找到，不能声称完整网格已可复现 |
| val_fraction_ablation | fraction={0.1,…,0.9}，seeds={42,123,0}，原生默认fold0；跨seed统计不等于跨fold统计 |
| dyn2d_ablation | strength={1,2,3,4,5,6} × t={0,0.2,0.4,0.6,0.8,1}；Llama/Qwen两折；每个相同网格单元取两折算术均值，若再挑test最佳需标明 |
| layer_performance | Llama blocks0–31两折；launcher fast：L strength={1,2},t={0.2,0.6}；R κ={3,10},margin={-0.7,-0.3}。每层选两折均值最大的共享参数点，selection=test_set_sweep_best；run.py直接调用默认full，与launcher不同 |
| steering_vector_ablation | 共享均值/逐样本、linear/spherical对照；LINEAR_ALPHA默认1、各模型全部层；具体fold/变体须取结果元数据，不把oracle向量当作预测器 |
| residual_analysis | FAMILY=LLAMA，METHOD=S2S_R，可改S2S_L_Static；oracle alignment、shuffle、residual swap。R的Mean-only为cross-fitted Spherical；L的Mean-only为线性h+strength*v_avg，二者不能混称同一mean baseline |
| truthfulqa_generic_transfer | Llama；在各generic目标层训练TruthfulQA两折MLP并保留各源fold的TruthfulQA mean，generic只评估；目标层/κ/margin见第8节Llama R，报告源fold均值/sample SD |
| mlp_prediction_error | 官方generic validation/test与同层TruthfulQA held-out对照；raw-pair cosine/relative norm diagnostics与angle/MAE协议不同；cached-train CV不能混称官方测试 |
| geometry_decision_alignment | pilot每任务最多128题、COPA100，seed20261111，batch4；full runner另评完整声明集合。题标签选权重/oracle仅诊断上界；sampled/full、FP16 finite-difference精度须区分 |
| kappa_view | token-level三方向权重密度、可选gate过滤；须记录κ/token集合/过滤规则；不是新主表MC运行 |
| generation_efficiency | Llama L14、S2S fold0配Spherical native fold1；单RTX3090设计，greedy强制50最终token，warmup、CUDA同步、5计时重复，preset=null；generation-only/end-to-end分别报告 |
| generation / safety | greedy，max_new_tokens200，不强制最短长度；同prompt IDs，记录EOS/length。S2S默认three-pass frozen-probe。缺原MMsafe数据、judge及历史生成记录，不能验证历史安全百分数；文本adapter不接受多模态输入 |
| bipo_truthfulqa | Llama L14 FP16，seed42，20 epochs，batch2，lr5e-4，β0.1，cosine，warmup100，clip1，weight_decay0；训练/推理all-token；20% dev、dev loss选checkpoint，再dev MC1选参；BiPO alone与BiPO+S2S分行 |
| 历史分析/绘图 | l2s_reliability、部分cosine_similarity入口仍引用已移除method.l2s；truthfulqa、word_number为比较/绘图工具，以实际输入决定协议，不能声称所有旧入口可运行 |

BiPO当前归档top-level pooled MC1=62.1787026%，与review中60.18%尚无匹配证据，不能静默替换。吞吐率强制50-token设计用于固定分母，不能作为自由生成EOS/安全性结果。

### 9.1 layer_performance的其余预设

| profile | S2S-L strength | S2S-L t | S2S-R κ | S2S-R margin |
|---|---|---|---|---|
| fast | 1,2 | 0.2,0.6 | 3,10 | -0.7,-0.3 |
| balanced | 1,2,3 | 0.2,0.6,1 | 1,3,10 | -0.7,-0.3,0.3 |
| full | 0.5,1,1.5,2,2.5,3 | 0,0.2,0.4,0.6,0.8,1 | 1,2,3,5,10,20 | -0.9,-0.7,-0.5,-0.3,0,0.3 |

训练默认hidden256/epochs2000/batch256/lr0.001/seed42/fraction0.5，全部仍为test sweep，不是独立dev selection。

## 10. 已存在的规模实验模型配置

逐项读取实际run_config.json；有summary仅表示文件存在，不代表所有独立验证通过。未找到32B同层级正式run_config，README支持不等于已完成运行。

| 模型 | block | precision | revision | 状态 |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 16 | bfloat16 | 7ae557604adf67be50417f59c2c2f167def9a775 | 有summary |
| Qwen2.5-1.5B-Instruct | 20 | bfloat16 | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | 有summary |
| Qwen2.5-14B-Instruct | 32 | NF4/BF16 compute | cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 | 有summary |
| Qwen2.5-3B-Instruct | 22 | bfloat16 | aa8e72537993ba99e69dfaafa59ed015b17504d1 | 有summary |
| Qwen3.5-9B | 21 | bfloat16 | c202236235762e1c871ad0ccb60c8ee5ba337b9a | 未见summary |

## 11. 复现入口与环境边界

下面默认仅打印完整计划；加 `--execute --resume` 才实际执行。本文没有执行模型实验。应在安装环境、下载模型后使用独立新输出目录，保留完整receipt。

```bash
python -m reproducibility.run main --model llama --model-dir models/Llama
python -m reproducibility.run main --model qwen --model-dir models/Qwen
python -m reproducibility.run scaling --scaling-models qwen3 qwen14
python -m reproducibility.run noise --model llama --model-dir models/Llama
python -m reproducibility.run generic --model llama --model-dir models/Llama
python -m reproducibility.run component --model llama --model-dir models/Llama
python -m reproducibility.run gate --model llama --model-dir models/Llama
python -m reproducibility.run margin --model llama --model-dir models/Llama
python -m reproducibility.run bipo --model llama --model-dir models/Llama
python -m reproducibility.run generation --model llama --model-dir models/Llama --prompts data/mmsafe.jsonl
```

注意：统一main的Qwen参数差异见4.2。复查历史checkpoint时，L2S两折需显式alpha1.4，S2S-R fold0需κ3/margin-0.3，fold1需κ3/margin-0.7；使用原evaluator并完整记录checkpoint/output/prompt/model revision，不能用新默认配置宣称恢复原数。

当前依赖文件记录Python3.10、主环境torch2.9.0/Transformers4.57.6/NumPy2.2.6；scaling环境Transformers5.20.2、NF4使用bitsandbytes0.50.2。这里仅转述仓库pin，不声称本次安装验证，也不据此推断历史每次运行环境。历史主模型revision缺失项应明确标unknown。

## 12. 可补入正文的英文实验协议

下段用于主实验，须同时提供第4.2节的模型/逐折参数表。

> We evaluate TruthfulQA using question-level two-fold evaluation over its 817 multiple-choice questions. In the dataset's original Hugging Face order, questions 0–408 form one half and questions 409–816 form the other. Fold 0 trains on the first half and tests on the second (408 questions); fold 1 reverses these roles (409 test questions). For learned predictors, each outer-training half is further partitioned into predictor-training and geometry subsets using a predictor fraction of 0.5 and seed 42; no held-out test question is used in either subset. Table 1 reports the arithmetic mean of the two unrounded fold scores. We report fold-level scores and align methods by held-out question identities, since the native Spherical Steering fold labels use the opposite convention. Model-, method-, and fold-specific parameters are disclosed in the configuration appendix.

规模实验须另段说明：

> The model-scaling study uses a separate development-selection protocol: 20% of each outer-training half is reserved for development, while the remainder supplies geometry and predictor training. Parameters are selected independently within each fold using development MC1, with MC2 and grid order as tie breakers. Scaling scores pool the two disjoint test folds over all 817 questions. Qwen2.5-3B uses BF16 and Qwen2.5-14B uses NF4 weights with BF16 computation.

## 13. 英文rebuttal草稿

以下为拟提交文本，保留will以符合当前完成状态；实际补完论文/基线后再相应修改。没有将尚未追溯的数字定性为笔误。

> Thank you for checking the fold averages and identifying the reporting inconsistencies. We agree that the approximately 20-point fold difference makes the exact split and per-fold baseline comparisons necessary. We will add the question-level split and per-fold results to the manuscript. Fold 0 trains on question IDs 0–408 and tests on IDs 409–816 (n=408); fold 1 reverses these roles (n=409). Table 1 uses the arithmetic mean of the two unrounded fold scores. For example, the archived Llama S2S-R scores are 67.4019608% and 46.9437653%, giving 57.1728630% (57.17%).
>
> The Llama S2S-L noise value of 49.02% is the clean fold-0-only score, rather than a two-fold average. Its corresponding fold-1 score is 38.8753056%, giving the Table 1 arithmetic mean of 43.9474567% (43.95%). The archived clean setting uses Llama-3.1-8B-Instruct at layer 14, S2S-L-Static with strength 3.0 and interpolation coefficient 0.2, and inference residual-noise sigma 0. We will state the fold, noise location, and aggregation explicitly.
>
> For scaling, the available run records support S2S-L MC1 of 32.68% for Qwen2.5-3B-Instruct (BF16, layer 22) and 44.68% for Qwen2.5-14B-Instruct (NF4 with BF16 computation, layer 32). These runs use per-fold development selection and question-weighted pooling, unlike the main table's arithmetic fold mean. The selected (strength, interpolation) pairs are (2.0, 0.5) and (3.0, 0.0) for the 3B folds, and (3.0, 0.2) and (3.0, 1.0) for the 14B folds. We have not established matching provenance for the previously quoted 36.68% and 48.29%, so we cannot attribute them to a specific configuration or a typographical error. We will replace the unverified entries with one consistently documented, traceable result set and identify the correction explicitly.
>
> The audit additionally shows that the archived Qwen-7B L2S baseline uses alpha 1.4 and its S2S-R margins differ across folds (−0.3 for fold 0 and −0.7 for fold 1). We will disclose these settings, avoid describing the archived mean as a common-margin comparison, and separate any new fixed-parameter reruns from the archived results.

在缺失的No Steering/Additive逐折结果补齐前，不应写“所有主要基线已完整报告”。若回复必须先展示表，相关项需标pending；建议补齐后再提交最终稿。

## 14. 提交前证据清单

- [ ] 找到两份原回复表/版本，每表绑定配置ID。
- [ ] 追溯36.68%/48.29%；无法确认则明确撤回并使用可追溯结果，不编造原因。
- [ ] 补齐原主表No Steering/Additive逐折原文件；新运行明确标新身份。
- [ ] 披露Qwen-7B逐折margin；若统一参数，完成两折新运行再换表。
- [ ] 正文写明question IDs、predictor/geometry split和均值公式。
- [ ] 噪声图明确legacy inference noise或新target noise，标注fold/σ/噪声位置。
- [ ] 每表保存revision、precision、layer、checkpoint、prompt、scorer、selection、fold IDs和输出文件。
- [ ] 所有“we have added/corrected/rerun”与实际论文修改/运行状态一致。

## 15. 文件证据及此次读取的SHA-256

原主表文件名与method/fold映射见 `artifacts/main_per_fold.csv`。`artifacts/evidence.json` 保留已有审计source/hash和逐题分数。下面固定本说明直接读取的关键输入身份；哈希不代表GPU重跑验证。

| 源文件（相对仓库根目录） | SHA-256 |
|---|---|
| artifacts/main_per_fold.csv | `52761ae804f6f43fb28d9019841483715456247d4887c22f421abbe31611975f` |
| artifacts/main_averages.csv | `a99af71affbb4d01485d9bcb5b49c269f71d0583b1d214dfce7fbaa72d54b2e9` |
| artifacts/number_provenance.json | `c26d87d9ba67a91bb1e7bc7c69ebaf8489b73cf0be0027d694ef3122f13e3c7c` |
| artifacts/audit_status.json | `09bd35fb659828ad712192be5a0b315ee0d7730295fd2e0016445041d5fcb533` |
| results/experiment/l2s_noise_robustness/llama3.1-8B-Instruct/fold0/noise_0/s2s_l_static.json | `b7053a1d808737d89709f72f3bd19c6ee3adb320b723181992f2bcd50cd6f4c0` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/run_config.json | `86a321578a0b4fff598274a0c1d407c9cea660e9fe07f334de0a5aaddc1419ad` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/summary.json | `4a657aa11fe4002e1accab0c97430cb1f06942021258c29c5ba2a625bb5fc8ab` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/fold0/split.json | `92f487970b1f5f0d230d6448b2bfd248d0fd448385c14c5927cff602a9572154` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/fold0/s2s_l_selection.json | `30acc8a33c84d92c33ebc333d969c4b6d68d1477e38b192740663022f327e0fa` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/fold0/s2s_l_test.json | `f343db0612453efd1ec518cac48bccbc422720a458719755092df85cb3a82a1b` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/fold1/split.json | `addad57a5b0e2f51e5c61c882341c6546418db367912e547d807585d76d3ddd4` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/fold1/s2s_l_selection.json | `00c539c978cff9442503a2c49a4b6f3a91a93db1f532ec9056c17a6f39691aa3` |
| results/experiment/model_scaling_ablation/Qwen2.5-3B-Instruct/bfloat16/fold1/s2s_l_test.json | `d0568d193e7cf19ca6112ffd6741d46471833f5b1fc59edb6e6370e5b75931b4` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/run_config.json | `f654fbd26ed754fcc370443ecbb21bcd0090e78ac15a19209bd37699fb6ab1b4` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/summary.json | `974e969bd6fbc7107cf3ea7851dff74ab522c520f35b470e29a2162e51f5f8ba` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/fold0/split.json | `92f487970b1f5f0d230d6448b2bfd248d0fd448385c14c5927cff602a9572154` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/fold0/s2s_l_selection.json | `2431df8ea2ad685c95e59a12a94e36f3373f65ddf44ef431cbad6950eb54bded` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/fold0/s2s_l_test.json | `d1f6904ff39c9f52db70705b92a8df9ecd2ea8a8a04c8e3b397f004da4fd4d08` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/fold1/split.json | `addad57a5b0e2f51e5c61c882341c6546418db367912e547d807585d76d3ddd4` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/fold1/s2s_l_selection.json | `16c74ee30c612b29aff8a9a29f7e6beafdd552b4b0fb5126ca3c351273b9e5f8` |
| results/experiment/model_scaling_ablation/Qwen2.5-14B-Instruct/nf4/fold1/s2s_l_test.json | `553f49d3fa3422c5c9b66903eaa540a5aac9e193b8440f074456a06f75071999` |

补充依据：`docs/FOLD_PROTOCOL.md`、`docs/RESULT_PROVENANCE.md`、`docs/EXPERIMENTS.md`、各实验README、`reproducibility/run.py`及实际trainer/evaluator。若文档默认值与原JSON冲突，保留原参数并说明冲突，不能用当前默认值覆盖历史事实。
