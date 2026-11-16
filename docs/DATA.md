# Data, models, and result boundaries

## Included

- The pinned local TruthfulQA CSV snapshot in `data/truthfulqa/TruthfulQA.csv`.
- The local evaluation helpers in `truthfulqa/`, with upstream notices preserved.
- Archived scores, per-fold tables, and provenance in `results/reference/`.

The TruthfulQA loaders still use the Hugging Face multiple-choice split for question alignment. Other tasks are obtained by their native dataset loaders; StoryCloze and gated models require provider access.

## Inputs not bundled

- Original MMsafe prompts, safety judge/configuration, and historical generation records.
- Original ITI/SADI/LoFiT/LoReFT external implementations and vector artifacts, including the full Figure 4(c)/Figure 5-left inputs.
- Model weights and the two local TruthfulQA truth/info judge models.

No replacement scores are invented for missing artifacts. The text generation adapter rejects multimodal inputs rather than dropping their images/audio/video.

## MMsafe file schemas

| File | Required fields |
|---|---|
| Prompt JSONL | `id`, `prompt` |
| Original judge-score JSONL | `id`, `method`, `unsafe_probability` |
| Judge configuration JSON | `judge_id`, `revision`, `input_format`, `unsafe_score_definition` |

All methods must cover the same prompt IDs. Scores are unsafe probabilities in [0,1]. The implemented safety rule is `unsafe_probability < threshold`; EOS and token-limit outcomes are recorded independently.

## Archived versus new results

MC values in the reference CSV/JSON are fractions. `main_per_fold.csv` aligns held-out test halves; `main_averages.csv` preserves arithmetic fold means and question-weighted pooling. Original evidence IDs and score values are retained. `audit_status.json` identifies unresolved source records.

Fresh runs save resolved settings under `results/reproduction/run_configs/` and native outputs in the paths shown by the plan. CPU checks and parser checks do not establish final benchmark accuracy, and no full GPU rerun is claimed by this organization update.
