"""Question-paired gate-rate analysis, conditional on the fixed checkpoints."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

WINDOWS = ("answer_tokens", "scored_answer_predictors", "legacy_intervention_segment")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(correct, incorrect, folds, repeats=20000, seed=42):
    """Resample questions jointly for the two labels, preserving fold sizes."""
    correct, incorrect = np.asarray(correct), np.asarray(incorrect)
    folds = np.asarray(folds)
    if not (correct.shape == incorrect.shape == folds.shape) or not len(correct):
        raise ValueError("Nonempty paired vectors required")
    rng = np.random.default_rng(seed)
    draws = np.zeros((repeats, 2))
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        # Bound temporary memory for a full 817-question run.
        for start in range(0, repeats, 1000):
            count = min(1000, repeats - start)
            resampled = rng.choice(indices, size=(count, len(indices)), replace=True)
            draws[start:start + count, 0] += correct[resampled].sum(axis=1) / len(correct)
            draws[start:start + count, 1] += incorrect[resampled].sum(axis=1) / len(correct)
    difference = incorrect - correct
    ci = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
    return {
        "correct_rate": float(correct.mean()), "incorrect_rate": float(incorrect.mean()),
        "correct_ci95": ci(draws[:, 0]), "incorrect_ci95": ci(draws[:, 1]),
        "incorrect_minus_correct": float(difference.mean()),
        "difference_ci95": ci(draws[:, 1] - draws[:, 0]),
        "questions_incorrect_higher": int((difference > 0).sum()),
        "questions_correct_higher": int((difference < 0).sum()),
        "questions_tied": int((difference == 0).sum()),
        "questions": len(correct),
    }


def mc_metrics(correct, incorrect):
    t = np.asarray([r["log_probability"] for r in correct])
    f = np.asarray([r["log_probability"] for r in incorrect])
    answers = [r["answer"] for r in correct]
    best = answers.index(correct[0]["best_answer"])
    offset = max(t.max(), f.max())
    return {"MC1": float(t[best] > f.max()),
            "MC2": float(np.exp(t - offset).sum() / (np.exp(t - offset).sum() + np.exp(f - offset).sum())),
            "MC3": float((t > f.max()).mean())}


def analyze(records, out, repeats=20000, seed=42):
    out = Path(out)
    groups, pairs = defaultdict(list), defaultdict(dict)
    for r in records:
        groups[(r["mode"], r["fold"], r["hf_idx"])].append(r)
        key = (r["fold"], r["hf_idx"], r["label"], r["candidate_index"])
        if r["mode"] in pairs[key]:
            raise ValueError("Duplicate candidate-mode record")
        pairs[key][r["mode"]] = r
    question_rows = []
    for (mode, fold, qid), candidates in sorted(groups.items()):
        correct = [r for r in candidates if r["label"] == "correct"]
        incorrect = [r for r in candidates if r["label"] == "incorrect"]
        if not correct or not incorrect:
            raise ValueError("Incomplete question: both classes are required")
        row = {"mode": mode, "fold": fold, "hf_idx": qid,
               "question": candidates[0]["question"], "category": candidates[0]["category"],
               "correct_candidates": len(correct), "incorrect_candidates": len(incorrect),
               **mc_metrics(correct, incorrect)}
        for window in WINDOWS:
            for label, members in (("correct", correct), ("incorrect", incorrect)):
                row[f"{window}_{label}_rate"] = float(np.mean([r["windows"][window]["rate"] for r in members]))
                row[f"{window}_{label}_tokens"] = sum(r["windows"][window]["tokens"] for r in members)
                row[f"{window}_{label}_triggered"] = sum(r["windows"][window]["triggered"] for r in members)
            row[f"{window}_incorrect_minus_correct"] = row[f"{window}_incorrect_rate"] - row[f"{window}_correct_rate"]
        question_rows.append(row)
    write_csv(out / "questions.csv", question_rows)
    summaries, flat = {}, []
    modes = sorted({r["mode"] for r in records})
    for mode in modes:
        summaries[mode] = {}
        for scope in ["pooled"] + [f"fold{f}" for f in sorted({r["fold"] for r in records})]:
            rows = [r for r in question_rows if r["mode"] == mode and (scope == "pooled" or scope == f"fold{r['fold']}")]
            group_records = [r for r in records if r["mode"] == mode and (scope == "pooled" or scope == f"fold{r['fold']}")]
            summary = {"questions": len(rows), "metrics": {k: float(np.mean([r[k] for r in rows])) for k in ("MC1", "MC2", "MC3")}, "windows": {}}
            for window in WINDOWS:
                stats = paired_bootstrap([r[f"{window}_correct_rate"] for r in rows],
                                         [r[f"{window}_incorrect_rate"] for r in rows],
                                         [r["fold"] for r in rows], repeats, seed)
                stats["classes"] = {}
                for label in ("correct", "incorrect"):
                    members = [r for r in group_records if r["label"] == label]
                    total = sum(r["windows"][window]["tokens"] for r in members)
                    triggered = sum(r["windows"][window]["triggered"] for r in members)
                    stats["classes"][label] = {
                        "candidates": len(members), "tokens": total, "triggered": triggered,
                        "micro_token_rate": triggered / total,
                        "candidate_mean_rate": float(np.mean([r["windows"][window]["rate"] for r in members])),
                        "candidate_any_trigger_rate": float(np.mean([r["windows"][window]["triggered"] > 0 for r in members])),
                        "mean_candidate_tokens": total / len(members),
                    }
                summary["windows"][window] = stats
                flat.append({"mode": mode, "scope": scope, "window": window,
                             **{k: v for k, v in stats.items() if k != "classes"},
                             "correct_micro_rate": stats["classes"]["correct"]["micro_token_rate"],
                             "incorrect_micro_rate": stats["classes"]["incorrect"]["micro_token_rate"]})
            summaries[mode][scope] = summary
    comparison = {"paired_candidates": 0, "mask_mismatches": 0,
                  "score_changed_candidates": 0, "max_absolute_score_difference": 0.0}
    if set(modes) == {"apply", "observe"}:
        for pair in pairs.values():
            if set(pair) != {"apply", "observe"}:
                raise ValueError("Unpaired apply/observe candidate")
            a, b = pair["apply"], pair["observe"]
            comparison["paired_candidates"] += 1
            comparison["mask_mismatches"] += int(a["gate_mask_segment"] != b["gate_mask_segment"])
            diff = abs(a["log_probability"] - b["log_probability"])
            comparison["score_changed_candidates"] += int(diff > 1e-6)
            comparison["max_absolute_score_difference"] = max(comparison["max_absolute_score_difference"], diff)
    result = {
        "primary": "answer_tokens / equal question weights / equal candidate weights within each class-question",
        "difference_direction": "incorrect minus correct; multiply by 100 for percentage points",
        "bootstrap_repeats": repeats, "seed": seed,
        "ci_scope": "paired question bootstrap, stratified by fold, conditional on fixed trained checkpoints",
        "limitations": ["Teacher-forced candidate scoring, not free generation.",
                        "Production residual uses the final candidate hidden state and broadcasts it to earlier positions; this is answer-conditioned analysis.",
                        "Question balancing reduces count/length weighting, but does not causally isolate correctness from wording or length."],
        "native_hook_checks": all(r["native_count_check"] for r in records),
        "mode_comparison": comparison, "modes": summaries,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(out / "summary.csv", flat)
    plot(result, out)
    report(result, out)
    for mode in modes:
        print(mode, json.dumps(summaries[mode]["pooled"]["windows"]["answer_tokens"], ensure_ascii=False))
    return result


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from plot_specs import get_figure_spec, get_figure_size
    from plot_theme import apply_style, COLORS, SAVE_SETTINGS
    spec = get_figure_spec("gate_correctness")
    apply_style(spec["style"])
    fig, ax = plt.subplots(figsize=get_figure_size("gate_correctness"))
    modes = sorted(result["modes"])
    x, width = np.arange(len(modes)), 0.32
    for i, label in enumerate(("correct", "incorrect")):
        stats = [result["modes"][m]["pooled"]["windows"]["answer_tokens"] for m in modes]
        y = np.array([s[label + "_rate"] for s in stats]) * 100
        ci = np.array([s[label + "_ci95"] for s in stats]) * 100
        ax.bar(x + (i - 0.5) * width, y, width, label=spec["text"][label],
               color=COLORS["green" if i == 0 else "red"],
               yerr=np.maximum(0, np.stack((y - ci[:, 0], ci[:, 1] - y))), capsize=4)
    ax.set_xticks(x, [spec["text"][m] for m in modes])
    ax.set_ylabel(spec["text"]["ylabel"])
    ax.set_ylim(0, 105)
    ax.legend(**spec["legend"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"gate_correctness.{extension}", dpi=SAVE_SETTINGS["png_dpi"])
    plt.close(fig)


def report(result, out):
    config_path = out / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    settings = (f"{config.get('model_name', 'Llama-3.1-8B-Instruct')}，S2S-R，"
                f"第{config.get('layer', 14)}层，κ={config.get('weight_kappa', 3)}，"
                f"margin={config.get('margin', -0.7)}。实际运行配置见 config.json。")
    lines = ["# TruthfulQA 候选答案门控触发率", "",
             settings,
             "主指标：每个答案先计算答案 token 位置触发率；每题内分别对正确/错误候选取均值，再对题目等权平均。",
             "差值为错误减正确；95% CI 使用按题配对、折内重采样的 bootstrap，固定已训练检查点。", "",
             "| 模式 | 题数 | 正确答案 | 错误答案 | 差值（百分点） | 差值95% CI（百分点） |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode, scopes in result["modes"].items():
        s = scopes["pooled"]["windows"]["answer_tokens"]
        lo, hi = np.array(s["difference_ci95"]) * 100
        lines.append(f"| {mode} | {s['questions']} | {s['correct_rate']:.2%} | {s['incorrect_rate']:.2%} | {s['incorrect_minus_correct']*100:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines.extend(["", "## 两模式核对与评分", "",
                  f"配对候选数：{result['mode_comparison']['paired_candidates']}；门控 mask 不一致：{result['mode_comparison']['mask_mismatches']}；得分发生变化的候选数：{result['mode_comparison']['score_changed_candidates']}。",
                  "", "| 模式 | MC1 | MC2 | MC3 |", "|---|---:|---:|---:|"])
    for mode, scopes in result["modes"].items():
        metrics = scopes["pooled"]["metrics"]
        lines.append(f"| {mode} | {metrics['MC1']:.2%} | {metrics['MC2']:.2%} | {metrics['MC3']:.2%} |")
    lines.extend(["", "## 每折主指标", "",
                  "| 模式 | 折 | 题数 | 正确答案 | 错误答案 | 差值（百分点） |",
                  "|---|---|---:|---:|---:|---:|"])
    for mode, scopes in result["modes"].items():
        for scope, summary in scopes.items():
            if scope == "pooled":
                continue
            s = summary["windows"]["answer_tokens"]
            lines.append(f"| {mode} | {scope} | {s['questions']} | {s['correct_rate']:.2%} | {s['incorrect_rate']:.2%} | {s['incorrect_minus_correct']*100:+.3f} |")
    lines.extend(["", "apply 表示实际干预；observe 表示计算门控但返回未修改的层输出。",
                  "单层干预之前门控已计算，因此两个模式的触发率应相同；答案得分与 MC 指标可不同。",
                  "", "## 解释范围", "",
                  "这是现有 teacher-forcing 评分流程的观测。当前方法由候选最后一个隐藏状态预测残差，并广播到整个干预区间，因此不能把结果直接解释成在线自由生成中的检测能力。",
                  "问题等权避免候选数量和总 token 数支配平均值，但无法因果排除长度、内容措辞等混杂。",
                  "", "summary.json/summary.csv 同时报告每折结果、总 token 加权触发率、候选均值和得分位置边界的敏感性分析。",
                  "questions.csv 保存每题配对统计，candidates.jsonl 保存逐候选决策、答案 token ID 与得分。", ""])
    (out / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap_repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.output_dir / "candidates.jsonl").read_text().splitlines()]
    analyze(records, args.output_dir, args.bootstrap_repeats, args.seed)
