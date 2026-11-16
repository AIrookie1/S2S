"""Source-workspace export adapter; deliberately omitted from the public tree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

PAIRS = (
    ("get_2d_dynamic_intervention", "get_s2s_l_intervention"),
    ("get_3d_dynamic_intervention", "get_s2s_r_intervention"),
    ("get_static_2d_residual_intervention", "get_static_s2s_l_residual_intervention"),
    ("split_like_dyn3d", "split_like_s2s_r"),
    ("run_2d_dyn", "run_S2S_L"),
    ("run_3d_dyn", "run_S2S_R"),
    ("evaluate_2d_dyn", "evaluate_S2S_L"),
    ("evaluate_3d_dyn", "evaluate_S2S_R"),
    ("evaluate_llm_judge_2d", "evaluate_llm_judge_S2S_L"),
    ("evaluate_llm_judge_3d", "evaluate_llm_judge_S2S_R"),
    ("run_llm_judge_2d", "run_llm_judge_S2S_L"),
    ("run_llm_judge_3d", "run_llm_judge_S2S_R"),
    ("run_2d_extract", "run_S2S_L_extract"),
    ("run_2d_train", "run_S2S_L_train"),
    ("run_2d_eval", "run_S2S_L_eval"),
    ("run_3d_extract", "run_S2S_R_extract"),
    ("run_3d_train", "run_S2S_R_train"),
    ("run_3d_eval", "run_S2S_R_eval"),
    ("README_2D", "README_S2S_L"),
    ("README_3D", "README_S2S_R"),
    ("evaluate_dyn_mlp", "evaluate_predictor"),
    ("train_dyn_mlp", "train_predictor"),
    ("dyn_mlp", "predictor"),
    ("dynMLP", "predictor"),
    ("DynMLP", "Predictor"),
    ("dyn3d_sga", "S2S_R"),
    ("Dyn3D_SGA", "S2S_R"),
    ("DYN3D_SGA", "S2S_R"),
    ("dyn3d", "S2S_R"),
    ("Dyn3D", "S2S_R"),
    ("DYN3D", "S2S_R"),
    ("dyn2d", "S2S_L"),
    ("Dyn2D", "S2S_L"),
    ("DYN2D", "S2S_L"),
    ("3d-dyn", "S2S_R"),
    ("2d-dyn", "S2S_L"),
    ("3d_dyn", "S2S_R"),
    ("2d_dyn", "S2S_L"),
    ("Similarity-Graph Adaptive 3D Spherical Steering", "S2S_R"),
    ("Similarity-Graph Adaptive 3D Steering", "S2S_R"),
    ("Similarity-Graph Adaptive 3D steering", "S2S_R"),
    ("Similarity-Graph Adaptive 3D", "S2S_R"),
    ("Dyn2D SLERP + predicted-norm additive steering", "S2S_L"),
    ("self.dyn_dir", "self.s2s_r_dir"),
    ("def dyn(self, fold)", "def s2s(self, fold)"),
    ("self.dyn(", "self.s2s("),
    ("DYN_FEATURE_PATH", "S2S_FEATURE_PATH"),
)


def canonical(text):
    for old, new in PAIRS:
        text = text.replace(old, new)
    # Method-specific labels and output suffixes, not mathematical 3D plotting.
    text = re.sub(r"(?i)\b2d(?=\s+(?:method|addition|additive|steering))", "S2S_L", text)
    text = re.sub(r"(?i)\b3d(?=\s+(?:method|checkpoint|dynamic|steering|SLERP|spherical|rotation|intervention|steered))", "S2S_R", text)
    text = text.replace("llm_judge_2d", "llm_judge_S2S_L").replace("llm_judge_3d", "llm_judge_S2S_R")
    text = text.replace("judge-3d", "judge-S2S_R").replace("judge-2d", "judge-S2S_L")
    return text


def _source_path_hashes(value):
    if isinstance(value, dict):
        if isinstance(value.get("source"), str) and value["source"].startswith("results/"):
            value["archive_source_path_sha256"] = hashlib.sha256(value["source"].encode()).hexdigest()
        for item in list(value.values()):
            _source_path_hashes(item)
    elif isinstance(value, list):
        for item in value:
            _source_path_hashes(item)


def normalize_tree(root):
    root = Path(root)
    for path in (root / "artifacts").glob("*.json"):
        if path.name in ("evidence.json", "number_provenance.json"):
            value = json.loads(path.read_text())
            _source_path_hashes(value)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    # A regular package prevents the renamed release from importing similarly
    # named experiment modules from a neighbouring source checkout.
    experiment_init = root / "experiments/__init__.py"
    if not experiment_init.exists():
        experiment_init.write_text('"""Experiment implementations bundled with this release."""\n')
    files = [p for p in root.rglob("*") if p.is_file()]
    destinations = {}
    for path in files:
        new_rel = canonical(str(path.relative_to(root)))
        if new_rel in destinations and destinations[new_rel] != path:
            raise ValueError("Canonical filename collision: " + new_rel)
        destinations[new_rel] = path
    transformed = 0
    for target_rel, original in destinations.items():
        text = original.read_text(encoding="utf-8")
        replacement = canonical(text)
        target = root / target_rel
        if target != original or replacement != text:
            mode = original.stat().st_mode & 0o777
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(replacement, encoding="utf-8")
            target.chmod(mode)
            if target != original:
                original.unlink()
            transformed += 1
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p:len(p.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()

    launcher = root / "test.py"
    text = launcher.read_text()
    start = text.index("    aliases = {")
    end = text.index("    target_path =", start)
    text = text[:start] + """    method_names = {"s2s_l": "S2S_L", "s2s_r": "S2S_R"}
    method = method_names.get(launcher_args.method_dir.casefold(), launcher_args.method_dir)
""" + text[end:]
    launcher.write_text(text)

    audit = root / "reproducibility/audit_results.py"
    text = audit.read_text().replace('or "S2S_L" in value', 'or "s2s_l" in value')
    text = text.replace('("S2S_R", "s2s_r")', '("s2s_r",)')
    audit.write_text(text)

    guide = root / "docs/RESULT_PROVENANCE.md"
    guide.write_text(guide.read_text() + "\n## Release naming\n\nSource paths in this release use the canonical `S2S_L`/`S2S_R` layout.\n`source_sha256` still identifies each original archived artifact before naming\nnormalization; `archive_source_path_sha256` preserves a checksum of its original\nrelative path. Evidence IDs and numerical scores are unchanged.\n")
    readme = root / "README.md"
    text = readme.read_text().replace("method/                 scientific implementations", "method/S2S_L/           S2S_L scientific implementation\nmethod/S2S_R/           S2S_R scientific implementation")
    readme.write_text(text)

    old = re.compile(r"dyn[23]d|[23]d[_-]dyn|(?:get_)[23]d_dynamic_intervention", re.I)
    for path in root.rglob("*"):
        if old.search(str(path.relative_to(root))):
            raise ValueError("Unconverted method name in a release path")
        if path.is_file() and old.search(path.read_text()):
            raise ValueError("Unconverted method name in " + str(path.relative_to(root)))
    return transformed
