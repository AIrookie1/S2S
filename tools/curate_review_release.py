"""Source-only adapter for a compact paper/rebuttal review release."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

GROUPS = {
    "steering_vector_ablation": "fig2a_oracle_vectors",
    "l2s_reliability": "fig2bc_l2s_reliability",
    "S2S_L_ablation": "fig3a_s2s_l_hyperparameters",
    "mlp_hidden_ablation": "fig3b_hidden_size",
    "val_fraction_ablation": "fig3b_predictor_fraction",
    "margin_ablation": "fig3c_gate_margin",
    "weight_kappa_ablation": "fig3c_weight_sharpness",
    "kappa_view": "fig3d_adaptive_weights",
    "residual_analysis": "fig4ab_residual_transfer",
    "component_ablation": "table3_component_ablation",
    "generation_efficiency": "table4_generation_efficiency",
    "l2s_noise_robustness": "fig5_target_noise",
    "model_scaling_ablation": "rebuttal_R1W2_model_scaling",
    "gate_correctness": "rebuttal_R3Q3_candidate_gates",
    "margin_gate_correctness": "rebuttal_R3Q3_margin_gates",
    "layer_performance": "rebuttal_R2W5_layer_sweep",
    "bipo_truthfulqa": "rebuttal_R2Q4_bipo",
    "mlp_prediction_error": "rebuttal_prediction_error",
    "truthfulqa_generic_transfer": "rebuttal_R2Q2_cross_task_transfer",
    "geometry_decision_alignment": "rebuttal_geometry_diagnostics",
}
INDEX = [
 ("Table 1", "table1_truthfulqa", "MC: reproducibility.run main; generation: table1_run_generation.py", "two_folds; baseline_additive_L2S_spherical_S2S_L_S2S_R_MC; generation_requires_judges"),
 ("Table 2", "table2_generic_tasks", "reproducibility.run generic", "per_task_training; official_task_evaluation; native_task_epochs"),
 ("Table 3 / R3 Q2", "table3_component_ablation", "reproducibility.run component", "both_folds; default_control_and_kappa2.5_margin0.965"),
 ("Table 4", "table4_generation_efficiency", "run.sh", "forced_50_final_tokens; throughput_not_EOS_behavior"),
 ("Figure 2(a)", "fig2a_oracle_vectors", "evaluate_reference.py", "label_informed_train_fold_reference; historical_curve_not_certified"),
 ("Figure 2(b,c)", "fig2bc_l2s_reliability", "evaluate.py; plot.py", "matched_L2S_training_questions; raw_pair_diagnostics"),
 ("Figure 3(a)", "fig3a_s2s_l_hyperparameters", "run.sh; plot.sh", "test_grid_sweep"),
 ("Figure 3(b)", "fig3b_hidden_size", "run.sh; plot.sh", "width16_to2048; both_folds"),
 ("Figure 3(b)", "fig3b_predictor_fraction", "run.sh; plot.sh", "fractions0.1_to0.9; seeds42_123_0; native_fold0"),
 ("Figure 3(c)", "fig3c_gate_margin", "run.sh; plot.sh", "unique_margins_include_default_and_closed_gate"),
 ("Figure 3(c)", "fig3c_weight_sharpness", "run.sh; plot.sh", "kappa0.5_1_2_4_8_16_32"),
 ("Figure 3(d)", "fig3d_adaptive_weights", "plot.sh", "kappa1_5_10_20; token_weight_density"),
 ("Figure 4(a,b)", "fig4ab_residual_transfer", "run_paper.py", "both_folds; oracle_conditions_are_diagnostics"),
 ("Figure 4(c) / Figure 5 left", "external_vector_baselines", "not_bundled", "original_LoFiT_LoReFT_implementations_and_vector_artifacts_required"),
 ("Figure 5 middle", "fig5_target_noise", "reproducibility.run noise", "training_target_noise; inference_noise_zero"),
 ("R1 W1 inference noise", "rebuttal_R1W1_inference_noise", "run.py", "clean_predictor; inference_direction_noise; historical_table_fold0"),
 ("R1 W2 / R2 W6", "rebuttal_R1W2_model_scaling", "run.py; retune_spherical.py; compare_slerp.py", "BF16_except_14B_NF4; per_fold_dev_selection"),
 ("R2 W5", "rebuttal_R2W5_layer_sweep", "run_llama.sh", "test_set_sweep_best; not_dev_selected"),
 ("R2 Q4 / BiPO baseline", "rebuttal_R2Q4_bipo", "run.py; run_mlp_s2s.py", "BiPO_alone_and_calibrated_variants; distinct_protocols"),
 ("R3 Q3", "rebuttal_R3Q3_candidate_gates", "reproducibility.run gate", "teacher_forced_candidate_gates"),
 ("R3 Q3 margin", "rebuttal_R3Q3_margin_gates", "reproducibility.run margin", "MC_and_label_conditioned_gate_rates"),
 ("R2 MMsafe / R3 generation", "rebuttal_mmsafe", "reproducibility.run generation; reproducibility.safety", "original_prompt_data_and_safety_judge_required; text_adapter_only"),
 ("Supplement prediction errors", "rebuttal_prediction_error", "run_official.py; run_reliability.py", "requires_generic_training_features_and_TruthfulQA_transfer_checkpoints"),
 ("R2 Q2 supplement", "rebuttal_R2Q2_cross_task_transfer", "run.sh", "TruthfulQA_to_generic_transfer; distinct_from_Table2"),
 ("Supplement geometry", "rebuttal_geometry_diagnostics", "run.py", "sampled_or_full_diagnostic; not_a_new_paper_number"),
]


def remap(text):
    for old,new in GROUPS.items():
        text=re.sub(r"(?<![A-Za-z0-9_])"+re.escape(old)+r"(?![A-Za-z0-9_])",lambda _:new,text)
    return text


def curate_tree(root):
    root=Path(root)
    removed=[]
    def remove(relative):
        p=root/relative
        if p.exists():
            removed.append(relative)
            if p.is_dir():shutil.rmtree(p)
            else:p.unlink()
    # Preserve the noise evaluator used by an actual paper experiment before
    # removing its old comparison-tool directory.
    support=root/'experiments/truthfulqa/evaluate_l2s_linear_vs_l2s_slerp.py'
    noise=root/'experiments/l2s_noise_robustness/evaluate_l2s_baselines.py'
    if support.exists():shutil.copy2(support,noise)
    for p in ('experiments/cosine_similarity','experiments/word_number','experiments/truthfulqa',
              'tools/validate_S2S_L_setup.py','tools/validate_S2S_R_setup.py'):
        remove(p)
    # The old endpoint-only GT rerun cannot change the earlier answer-token
    # likelihoods; keep the explicit label-informed vector reference instead.
    for p in ('experiments/steering_vector_ablation/evaluate.py',
              'experiments/steering_vector_ablation/run.sh',
              'experiments/steering_vector_ablation/plot.py',
              'experiments/steering_vector_ablation/plot.sh'):
        remove(p)
    reference=root/'experiments/steering_vector_ablation/evaluate_train_vectors_reference.py'
    if reference.exists():reference.rename(reference.with_name('evaluate_reference.py'))
    # Remove internal draft/configuration prose from the reviewer-facing copy.
    for p in list((root/'docs').glob('*')):
        if p.is_file() and p.name!='VALIDATION.md':remove(str(p.relative_to(root)))
    # Fix the missing hidden-width configuration, without touching method code.
    hidden=root/'hyperparams/S2S_R/experiments/mlp_hidden_ablation.sh'
    hidden.write_text('#!/usr/bin/env bash\nread -r -a MLP_HIDDEN_VALUES <<< "${HIDDEN_SIZES:-16 32 64 128 256 512 1024 2048}"\n')
    margin=root/'hyperparams/S2S_R/experiments/margin_ablation.sh'
    margin.write_text('#!/usr/bin/env bash\nMARGINS=(-0.8 -0.7 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1.0)\n')
    # Repair imports and split-field compatibility in the Figure 2 entry point.
    reliability=root/'experiments/l2s_reliability/evaluate.py'
    text=reliability.read_text().replace('from method.l2s.model import','from method.l2s_linear.model import')
    text=text.replace('from method.l2s.spherical_steering import get_l2s_spherical_intervention',
                      'from .spherical_adapter import get_l2s_spherical_intervention' if False else
                      'from experiments.l2s_reliability.spherical_adapter import get_l2s_spherical_intervention')
    text=text.replace('from method.l2s.steering import','from method.l2s_linear.steering import')
    text=text.replace('    required = {"l2s_train_q_indices", "test_q_indices"}',
                      '    train_key = "predictor_q_indices" if "predictor_q_indices" in checkpoint.files else "l2s_train_q_indices"\n    required = {train_key, "test_q_indices"}')
    text=text.replace('np.asarray(checkpoint["l2s_train_q_indices"], dtype=np.int64)',
                      'np.asarray(checkpoint[train_key], dtype=np.int64)')
    reliability.write_text(text)
    # A raw base feature can be used directly for the explicit train-fold oracle
    # reference; the previous version required an absent materialized fold NPZ.
    if (reference.parent/'evaluate_reference.py').exists():
        p=reference.parent/'evaluate_reference.py';text=p.read_text()
        text=text.replace('    parser.add_argument("--feature_path", required=True)',
          '    parser.add_argument("--feature_path", required=True)\n    parser.add_argument("--fold_idx", type=int, choices=[0,1], default=0)')
        text=text.replace('load_pair_vector_data(args.feature_path)', 'load_pair_vector_data(args.feature_path, fold_idx=args.fold_idx)')
        text=text.replace('Fig. 1 MC1:', 'Figure 2(a) oracle diagnostic:')
        p.write_text(text)
    # Keep only one precise copy of the L2S-noise baseline evaluator.
    p=root/'experiments/l2s_noise_robustness/run.sh';text=p.read_text()
    text=text.replace('experiments/truthfulqa/evaluate_l2s_linear_vs_l2s_slerp.py',
                      'experiments/l2s_noise_robustness/evaluate_l2s_baselines.py')
    text=text.replace('${S2S_FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_S2S_R_pairs.npz}',
                      '${S2S_FEATURE_PATH:-$FEATURE_PATH}')
    p.write_text(text)
    # Make optional diagnostics honor the same model directory as the commands.
    for p in (root/'experiments').rglob('*.py'):
        text=p.read_text()
        if 'default="models/Meta-Llama-3.1-8B-Instruct"' in text:
            text=text.replace('default="models/Meta-Llama-3.1-8B-Instruct"',
                              'default=__import__("os").environ.get("LLAMA_MODEL_DIR", "models/Meta-Llama-3.1-8B-Instruct")')
            p.write_text(text)
    # Copy new thin entry points and adapter tests; they contain no source paths
    # or raw rebuttal drafts.
    assets=Path(__file__).with_name('review_release_assets')
    for p in assets.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        dest=root/p.relative_to(assets);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    # Rename every retained experiment group and matching configuration file.
    files=[p for p in root.rglob('*') if p.is_file()]
    targets={}
    for p in files:
        rel=remap(str(p.relative_to(root)))
        if rel in targets:raise ValueError('Experiment filename collision: '+rel)
        targets[rel]=p
    for rel,p in targets.items():
        q=root/rel;q.parent.mkdir(parents=True,exist_ok=True)
        text=remap(p.read_text())
        if q!=p or text!=p.read_text():
            mode=p.stat().st_mode & 0o777;q.write_text(text);q.chmod(mode)
            if q!=p:p.unlink()
    for p in sorted([p for p in root.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        if not any(p.iterdir()):p.rmdir()
    # Optional diagnostics consume the default outputs prepared by the commands
    # above, rather than an unrelated private cache layout.
    for folder in ('rebuttal_prediction_error','rebuttal_geometry_diagnostics'):
        for file in (root/'experiments'/folder).glob('*.py'):
            text=file.read_text()
            text=text.replace('features/generic/S2S_R/', 'results/reproduction/generic/features/S2S_R/')
            text=text.replace('checkpoints/generic/S2S_R/', 'results/reproduction/generic/checkpoints/S2S_R/')
            text=text.replace('prototypes/generic/spherical_steering/', 'results/reproduction/generic/prototypes/spherical_steering/')
            text=text.replace('checkpoints/S2S_R/truthfulqa/', 'results/reproduction/checkpoints/S2S_R/truthfulqa/')
            text=text.replace('prototypes/spherical_steering/truthfulqa/', 'results/reproduction/prototypes/{MODEL}/')
            text=text.replace('f"features/{MODEL}', 'f"results/reproduction/features/{MODEL}')
            file.write_text(text)
    # Bring replay settings into agreement with the supplied configuration note.
    runner=root/'reproducibility/run.py';text=runner.read_text()
    text=text.replace('SCALING = ("qwen05", "qwen15", "qwen3", "qwen14")',
      'SCALING = ("qwen05", "qwen15", "qwen3", "qwen14", "qwen35", "nemotron_nano8b", "qwen32")')
    text=text.replace('default=list(SCALING))','default=list(SCALING[:4]))')
    text=text.replace('"--eval_split", "test", "--alpha", 1.0,',
                      '"--eval_split", "test", "--alpha", (1.4 if self.args.model == "qwen" else 1.0),')
    text=text.replace('"--checkpoint_path", self.s2s(f), "--weight_kappa", 3, "--margin", -0.7,',
                      '"--checkpoint_path", self.s2s(f), "--weight_kappa", 3, "--margin", (-0.3 if self.args.model == "qwen" and f == 0 else -0.7),')
    # The current task-specific generic config carries 400 epochs for Qwen COPA.
    text=text.replace('EPOCHS=str(self.args.epochs), RUN_EXTRACT="1",',
                      'EPOCHS=str(400 if self.args.model == "qwen" and dataset == "copa" and self.args.epochs == 2000 else self.args.epochs), RUN_EXTRACT="1",')
    runner.write_text(text)
    generation=root/'reproducibility/generation.py';text=generation.read_text()
    text=text.replace('    p.add_argument("--prompts", type=Path, required=True)',
        '    p.add_argument("--spherical-prototype", type=Path)\n    p.add_argument("--prompts", type=Path, required=True)')
    text=text.replace('"l2s_checkpoint", "output_dir")', '"l2s_checkpoint", "spherical_prototype", "output_dir")')
    text=text.replace('methods=["baseline", "l2s", "s2s_l", "s2s_r"],',
        'methods=["baseline", "l2s", "s2s_l", "s2s_r"] + (["spherical"] if args.spherical_prototype else []),\n                  spherical_sha256=fingerprint(args.spherical_prototype) if args.spherical_prototype else None,')
    needle='    eos = model.generation_config.eos_token_id'
    extra="""    if args.spherical_prototype:
        from method.spherical_steering.steering import get_spherical_intervention
        with np.load(args.spherical_prototype, allow_pickle=False) as proto:
            if set(int(q) for q in proto['test_q_indices']) != set(splits['s2s']['test_q_indices']):
                raise ValueError('Spherical and S2S checkpoints have different test question IDs')
            mu_t=torch.tensor(proto['mu_T'],device=args.device,dtype=torch.float32)
            mu_h=torch.tensor(proto['mu_H'],device=args.device,dtype=torch.float32)
        alpha,beta=(.6,.4) if 'qwen' in args.model_name.lower() else (.7,-.15)
        hooks['spherical']=get_spherical_intervention(mu_t,mu_h,kappa=20.,alpha=alpha,beta=beta)
"""
    if needle not in text:raise ValueError('Generation insertion point missing')
    text=text.replace(needle,extra+needle)
    text=text.replace("return hook(output, start_idx=h.shape[1] - 1)", "return hook(output, layer_name=None, start_idx=h.shape[1] - 1)")
    generation.write_text(text)
    text=runner.read_text()
    text=text.replace('"--s2s-checkpoint", self.s2s(f), "--l2s-checkpoint", self.l2s(f),',
        '"--s2s-checkpoint", self.s2s(f), "--l2s-checkpoint", self.l2s(f),\n                "--spherical-prototype", self.prototype(f),')
    text=text.replace('"--dtype", self.args.dtype, "--device", self.args.device,\n                "--max-new-tokens", 200',
        '"--dtype", self.args.dtype, "--device", self.args.device,\n                "--l2s-alpha", (1.4 if self.args.model == "qwen" else 1.0),\n                "--margin", (-0.3 if self.args.model == "qwen" and f == 0 else -0.7),\n                "--max-new-tokens", 200')
    runner.write_text(text)
    # Use the exact native spherical transform for the Figure 2 adapter.
    # Assets are already copied above; no method implementation is replaced.
    # Compact the archived audit without changing evidence IDs or numeric scores.
    p=root/'artifacts/evidence.json';evidence=json.loads(p.read_text())
    for record in evidence['records']:
        record.pop('question_metrics',None)
    p.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    # A compact registry separates manuscript IDs from unnumbered rebuttal work.
    with (root/'EXPERIMENT_INDEX.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['paper_or_rebuttal_id','directory','entrypoint','protocol_or_missing_input'])
        writer.writerows(INDEX)
    (root/'experiments/README.md').write_text('# Experiment entry points\n\nSee the commands in the root README and `EXPERIMENT_INDEX.csv`.\nFolder prefixes identify manuscript figures/tables or unnumbered rebuttal studies.\n')
    # Remove lengthy per-study local execution diaries while retaining the source
    # formulas, comments, upstream licenses, test code, and the index.
    for p in (root/'experiments').rglob('README.md'):
        if p.parent==root/'experiments':continue
        group=p.parent.name
        p.write_text('# '+group+'\n\nRun commands: see the root README and EXPERIMENT_INDEX.csv.\n')
    (root/'artifacts/README.md').write_text('# Archived evidence\n\nCSV/JSON MC scores are fractions in [0,1]. `main_per_fold.csv` aligns test halves;\n`main_averages.csv` provides macro and pooled values. Evidence IDs, recorded scores,\nsource hashes and original-path hashes are preserved; displayed paths use release\nnames. Repeated per-question copies were omitted from this compact package.\n\nNo full GPU rerun is claimed. Missing source records remain marked in\n`audit_status.json`; original MMsafe inputs/judge and external vector baselines\nare not bundled.\n')
    (root/'docs/RELEASE_CONTENTS.md').write_text('# Review release contents\n\nPaper mapping: `EXPERIMENT_INDEX.csv`. Commands: root `README.md`.\n\nRemoved from the public copy: obsolete basis-routing/cosine scripts, old word-count\nplots, the final-token-only GT rerun, duplicate comparison entry points, long local\nexecution diaries, raw response/strategy drafts, and redundant setup validators.\nRequired noise-evaluation helpers, upstream licenses, benchmark data, tests and\ncompact archived evidence remain. Original workspace files are not deleted.\n\nFigure 1 and the right-hand Figure 5 diagrams are schematics. Figure 4(c) and the\nfull left-hand Figure 5 baseline comparison require original external LoFiT/LoReFT\nimplementations/vector artifacts. They are recorded as unavailable rather than\nassigned a fictitious runnable experiment. The Figure 2(a) entry is the retained\nlabel-informed train-fold diagnostic, not a certified historical curve replay.\n')
    # Publishing-only instructions belong in the source workspace, not a rebuttal
    # link intended for reviewers to run commands.
    (root/'README.md').write_text((assets/'README.md').read_text())
    (root/'cleanup_manifest.json').write_text(json.dumps(dict(removed_public_paths=removed,
        preserved_original_workspace=True,retained_evidence_records=len(evidence['records']),
        paper_pages=19,raw_rebuttal_drafts_included=False),indent=2)+'\n')
    # Prefix the executable filenames as well as their containing folders.
    prefixes={
      'fig2a_oracle_vectors':'fig2a','fig2bc_l2s_reliability':'fig2bc',
      'fig3a_s2s_l_hyperparameters':'fig3a','fig3b_hidden_size':'fig3b_hidden',
      'fig3b_predictor_fraction':'fig3b_fraction','fig3c_gate_margin':'fig3c_margin',
      'fig3c_weight_sharpness':'fig3c_kappa','fig3d_adaptive_weights':'fig3d',
      'fig4ab_residual_transfer':'fig4ab','fig5_target_noise':'fig5',
      'table3_component_ablation':'table3','table4_generation_efficiency':'table4',
      'rebuttal_R1W1_inference_noise':'rebuttal_R1W1',
      'rebuttal_R1W2_model_scaling':'rebuttal_R1W2',
      'rebuttal_R2Q4_bipo':'rebuttal_R2Q4','rebuttal_R2W5_layer_sweep':'rebuttal_R2W5',
      'rebuttal_R3Q3_candidate_gates':'rebuttal_R3Q3_candidate',
      'rebuttal_R3Q3_margin_gates':'rebuttal_R3Q3_margin',
      'rebuttal_prediction_error':'rebuttal_prediction_error',
      'rebuttal_R2Q2_cross_task_transfer':'rebuttal_R2Q2',
      'rebuttal_geometry_diagnostics':'rebuttal_geometry',
    }
    renames={};local_names={}
    for group,tag in prefixes.items():
        directory=root/'experiments'/group
        local_names[group]={}
        for old in ('run_paper.py','run.py','run.sh','plot.py','plot.sh'):
            path=directory/old
            if not path.exists():continue
            kind='plot' if old.startswith('plot') else 'run'
            new=tag+'_'+kind+path.suffix
            target=directory/new
            if target.exists():raise ValueError('Executable filename collision: '+str(target))
            renames[str(path.relative_to(root))]=str(target.relative_to(root))
            local_names[group][old]=new
    for path in [p for p in root.rglob('*') if p.is_file()]:
        text=path.read_text()
        for old,new in sorted(renames.items(),key=lambda pair:-len(pair[0])):
            text=text.replace(old,new)
            if old.endswith('.py'):
                text=re.sub(re.escape(old[:-3].replace('/','.'))+r'(?![A-Za-z0-9_])',lambda _:new[:-3].replace('/','.'),text)
        rel=path.relative_to(root)
        if len(rel.parts)>2 and rel.parts[0]=='experiments' and rel.parts[1] in local_names:
            for old,new in local_names[rel.parts[1]].items():
                text=text.replace('"'+old+'"','"'+new+'"').replace("'"+old+"'","'"+new+"'")
                for variable in ('$HERE','${HERE}','$SCRIPT_DIR','${SCRIPT_DIR}'):
                    text=re.sub(re.escape(variable+'/'+old)+r'(?![A-Za-z0-9_])',lambda _:variable+'/'+new,text)
        if text!=path.read_text():path.write_text(text)
    for old,new in renames.items():(root/old).rename(root/new)
    # The compact registry contains local entry names, not full import paths.
    with (root/'EXPERIMENT_INDEX.csv').open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for old,new in local_names.get(row['directory'],{}).items():
            row['entrypoint']=re.sub(r'(?<![A-Za-z0-9_])'+re.escape(old)+r'(?![A-Za-z0-9_])',new,row['entrypoint'])
    with (root/'EXPERIMENT_INDEX.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    component=root/'experiments/table3_component_ablation/table3_run.sh'
    text=component.read_text().replace('FOLD_IDX=1','FOLD_IDX="${FOLD_IDX:-0}"').replace('--expected_fold 1','--expected_fold "$FOLD_IDX"')
    component.write_text(text)
    (root/'docs/VALIDATION.md').write_text('# Validation\n\nRun `python -m unittest discover -s reproducibility/tests -v` for lightweight checks.\nThe main and scaling environments have separate CPU regression suites.\n`validation_report.json` records the checks performed for this snapshot; a fresh\nexport starts as unverified. CLI/parser checks do not establish benchmark accuracy.\nNo full GPU rerun or historical MMsafe replication is claimed.\n')
    layout_test=root/'reproducibility/tests/test_release_layout.py'
    text=layout_test.read_text().replace('(".git", "__pycache__", ".pytest_cache")',
      '(".git", "__pycache__", ".pytest_cache", "models", "results", "features", "checkpoints", "prototypes", ".venv", ".venv-scaling")')
    layout_test.write_text(text)
    (root/'validation_report.json').write_text(json.dumps(dict(status='not_run_by_exporter',full_gpu_experiments=False),indent=2)+'\n')
    return dict(experiment_groups=len(GROUPS),removed_items=len(removed))
