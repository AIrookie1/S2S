# Protocols retained by the reorganized interface

- Main TruthfulQA is question-level two-fold evaluation: aligned fold 0 tests IDs 409–816 (408 questions); fold 1 tests IDs 0–408 (409 questions). Native Spherical prototype fold IDs use the opposite convention.
- Main learned predictors use a 0.5 predictor/geometry split within the outer training half, seed 42, hidden width 256, 2000 epochs, batch 256, and learning rate 0.001. Main model presets retain their recorded fold-specific inference parameters.
- Main tables use an arithmetic mean of unrounded fold scores; question-weighted pooling is reported separately. Baseline, additive, L2S, Spherical, S2S_L, and S2S_R MC commands cover both folds.
- Table 2 trains on each benchmark's own training data. TruthfulQA-to-generic transfer is a separate supplementary experiment. Per-task settings, including Qwen COPA's 400 epochs, remain in native task configurations.
- Scaling/BiPO use their declared development sets and selection rules. Test-grid optima in layer and hyperparameter studies remain exploratory test-sweep results.
- Table 3 includes the fixed kappa=2.5, margin=0.965 mean-only review control on both folds; it is not replaced by another grid cell.
- Training-target corruption and inference residual-direction noise have independent CLI IDs and outputs. The historical inference-noise response table uses fold 0; it is not the two-fold main mean.
- Candidate teacher-forced MC scoring and free generation have different probe information. Candidate gate rates and MC gains do not establish free-generation safety or termination improvements. Generation records actual EOS/length outcomes.
- Archived results in `results/reference/` remain unchanged. Missing records, original external baselines, and original MMsafe materials are not fabricated.
