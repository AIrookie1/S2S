# Third-party notices

- `truthfulqa/{models,utilities,configs,presets}.py` and
  `TruthfulQA/data/v1/TruthfulQA.csv` are the local TruthfulQA evaluation dependency
  and dataset used by this project. Upstream:
  https://github.com/sylinrl/TruthfulQA ; the local checkout identifies commit
  `23b7eace240507542b766128290700fd961432cd`. The local evaluation files may contain
  adaptations; this is not a claim of byte-identical upstream code. The upstream
  Apache-2.0 license is retained as `TruthfulQA/LICENSE`.
- `experiments/rebuttal_R2Q4_bipo/vendor/` preserves upstream reference files and the
  MIT license from https://github.com/CaoYuanpu/BiPO at commit
  `cef1d00ab108d0e265578589e6b344b200ea3db8`. The adapted active runner is documented
  separately. Upstream copyright notices remain intact.
- `baukit` is an installed dependency from https://github.com/davidbau/baukit ;
  its source revision is pinned in the root requirements files. Model weights and
  third-party benchmark datasets retain their respective providers' terms and
  are not repackaged as project-owned assets.

This release does not assign a new license to original project code or relicense
third-party material. Removing local personal paths does not remove upstream
attribution obligations. Original scientific implementations are preserved;
release tooling, configuration orchestration, and privacy filtering are new.
