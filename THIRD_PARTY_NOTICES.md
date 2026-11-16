# Third-party notices

- The local `truthfulqa/` helpers and `data/truthfulqa/TruthfulQA.csv` originate from the TruthfulQA evaluation dependency used by this project: https://github.com/sylinrl/TruthfulQA . Local adaptations may be present; an exact upstream Git revision was not established. The preserved Apache-2.0 license is `third_party/licenses/TruthfulQA-LICENSE`; the data snapshot checksum is recorded in `data/truthfulqa/README.md`.
- `experiments/rebuttal_R2Q4_bipo/vendor/` retains upstream BiPO reference files and the MIT license from https://github.com/CaoYuanpu/BiPO at commit `cef1d00ab108d0e265578589e6b344b200ea3db8`. The active adapted runner is separate from the vendored trainer.
- `baukit` is installed from https://github.com/davidbau/baukit at the revision pinned in `requirements/main.txt` and `requirements/scaling.txt`.

This organization update does not assign a new license to original project code or relicense third-party material. Upstream copyright and attribution notices are retained. Model weights and external datasets keep their providers' terms.
