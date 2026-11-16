# Reviewer-number provenance and unresolved inputs

The `artifacts/` directory is a compact snapshot of files already present, not a
new run. Every audited metric has a source-relative filename and SHA-256; compact
question-level scores are checked against their saved aggregate when available.
`main_per_fold.csv` aligns test halves, preserving the native fold ID separately.

## Q1

- Llama S2S-R: fold 0 MC1=67.4019608%, fold 1=46.9437653%; macro=57.1728630%.
- Llama S2S-L-Static: fold 0=49.0196078%, fold 1=38.8753056%; macro=43.9474567%.
  The archived noise_0 S2S-L result is the same **fold-0** 49.0196078% score,
  with strength=3.0, t=0.2 and inference residual_noise_sigma=0. It is not a
  contradictory two-fold 49.02% result. The current target-noise runner is a
  separate, explicitly labeled training perturbation experiment.
- Qwen2.5-3B's existing BF16 scaling `summary.json` reports S2S-L=32.6805386%.
  Qwen2.5-14B's NF4 scaling `summary.json` reports 44.6756426%. These are pooled
  scores with the model-specific dev selection and current S2S-L-Static hook.
- The 36.68% and 48.29% response-table values have no matching audited record.
  They remain unresolved; there is no evidence here to label either as a typo
  or attribute them to an invented configuration. Recover the original response
  tables/run receipts before choosing a replacement.
- Original unsteered/additive main-table result files were not found in the
  audited main paths. Both-fold commands are provided; those missing scores
  are not inferred from nearby gate-closed results.

## Q2

The 38.8753056% mean + AW + Gate cell occurs in
`fold1_vavg_targeted/grid_summary.json` at kappa=2.5, margin=0.965. The unified
component suite fixes that exact pair and evaluates both folds, alongside the
six-row default kappa=3, margin=-0.7 ablation. Default and retuned controls are not
interchangeable. Other archived test-tuned settings have higher scores, so report
all selection details and avoid inferring a residual-necessity claim from one
chosen cell. The new fold-0 command has not been executed on a GPU in this cleanup.

## Q3

Teacher-forced candidate MC gains and candidate gate rates do not establish
improved free generation, safety or termination behavior. The added generation
adapter records actual token IDs, EOS and 200-token-limit outcomes, uses the same
prompt IDs for all four methods, and accepts the original safety judge's explicit
scores at threshold .5. It does not fabricate the absent MMsafe dataset, original
judge, or historical 79%/31% generation records. If the original benchmark is
multimodal, the text-only adapter rejects those inputs and a matching multimodal
loader is still required.

## BiPO

BiPO alone is included as a named baseline in the existing implementation and
new unified command. The top-level archived `bipo_truthfulqa/performance.json`
currently reports BiPO MC1=62.1787026% pooled over 817 questions; this is different
from the 60.18% value quoted during review. Several supplementary BiPO+MLP and
shared-parameter configurations coexist. Retain the exact snapshot, protocol,
selection and fold identity for every table; do not silently substitute 62.18%
for 60.18% or compare all-token BiPO to another protocol without labels.

## Validation boundary

The audit checks the available files, not all historical runs. A missing match is
not proof that a run never happened. Original model-weight revisions were not
recorded for every historical main run. New download receipts and run fingerprints
make future configurations explicit. Consult `audit_status.json` for artifact
validation failures and unresolved claims before preparing a paper table.
