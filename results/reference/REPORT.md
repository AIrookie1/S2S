# Archived result audit

All MC values in CSV/JSON are fractions, not percentages.
This audit reads existing files; it does not rerun models or certify the paper tables.

| Model | Method | Fold 0 MC1 (%) | Fold 1 MC1 (%) | Macro MC1 (%) | Pooled MC1 (%) |
|---|---|---:|---:|---:|---:|
| llama3.1-8B-Instruct | l2s | 37.75 | 35.45 | 36.60 | 36.60 |
| llama3.1-8B-Instruct | spherical | 54.17 | 45.72 | 49.94 | 49.94 |
| llama3.1-8B-Instruct | s2s_r | 67.40 | 46.94 | 57.17 | 57.16 |
| llama3.1-8B-Instruct | s2s_l | 49.02 | 38.88 | 43.95 | 43.94 |
| Qwen2.5-7B-Instruct | l2s | 37.50 | 34.96 | 36.23 | 36.23 |
| Qwen2.5-7B-Instruct | spherical | 48.77 | 48.41 | 48.59 | 48.59 |
| Qwen2.5-7B-Instruct | s2s_r | 66.18 | 58.92 | 62.55 | 62.55 |
| Qwen2.5-7B-Instruct | s2s_l | 46.32 | 43.77 | 45.04 | 45.04 |

## Reviewer-number provenance

- `qwen3_s2sl_table_a` (32.68%): matching_artifact_found; 1 matching record(s).
- `qwen3_s2sl_table_b` (36.68%): unresolved_no_matching_audited_artifact; 0 matching record(s).
- `qwen14_s2sl_table_a` (44.68%): matching_artifact_found; 1 matching record(s).
- `qwen14_s2sl_table_b` (48.29%): unresolved_no_matching_audited_artifact; 0 matching record(s).
- `llama_noise_s2sl` (49.02%): matching_artifact_found; 4 matching record(s).
- `q2_fold1_mean_control` (38.88%): matching_artifact_found; 3 matching record(s).
- `review_bipo` (60.18%): unresolved_no_matching_audited_artifact; 0 matching record(s).

## Limitations

- Baseline/additive original main-table files are absent from the audited paths; the main suite provides both-fold reruns.
- Historical Spherical fold labels are inverted in the aligned table, using its opposite test-half convention.
- Main-table aggregation is an arithmetic fold mean; scaling and BiPO use question-weighted pooling. Both are retained.
- The archived Llama S2S-L fold-0 value is 49.02%; its two-fold macro mean is 43.95%.
- BiPO files currently present may differ from the 60.18% review snapshot. Do not substitute one snapshot silently.
- MM safe dataset, original safety judge, and original generation records were not found; the new generation runner cannot establish the historical 79%/31% claim.
- Test-grid-selected component controls are labeled exploratory and kept separate from fixed-parameter controls.
- Artifact validation errors: 0. See evidence.json; invalid files are excluded.
