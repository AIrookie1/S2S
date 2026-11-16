# Validation record: 2026-11-15

Checks performed while preparing the anonymous source release:

| Check | Result |
|---|---|
| Main CPU regression suite, including workflow and canonical layout tests | 59 passed |
| Separate scaling-environment CPU regression suite | 9 passed |
| Combined distinct CPU tests | 68 passed |
| Generated Python command argument sets parsed by their native parsers | 55 accepted |
| Native generic shell dry runs, Llama | 20 passed |
| Native generic shell dry runs, Qwen | 20 passed |
| Archived numeric evidence | 666 records; zero aggregate-validation errors |
| Temporary anonymous export scan and SHA-256 verification | Passed |
| Exported Python AST and shell syntax checks | Passed |

The main environment used Python 3.10, torch 2.9.0, Transformers 4.57.6;
the scaling environment used Python 3.10, torch 2.9.0, Transformers 5.20.2.
Exact package pins are in the two root requirements files. A clean dependency
installation on every CUDA platform has not been tested.

No full GPU experiment or historical MMsafe replication was performed during this
code organization. CPU tests, parser checks, and shell dry runs do not establish
final benchmark scores. Missing historical records remain marked unresolved in
`artifacts/audit_status.json`; original model snapshots and the original MMsafe
judge/data are prerequisites for exact historical reproduction.

The final export has its own `release_manifest.json` and `privacy_report.json`.
Check its actual contents again before publication:

```bash
python -m reproducibility.verify_release .
python -m unittest discover -s reproducibility/tests -v
```

The scan excludes original Git history, raw binary checkpoints/features, media
metadata, editor state and runtime logs. It also checks text for local personal
paths, common credential patterns, email addresses and private deny-list terms.
It cannot guarantee anonymity against account metadata or code recognition.

## Canonical package naming

The public source tree uses `S2S_L` and `S2S_R` for method packages, launchers,
hyperparameters, output names, imports, APIs and documentation. The launcher
resolves those canonical names without lowercasing case-sensitive paths.
A regular experiment package isolates imports from neighbouring source trees.
Archived evidence IDs and numeric metrics are preserved while their displayed
source paths use the release naming convention.
