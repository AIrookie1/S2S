# Validation

Main and scaling environments are tested separately:

```bash
python -m pytest -q
.venv-scaling/bin/python -m pytest -q tests/rebuttal_R1W2_model_scaling
```

Lightweight source/CLI checks can run without model dependencies:

```bash
python -m unittest discover -s tests/reproducibility -v
```

`reports/validation.json` records checks for this snapshot; `reports/release_manifest.json` records release-source hashes. Verify a working checkout with `--allow-git`; use `--strict` on a clean anonymous export. Full model benchmarks and original MMsafe replication are not certified by these checks.
