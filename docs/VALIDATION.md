# Validation

Run `python -m unittest discover -s reproducibility/tests -v` for lightweight checks.
The main and scaling environments have separate CPU regression suites.
`validation_report.json` records the checks performed for this snapshot; a fresh
export starts as unverified. CLI/parser checks do not establish benchmark accuracy.
No full GPU rerun or historical MMsafe replication is claimed.
