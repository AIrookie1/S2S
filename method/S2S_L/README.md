# S2S_L

S2S_L shares the existing direct-vector MLP training pipeline with S2S_R and
changes only the evaluation-time steering rule.

For the last hidden state in the intervention segment:

```text
v_hat = MLP(normalize(h_last))
V = normalize(v_avg)
R = normalize(v_hat - v_avg)
D = SLERP(V, R, slerp_t)
D_scaled = D * ||v_hat||_2
h_new = h + strength * D_scaled
```

The same `D_scaled` is broadcast over the intervention segment, matching the
existing S2S_R segment semantics. `slerp_t=0` gives the mean direction and
`slerp_t=1` gives the predicted residual direction.

TruthfulQA pipeline example:

```bash
bash script/S2S_L/LLAMA/run_S2S_L.sh
```

The direct-MLP checkpoint format is intentionally compatible with S2S_R.
