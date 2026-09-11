# router_topk_ablation

Sweep requested router Top-K over 1, 5, 10, 20, 40, 80. Effective K is recorded because K is clamped to the energy-selected SVD rank.

Run: `bash experiments/router_topk_ablation/run.sh`
