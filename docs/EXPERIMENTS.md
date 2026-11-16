# Main experiments

Activate the main environment first. Commands preview their plan unless `--execute` is present. Use `--model qwen` for the Qwen main comparison; Figure 3 studies should be run for both model presets.

## Table 1: TruthfulQA

```bash
python -m reproducibility run table1 --model llama --execute --resume
python -m reproducibility run table1 --model qwen --execute --resume
python -m reproducibility run table1-generation --model llama --truth-judge-dir models/truth-judge --info-judge-dir models/info-judge --execute
python -m reproducibility run table1-generation --model qwen --truth-judge-dir models/truth-judge --info-judge-dir models/info-judge --execute
```

MC outputs: `results/reproduction/main/<model>/fold{0,1}/`. Fold and macro/pooled reports: `results/reproduction/reports/<model>/`. The model presets retain the recorded fold-specific settings, including the Qwen S2S_R margins.

## Table 2: independent task-specific training

```bash
python -m reproducibility run table2 --model llama --execute --resume
python -m reproducibility run table2 --model qwen --execute --resume
```

## Tables 3–4: components and throughput

```bash
python -m reproducibility run table3 --model llama --execute --resume
python -m reproducibility run table4 --model llama --execute
```

Run Table 1 first for the throughput inputs. Table 3 includes the mean + AW + Gate review control on both folds. Table 4 uses forced 50-token generation for timing, not free-generation termination analysis.

## Figure 2: oracle reference and vector reliability

```bash
python -m reproducibility run fig2a --model llama --execute --resume
python -m reproducibility run fig2bc --model llama --execute --resume
```

The Figure 2(a) program is the retained label-informed training-fold diagnostic; its historical curve is not certified by this release.

## Figure 3: hyperparameters and adaptive weights

```bash
for model in llama qwen; do
  python -m reproducibility run fig3a --model "$model" --execute --resume
  python -m reproducibility run fig3b-hidden --model "$model" --execute --resume
  python -m reproducibility run fig3b-fraction --model "$model" --execute --resume
  python -m reproducibility run fig3c-margin --model "$model" --execute --resume
  python -m reproducibility run fig3c-kappa --model "$model" --execute --resume
  python -m reproducibility run fig3d --model "$model" --execute
done

# Render paired-model curves after both model runs finish.
for figure in fig3a fig3b-hidden fig3b-fraction fig3c-margin fig3c-kappa; do
  python -m reproducibility plot "$figure" --execute
done
```

The fraction study retains its native fold-0 protocol; the weight-density plots use kappa 1/5/10/20. Hyperparameter maxima chosen on test remain labeled exploratory test sweeps.

## Figure 4(a,b): residual transfer

```bash
python -m reproducibility run fig4ab --model llama --execute
python -m reproducibility run fig4ab --model qwen --execute
```

## Figure 5: training-target noise

```bash
python -m reproducibility run fig5-noise --model llama --execute --resume
```

Figure 1 and the Figure 5 right-hand diagrams are schematics. Original external vector artifacts for Figure 4(c) / Figure 5 left are not bundled; see [DATA.md](DATA.md).
