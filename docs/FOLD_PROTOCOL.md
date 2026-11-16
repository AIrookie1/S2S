# Fold identity, aggregation, and paper wording

## Main TruthfulQA protocol

There are 817 questions indexed by the Hugging Face multiple-choice validation
ordering. Sorted IDs 0..408 form the first half (409 questions); IDs 409..816 form
the second half (408 questions). Splits are at question level, never answer level.

| Aligned fold | S2S/L2S native fold | Spherical native fold | Outer training IDs | Held-out test IDs | Test n |
|---|---:|---:|---|---|---:|
| 0 | 0 | 1 | 0..408 | 409..816 | 408 |
| 1 | 1 | 0 | 409..816 | 0..408 | 409 |

In S2S/L2S the selected outer training half is shuffled by NumPy default_rng(seed=42).
Its within-fold fraction of 0.5 trains the predictor; the complementary question
subset estimates geometry/mean quantities. This historical `val_fraction` is a
predictor-training fraction, **not a held-out hyperparameter-development set**.
The original main configuration is fixed. The historical Spherical baseline
computes its class-centroid prototypes on its outer training half, using KFold
without shuffling. Native fold numbers alone cannot establish matched test sets.

The main-paper values use an unweighted fold mean, computed from unrounded scores:
`macro = (fold0 + fold1) / 2`. Question-weighted pooling is also supplied:
`pooled = (408*fold0 + 409*fold1) / 817`. These can differ at the second decimal.
For archived Llama S2S-R, macro MC1=57.172863% (57.17%) and pooled MC1=57.160343%
(57.16%). The fold gap is preserved and should be visible for every baseline.

## Different protocols must remain separate

- Scaling reserves 20% of each outer training half for dev, and then splits the
  remaining training questions into geometry and predictor subsets. Checkpoints
  and inference settings are selected without that fold's test questions.
- BiPO reserves 20% for dev, trains the shared vector on optimization questions,
  selects by bidirectional dev loss, then selects strengths/calibration settings
  by dev MC1. Its all-token intervention and training objective are distinct.
- Shared-dev or fold-0-selected settings reused on both folds are supplementary
  common-setting comparisons. A dev question in one outer fold can be a test
  question in the opposite fold; pooled selection must not be presented as a new
  independent nested test result.
- Component grids that select the best score on test are exploratory test sweeps.
  Keep them separate from the fixed Q2 setting and label the selection rule.
- Core legacy evaluators retain their existing scoring code, including the
  shared scorer's `lp[3:]` convention. Scaling/BiPO sum completion-token log
  probabilities under their own documented prompt and token-boundary protocols.
  New protocols must not silently overwrite a table using a different scorer.

## Suggested manuscript insertion

> We use question-level two-fold evaluation on TruthfulQA's 817 questions.
> The sorted Hugging Face question IDs are split into halves of 409 and 408
> questions. In fold 0 the first half supplies training and the second half is
> held out; fold 1 swaps these roles. For the learned S2S/L2S predictors, the
> selected training half is further partitioned into predictor and geometry
> subsets using a 0.5 fraction and seed 42. No held-out test question contributes
> to either subset. Table 1 reports the arithmetic mean of the two unrounded
> fold scores. We also report each fold separately and distinguish this mean
> from question-weighted pooling. Baseline folds are aligned by held-out
> question identities because the native Spherical Steering fold labels use
> the opposite convention.

This is suggested text for the main fixed-setting protocol. Describe scaling,
BiPO and exploratory grid-selection protocols separately. No manuscript file
was present to edit in this source repository.
