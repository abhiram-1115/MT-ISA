# Agreement diagnostic finding

## Local diagnostic

The paired files contain 1,030 shared rows. The original agreement diagnostic
used lowercase Unicode word tokens only. Five Qwen rows contain mixed
non-English output in the aspect or opinion fields. The diagnostic now strips
non-ASCII characters, lowercases, and applies conservative suffix
normalization before token F1 and target matching. Auxiliary JSON files are
unchanged; only diagnostic values and agreement-derived weights change.

| Diagnostic | Original | Normalized |
| --- | ---: | ---: |
| Agreement mean | 0.07407 | 0.07664 |
| Mistral target-match rate | 0.72816 | 0.73301 |
| Qwen target-match rate | 0.04563 | 0.04951 |
| High-agreement target-match rate | 0.38314 | 0.38738 |
| Low-agreement target-match rate | 0.39075 | 0.39515 |
| Agreement/target-match correlation | -0.14869 | -0.13770 |

The polarity field is absent in these paired files, so polarity correctness
correlation is unavailable. The normalization does not materially change the
diagnostic conclusion: agreement is not tracking target-match quality here.

## PowerShell runs

Run from the repository root after activating `venv`.

Regression against the prior `out_stored_lr1e-4_s42` baseline (`test macro-F1
0.6426`):

```powershell
python train_mt_isa.py --train-data data/processed/restaurant14_train_implicit.json --aux-data data/auxiliary/restaurant14_train_implicit_aux_mistral7b_full.json --test-data data/processed/restaurant14_test_implicit.json --model-name google/flan-t5-base --d-awl-strategy output --lr 1e-4 --grad-accum 1 --batch-size 32 --num-epochs 20 --seed 42 --weight-source confidence --output-dir models_agreement/regression_confidence_s42
```

Then run the requested A-D matrix, three seeds each. `-TuneAlpha` also
evaluates alpha values 0.25, 0.5, and 0.75 before the final combined runs:

```powershell
\.\run_agreement_experiments.ps1 -TuneAlpha -OutputDir models_agreement
```

Record test macro-F1 and accuracy from each `test_metrics.json`, and compare
the A-confidence seed-42 run with the regression above before interpreting B,
C, or D.

## Finding wording

> Ensemble agreement between two 7B models is not a reliable quality signal
> for these auxiliary samples. In the available paired data, normalized
> agreement has a near-zero/negative relationship with target-match quality;
> the three-seed A-D matrix is required to test whether that diagnostic
> weakness also explains the downstream training result.