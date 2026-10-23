# KG-MRL: Knowledge-guided multisource readiness learning

Reproducible **research software**, not a completed empirical study. Main-text outputs: **exactly two vector PDF figures and two CSV tables**. No fabricated empirical results.

## Installation and smoke test

```bash
pip install -r requirements.txt
python experiment.py --demo --out demo_results
```

Synthetic mode is **only a software smoke test and controlled simulation**. It is not evidence of firm-level prediction accuracy or model novelty.

## Real-data experiment

1. Download a legitimately accessible, documented firm-level dataset (candidate: [World Bank FAT survey](https://microdata.worldbank.org/collections/fat)). Observe its license, access restrictions, and citation requirements.
2. Construct a **measured and non-circular binary outcome**. A technology adoption indicator that is itself used among predictors **must not** be used as the target. Prefer subsequent adoption if longitudinal measurement exists; otherwise describe a cross-sectional proxy without calling it prospective readiness.
3. Harmonize features into a local CSV or Parquet file with the columns specified in `knowledge_rules.json`. The default configuration is a **placeholder schema**, not a verified mapping to any World Bank data release. Do not impute missing survey fields with made-up observations. Numeric variables and rule thresholds require documented harmonized units.
4. Edit `knowledge_rules.json` to define source-specific columns, outcome, group, and independently justified rules. Rule names and thresholds shipped here are **unvalidated hypotheses**, not established domain knowledge.
5. Run:

```bash
python experiment.py --data path/to/validated_firms.csv --config knowledge_rules.json --out real_results
```

## Outputs

- `Table_1_benchmark.csv`: out-of-fold discrimination, precision-recall, Brier score, ECE10, runtime.
- `Table_2_ablation_stress.csv`: 2x2 source-fusion × knowledge-rule ablation, plus missingness and noise stress tests.
- `Figure_1_architecture.pdf`: conceptual model and evaluation diagram.
- `Figure_2_results.pdf`: AUROC and missingness robustness.
- `out_of_fold_predictions.csv`, `bootstrap_intervals.csv`, `manifest.json`.

GroupKFold is used if a group column is present (e.g., country), otherwise stratified folds. Grouped folds may fail if class distribution is unsuitable; that failure is intentional. Imputation and encoding are fit only on training folds. Stress corruptions affect **test predictors only**, not the outcomes. Bootstrap intervals are descriptive and do not substitute for independent external validation. The knowledge model uses rule-derived predictors; it is **not yet a new constrained-optimization algorithm**, and any novelty claim requires literature comparison and stronger methodological development.

## Important limitations

No real dataset is bundled, no real-data performance is claimed, and no outcome definition or cross-country harmonization has been validated. Synthetic labels depend on constructed predictors. Survey weights, clustering, nested hyperparameter tuning, fairness, temporal validity, and external replication need study-specific decisions before journal submission. Do not publish restricted microdata. The software is a reproducible starting implementation, not a completed *Knowledge-Based Systems* experiment.
