# Knowledge-guided ML for entrepreneurial digital readiness

Minimal reproducible research prototype for a *Knowledge-Based Systems* short communication.

## Run

```bash
pip install -r requirements.txt
python experiment.py --demo --out results
python experiment.py --csv YOUR_DATA.csv --out results
```

Expected CSV: `digital_skills,cloud_adoption,cybersecurity,finance_access,leadership_support,firm_age,sector,readiness` with binary readiness (0/1). Numeric inputs must be consistently coded. The label must be measured independently of predictors and expert rules to avoid circular evaluation. Missing numeric values are imputed within training folds. The category encoder handles unseen sectors.

Outputs: two vector PDF figures, `metrics.csv`, and out-of-fold predictions. Figures and scores generated with `--demo` are **synthetic software demonstrations only** and must not be presented as findings about firms. `knowledge_rule` is a transparent unvalidated heuristic; replace it with independently elicited, preregistered expert knowledge for substantive research. For actual research, document provenance, consent, inclusion criteria, target construct validity, temporal separation, missingness, class balance, leakage controls, external validation, uncertainty intervals, and fairness/sector subgroup analyses.

Five-fold stratified cross-validation compares logistic regression, histogram gradient boosting, and a rule-augmented logistic regression. Preprocessing is fit inside folds. Predictions are out of fold. This is a *baseline prototype*, not a demonstrated methodological advance or publication-ready validation. Before claiming novelty, compare to stronger baselines and perform ablations, independent external validation, and significance/uncertainty analysis.

Repository intentionally uses only three tracked files; outputs are generated locally. Do not commit confidential firm-level data. MIT-style license not assigned; choose a license before public release.
