"""Audit a locally downloaded World Bank FAT file before defining a target.

Does not guess the outcome or automatically label firms as ready.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def read_data(path):
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path, low_memory=False)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix == '.dta':
        return pd.read_stata(path, convert_categoricals=False)
    raise ValueError('Supported formats: .csv, .parquet, .dta')

def audit(df, source):
    rows = []
    for name in df.columns:
        s = df[name]
        samples = [str(x)[:70] for x in s.dropna().unique()[:5]]
        rows.append({'column': name, 'dtype': str(s.dtype),
                     'missing_fraction': round(float(s.isna().mean()), 5),
                     'distinct_nonnull': int(s.nunique(dropna=True)),
                     'sample_values': ' | '.join(samples)})
    return pd.DataFrame(rows), {
        'source_file': str(source), 'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'warning': 'Descriptive audit only. No outcome or knowledge rules inferred.',
        'outcome_checklist': [
            'Specify the exact observed survey question or derived target',
            'Exclude every target component and deterministic proxy from predictors',
            'Document whether the outcome is concurrent or measured later',
            'Check survey skip patterns and country comparability',
            'Inspect missingness, survey weights, and design variables',
            'Validate rule direction and thresholds against independent evidence',
            'Freeze mapping and evaluation protocol before seeing test results'
        ]}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--out', default='audit_results', type=Path)
    args = p.parse_args()
    df = read_data(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    cols, meta = audit(df, args.data)
    cols.to_csv(args.out / 'fat_column_audit.csv', index=False)
    (args.out / 'fat_audit_summary.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Audited {len(df)} rows and {df.shape[1]} columns; output: {args.out}')
    print('No readiness target has been defined or validated.')

if __name__ == '__main__':
    main()
