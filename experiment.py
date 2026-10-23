"""KG-MRL: leakage-aware, configurable, reproducible real-data evaluation.
Synthetic mode is a software stress-test, NOT empirical evidence.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(path):
    with open(path, encoding='utf-8') as f:
        c = json.load(f)
    required = ['target', 'numeric', 'categorical', 'groups', 'sources', 'rules']
    if any(k not in c for k in required):
        raise ValueError('Config requires: ' + ', '.join(required))
    cols = c['numeric'] + c['categorical']
    if len(cols) != len(set(cols)) or c['target'] in cols:
        raise ValueError('Predictors must be unique and exclude the outcome')
    for r in c['rules']:
        if not set(r['features']).issubset(cols):
            raise ValueError('Rule references unknown predictors: ' + r['name'])
        if r.get('direction', 1) not in [-1, 1]:
            raise ValueError('Rule direction must be +1 or -1')
    return c


def synthetic_data(n=900, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 5))
    d = pd.DataFrame(z, columns=['digital_skills', 'cloud_adoption', 'cybersecurity', 'finance_access', 'leadership_support'])
    d['firm_age'] = rng.integers(1, 25, n)
    d['sector'] = rng.choice(['retail', 'services', 'manufacturing'], n)
    d['country'] = rng.choice(['A', 'B', 'C', 'D', 'E'], n)
    # Outcome is artificially generated; never interpret as evidence about real firms.
    s = .5*z[:, 0] + .45*z[:, 1] + .35*z[:, 2] + .25*z[:, 3] + .3*z[:, 4] + .35*((z[:, :3] > 0).all(axis=1))
    d['readiness'] = rng.binomial(1, 1/(1+np.exp(-s)))
    return d


def rule_matrix(frame, config, allowed=None):
    columns = []
    for rule in config['rules']:
        if allowed is not None and not set(rule['features']).issubset(set(allowed)):
            continue
        masks = []
        for name, threshold in zip(rule['features'], rule['thresholds']):
            x = pd.to_numeric(frame[name], errors='coerce')
            masks.append((x >= threshold).fillna(False).to_numpy())
        columns.append(np.logical_and.reduce(masks).astype(float) * rule.get('direction', 1))
    return np.column_stack(columns) if columns else np.empty((len(frame), 0))


def make_model(config, variant, seed):
    use_fusion = variant in ('fusion', 'full')
    use_rules = variant in ('knowledge', 'full')
    numeric = list(config['numeric'])
    categorical = list(config['categorical'])
    if not use_fusion:
        allowed = set(config['sources'][0]['columns'])
        numeric = [x for x in numeric if x in allowed]
        categorical = [x for x in categorical if x in allowed]
    if not numeric and not categorical:
        raise ValueError('First source must contain at least one predictor')
    transformers = []
    if numeric:
        transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=False)), ('scale', StandardScaler())]), numeric))
    if categorical:
        transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), categorical))
    prep = ColumnTransformer(transformers)
    return prep, numeric + categorical, use_rules


def fit_predict(train, test, config, variant, seed):
    prep, cols, use_rules = make_model(config, variant, seed)
    prep.fit(train[cols])
    a = prep.transform(train[cols]); b = prep.transform(test[cols])
    if hasattr(a, 'toarray'): a = a.toarray()
    if hasattr(b, 'toarray'): b = b.toarray()
    if use_rules:
        a = np.column_stack([a, rule_matrix(train, config, cols)])
        b = np.column_stack([b, rule_matrix(test, config, cols)])
    y = train[config['target']].to_numpy().astype(int)
    if variant == 'forest':
        model = RandomForestClassifier(n_estimators=150, min_samples_leaf=4, n_jobs=1, random_state=seed)
    elif variant == 'boosting':
        model = HistGradientBoostingClassifier(max_iter=120, l2_regularization=1.0, random_state=seed)
    else:
        model = LogisticRegression(max_iter=1500, C=1., random_state=seed)
    model.fit(a, y)
    return model.predict_proba(b)[:, 1]


def score(y, p):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    out = {'AUROC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
           'AUPRC': float(average_precision_score(y, p)),
           'Brier': float(brier_score_loss(y, p))}
    edges = np.linspace(0, 1, 11)
    ece = 0.
    for i in range(10):
        mask = (p >= edges[i]) & (p < edges[i+1] if i < 9 else p <= 1)
        if mask.any(): ece += mask.mean() * abs(p[mask].mean() - np.asarray(y)[mask].mean())
    out['ECE10'] = float(ece)
    return out


def perturb(df, config, level, mode, seed):
    if level == 0: return df.copy()
    rng = np.random.default_rng(seed)
    d = df.copy()
    numeric = config['numeric']
    if mode == 'missing':
        for c in numeric:
            d.loc[rng.random(len(d)) < level, c] = np.nan
    elif mode == 'noise':
        for c in numeric:
            x = pd.to_numeric(d[c], errors='coerce')
            d[c] = x + rng.normal(0, level * max(float(x.std()), .001), len(d))
    else: raise ValueError('Unsupported stress type')
    return d


def evaluate(df, config, out, seed=42, n_splits=5, stress=(0., .15, .3)):
    out.mkdir(parents=True, exist_ok=True)
    target = config['target']; features = config['numeric'] + config['categorical']
    required = features + [target]
    if any(c not in df for c in required):
        raise ValueError('Missing columns: ' + ', '.join(c for c in required if c not in df))
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if not set(df[target].unique()).issubset({0, 1}): raise ValueError('Outcome must be binary 0/1')
    if df[target].value_counts().min() < n_splits: raise ValueError('Insufficient minority-class observations')
    groups = config.get('groups')
    if groups and groups in df:
        if df[groups].nunique() < n_splits: raise ValueError('Insufficient distinct groups')
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, df[target], groups=df[groups]))
        split_kind = 'GroupKFold'
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df, df[target]))
        split_kind = 'StratifiedKFold'
    variants = ['baseline', 'fusion', 'knowledge', 'full', 'forest', 'boosting']
    # Conventional forest and boosting use the full feature set but no knowledge rules.
    preds = {v: np.full(len(df), np.nan) for v in variants}
    elapsed = {v: 0. for v in variants}
    fold_ids = np.full(len(df), -1, dtype=int)
    for fold, (tr, te) in enumerate(splits):
        train, test = df.iloc[tr], df.iloc[te]
        if train[target].nunique() != 2 or test[target].nunique() != 2:
            raise ValueError('A grouped fold has one outcome class; choose a viable grouping')
        fold_ids[te] = fold
        for v in variants:
            variant = 'fusion' if v in ('forest', 'boosting') else v
            start = time.perf_counter()
            if v in ('forest', 'boosting'):
                # Baseline type dispatch is explicit; fit_predict recognizes these variants.
                prep, cols, _ = make_model(config, 'fusion', seed)
                prep.fit(train[cols]); a=prep.transform(train[cols]); b=prep.transform(test[cols])
                if hasattr(a,'toarray'): a=a.toarray()
                if hasattr(b,'toarray'): b=b.toarray()
                model = (RandomForestClassifier(n_estimators=150,min_samples_leaf=4,n_jobs=1,random_state=seed+fold)
                         if v=='forest' else HistGradientBoostingClassifier(max_iter=120,l2_regularization=1.,random_state=seed+fold))
                model.fit(a, train[target].to_numpy().astype(int)); p=model.predict_proba(b)[:,1]
            else:
                p = fit_predict(train, test, config, variant, seed+fold)
            elapsed[v] += time.perf_counter()-start
            preds[v][te] = p
    y = df[target].to_numpy().astype(int)
    results = []
    for v in variants:
        results.append({'model':v, **score(y, preds[v]), 'runtime_s':round(elapsed[v], 3)})
    metrics = pd.DataFrame(results)
    metrics.to_csv(out/'Table_1_benchmark.csv', index=False)
    pred_frame = pd.DataFrame({'outcome':y, 'fold':fold_ids, **preds})
    if groups and groups in df: pred_frame['group'] = df[groups].astype(str)
    pred_frame.to_csv(out/'out_of_fold_predictions.csv', index=False)
    stress_rows=[]
    for mode in ['missing', 'noise']:
        for level in stress:
            if not (0 <= level <= 1): raise ValueError('Stress levels must be within [0,1]')
            for v in ['baseline','fusion','knowledge','full']:
                p = np.full(len(df),np.nan)
                for fold,(tr,te) in enumerate(splits):
                    train = df.iloc[tr]; test=perturb(df.iloc[te],config,level,mode,seed+fold+1000)
                    p[te] = fit_predict(train,test,config,v,seed+fold)
                stress_rows.append({'stress':mode,'level':level,'model':v,**score(y,p)})
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(out/'Table_2_ablation_stress.csv',index=False)
    # Paired bootstrap of out-of-fold predictions; descriptive uncertainty, not independent external validation.
    rng=np.random.default_rng(seed+999)
    ci=[]
    for v in variants:
        diffs=[]
        for _ in range(300):
            ids=rng.integers(0,len(y),len(y))
            if len(np.unique(y[ids])) < 2: continue
            diffs.append(roc_auc_score(y[ids],preds[v][ids]))
        ci.append({'model':v,'AUROC_CI_low':float(np.quantile(diffs,.025)),'AUROC_CI_high':float(np.quantile(diffs,.975))})
    pd.DataFrame(ci).to_csv(out/'bootstrap_intervals.csv',index=False)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,3.1)); ax.axis('off')
    labels=['Firm data\nsource A','Context\nsource B','Train-only\npreprocessing','Rule-augmented\nlearning','Held-out\nevaluation']
    xs=np.linspace(.09,.91,len(labels))
    for x,t in zip(xs,labels): ax.text(x,.53,t,transform=ax.transAxes,ha='center',va='center',bbox={'boxstyle':'round,pad=.5','facecolor':'#e8f0f8','edgecolor':'#34495e'})
    for x0,x1 in zip(xs[:-1],xs[1:]): ax.annotate('',xy=(x1-.075,.53),xytext=(x0+.075,.53),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#34495e'})
    ax.text(.5,.12,'Knowledge constraints specified before evaluating held-out labels',transform=ax.transAxes,ha='center',fontsize=9)
    fig.savefig(out/'Figure_1_architecture.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    m=metrics.set_index('model'); axes[0].bar(np.arange(len(variants)),m.loc[variants,'AUROC']);axes[0].set_xticks(range(len(variants)),variants,rotation=30,ha='right');axes[0].set_ylim(0,1);axes[0].set_ylabel('Out-of-fold AUROC');axes[0].set_title('(a) Model discrimination')
    for v in ['baseline','fusion','knowledge','full']:
        s=stress_df[(stress_df['stress']=='missing')&(stress_df['model']==v)]
        axes[1].plot(s['level'],s['Brier'],marker='o',label=v)
    axes[1].set_xlabel('Held-out missingness probability');axes[1].set_ylabel('Brier score (lower better)');axes[1].set_title('(b) Missingness stress test');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'Figure_2_results.pdf',bbox_inches='tight');plt.close(fig)
    manifest={'synthetic':config.get('_synthetic',False),'n':len(df),'seed':seed,'folds':n_splits,'split':split_kind,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'config':config,'data_sha256':hashlib.sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print(metrics.to_string(index=False));print('Results:',out.resolve())
    if config.get('_synthetic'): print('WARNING: synthetic demonstration only; no real-world claims permitted.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,help='Real CSV or Parquet firm-level data')
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('knowledge_rules.json'))
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--demo',action='store_true',help='Synthetic software test, NOT empirical research')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--folds',type=int,default=5)
    args=p.parse_args()
    if args.demo == bool(args.data): p.error('Specify exactly one of --data or --demo')
    c=load_config(args.config)
    if args.demo: df=synthetic_data(seed=args.seed);c['_synthetic']=True
    else:
        df=pd.read_parquet(args.data) if args.data.suffix.lower()=='.parquet' else pd.read_csv(args.data)
        c['_synthetic']=False
    evaluate(df,c,args.out,args.seed,args.folds)

if __name__=='__main__':main()
