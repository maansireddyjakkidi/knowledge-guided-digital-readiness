"""Reproducible prototype; synthetic data are NOT empirical evidence."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

FEATURES = ['digital_skills','cloud_adoption','cybersecurity','finance_access','leadership_support','firm_age','sector']
NUMERIC = FEATURES[:-1]
RULE_FEATURES = ['digital_skills','cloud_adoption','cybersecurity','finance_access','leadership_support']

def demo_data(n=1200, seed=42):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(n,5))
    age=rng.integers(1,26,n)
    sector=rng.choice(['retail','services','manufacturing'],n)
    # The outcome explicitly depends on knowledge-rule-like conditions; demo is illustrative only.
    latent=.65*z[:,0]+.5*z[:,1]+.45*z[:,2]+.3*z[:,3]+.45*z[:,4]+.6*((z[:,:3]>0).all(axis=1))+.12*(sector=='services')
    p=1/(1+np.exp(-latent))
    df=pd.DataFrame(z,columns=RULE_FEATURES)
    df['firm_age']=age
    df['sector']=sector
    df['readiness']=rng.binomial(1,p)
    return df

def knowledge_score(df):
    """Prespecified transparent domain heuristic, not expert-validated knowledge."""
    a=df[RULE_FEATURES].astype(float).fillna(0)
    return ((a['digital_skills']>0)&(a['cloud_adoption']>0)&(a['cybersecurity']>0)).astype(float).to_numpy()

def pipeline(model):
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['sector'])])
    return Pipeline([('prep',prep),('model',model)])

def run(df, out, folds=5, seed=42):
    if 'readiness' not in df or not set(FEATURES).issubset(df):
        raise ValueError('Expected columns: '+', '.join(FEATURES+['readiness']))
    if df['readiness'].isna().any() or not set(df['readiness'].unique()).issubset({0,1}):
        raise ValueError('readiness must be binary 0/1 without missing values')
    X=df[FEATURES].copy(); y=df['readiness'].astype(int).to_numpy()
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    predictions={k:np.zeros(len(df)) for k in ['logistic','boosting','knowledge_guided']}
    for train,test in cv.split(X,y):
        base=pipeline(LogisticRegression(max_iter=1500,random_state=seed))
        boost=pipeline(HistGradientBoostingClassifier(max_iter=130,random_state=seed))
        base.fit(X.iloc[train],y[train]); boost.fit(X.iloc[train],y[train])
        predictions['logistic'][test]=base.predict_proba(X.iloc[test])[:,1]
        predictions['boosting'][test]=boost.predict_proba(X.iloc[test])[:,1]
        # Knowledge-guided variant: add prespecified rule to a regularized logistic model.
        # Rule is computed using ONLY predictors, never labels.
        xtr=X.iloc[train].copy(); xte=X.iloc[test].copy()
        xtr['knowledge_rule']=knowledge_score(xtr); xte['knowledge_rule']=knowledge_score(xte)
        kgprep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC+['knowledge_rule']),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['sector'])])
        kg=Pipeline([('prep',kgprep),('model',LogisticRegression(max_iter=1500,random_state=seed))])
        kg.fit(xtr,y[train]); predictions['knowledge_guided'][test]=kg.predict_proba(xte)[:,1]
    rows=[]
    for name,p in predictions.items():
        rows.append({'model':name,'AUROC':roc_auc_score(y,p),'AUPRC':average_precision_score(y,p),'Brier':brier_score_loss(y,p)})
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame({'outcome':y,**predictions}).to_csv(out/'out_of_fold_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3.8))
    ax.bar([r['model'] for r in rows],[r['AUROC'] for r in rows]); ax.set_ylabel('Out-of-fold AUROC'); ax.set_ylim(0,1); ax.tick_params(axis='x',rotation=12); fig.tight_layout(); fig.savefig(out/'Figure_2.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.8)); ax.axis('off')
    boxes=[('Multisource features',.04),('Knowledge rule + ML',.37),('Readiness prediction',.70)]
    for t,x in boxes: ax.text(x,.5,t,ha='left',va='center',fontsize=12,bbox=dict(boxstyle='round',facecolor='#e9f1f8',edgecolor='#345'))
    ax.annotate('',xy=(.36,.5),xytext=(.25,.5),arrowprops=dict(arrowstyle='->'));ax.annotate('',xy=(.69,.5),xytext=(.58,.5),arrowprops=dict(arrowstyle='->'))
    fig.savefig(out/'Figure_1.pdf',bbox_inches='tight');plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path);p.add_argument('--out',type=Path,default=Path('results'));p.add_argument('--demo',action='store_true');args=p.parse_args()
    if not args.demo and not args.csv:p.error('Supply --csv REAL_DATA.csv or --demo')
    data=demo_data() if args.demo else pd.read_csv(args.csv)
    run(data,args.out)
    print('DEMONSTRATION ONLY: synthetic outcomes are not evidence of external validity.' if args.demo else 'Real-data analysis: validate data provenance and label definition before reporting.')
