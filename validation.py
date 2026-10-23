"""FAT sensitivity benchmark. Cross-sectional adoption, not prospective readiness.
Run: python validation.py --data WLD_2019-2023_FAT_v01_M_CSV.zip --out validation_results
"""
import argparse, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES=['manager_experience','firm_age','size_code','manager_education','internet','log_computers']
CAP=FEATURES[:4]
RULE_FEATURES=['internet','log_computers'] # Hypothesized nonnegative effects, NOT externally validated.

def load(path):
    with zipfile.ZipFile(path) as z:
        file=next(n for n in z.namelist() if n.endswith('fat0_raw_data_qje.csv'))
        with z.open(file) as f: d=pd.read_csv(f,low_memory=False)
    required=['country','base_wt','db11a5','a7a','a7c','b5a','b4a','a6','s1b']
    if not set(required).issubset(d):raise ValueError('Missing: '+str(set(required)-set(d)))
    for c in required[1:]:d[c]=pd.to_numeric(d[c],errors='coerce');d.loc[d[c]<0,c]=np.nan
    d=d[d.db11a5.isin([0,1])].copy()
    a=pd.DataFrame(index=d.index)
    a['country']=d.country.astype(str)
    a['y']=d.db11a5.astype(int)
    a['base_wt']=d.base_wt.where(d.base_wt>0)
    a['manager_experience']=d.a7a.where(d.a7a.between(0,80))
    # Fixed 2023 reference is only a proxy; do not infer survey-specific age.
    a['firm_age']=(2023-d.a6).where((2023-d.a6).between(0,150))
    a['size_code']=d.s1b.where(d.s1b.isin([1,2,3]))
    a['manager_education']=d.a7c.map({1:0,2:0,3:0,4:1,5:1,6:0})
    a['internet']=d.b5a.map({1:1,2:0})
    a['log_computers']=np.log1p(d.b4a.where(d.b4a.between(0,50000)))
    return a.reset_index(drop=True),len(d)

def prep(train,test,columns):
    imp=SimpleImputer(strategy='median',add_indicator=False)
    a=imp.fit_transform(train[columns]);b=imp.transform(test[columns]);scale=StandardScaler()
    return scale.fit_transform(a),scale.transform(b)

def fit_predict(train,test,variant):
    cols=CAP if variant=='capability' else FEATURES
    a,b=prep(train,test,cols)
    y=train.y.to_numpy()
    if variant=='constrained':
        # Logistic negative log likelihood with ridge and exact nonnegative bounds
        # on selected *standardized* feature coefficients.
        penalty=1.0
        def objective(t):
            z=t[0]+a@t[1:]
            val=np.logaddexp(0,z).sum()-y@z + .5*penalty*np.dot(t[1:],t[1:])
            grad=np.r_[np.sum(expit(z)-y), a.T@(expit(z)-y)+penalty*t[1:]]
            return val,grad
        bounds=[(None,None)]*(a.shape[1]+1)
        for name in RULE_FEATURES:bounds[1+cols.index(name)]=(0,None)
        sol=minimize(objective,np.zeros(a.shape[1]+1),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':500})
        if not sol.success:raise RuntimeError('Constrained optimization failed: '+sol.message)
        return expit(sol.x[0]+b@sol.x[1:]),{name:float(sol.x[1+cols.index(name)]) for name in RULE_FEATURES}
    model=LogisticRegression(C=1.,max_iter=1500,random_state=41)
    model.fit(a,y)
    return model.predict_proba(b)[:,1],({name:float(model.coef_[0][cols.index(name)]) for name in RULE_FEATURES} if variant!='capability' else {})

def metrics(y,p,weights=None):
    kw={'sample_weight':weights} if weights is not None else {}
    return {'AUROC':float(roc_auc_score(y,p,**kw)), 'AUPRC':float(average_precision_score(y,p,**kw)), 'Brier':float(brier_score_loss(y,p,**kw))}

def ece(y,p,w=None):
    w=np.ones(len(y)) if w is None else np.asarray(w)
    bins=np.minimum((p*10).astype(int),9)
    return float(sum(w[bins==i].sum()/w.sum()*abs(np.average(y[bins==i],weights=w[bins==i])-np.average(p[bins==i],weights=w[bins==i])) for i in range(10) if np.any(bins==i)))

def evaluate(df,out):
    out.mkdir(parents=True,exist_ok=True)
    y=df.y.to_numpy();groups=df.country.to_numpy();w=df.base_wt.to_numpy()
    # Avoid claiming design-weighted representativeness: missing weights excluded only from weighted metrics.
    valid_w=np.isfinite(w)&(w>0)
    print('Sampling weights available:',valid_w.sum(),'/',len(w))
    variants=['capability','fusion','constrained']
    pred={v:np.full(len(df),np.nan) for v in variants}
    rule_rows=[];stress_rows=[];country_rows=[]
    folds=list(GroupKFold(n_splits=5).split(df,y,groups))
    for fold,(tr,te) in enumerate(folds):
        if len(set(y[tr]))<2 or len(set(y[te]))<2:raise ValueError('Single-class grouped fold')
        for v in variants:
            p,coefs=fit_predict(df.iloc[tr],df.iloc[te],v)
            pred[v][te]=p
            if coefs:rule_rows.append({'fold':fold,'model':v,**coefs,'violations':sum(x< -1e-9 for x in coefs.values())})
            for rate in [0,.15,.30]:
                test=df.iloc[te].copy();rng=np.random.default_rng(501+fold)
                if rate:
                    for c in FEATURES:test.loc[rng.random(len(test))<rate,c]=np.nan
                pp,_=fit_predict(df.iloc[tr],test,v)
                stress_rows.append({'fold':fold,'model':v,'missingness':rate,'n':len(te),**metrics(y[te],pp)})
    table=[]
    for v in variants:
        p=pred[v]; row={'model':v,'n':len(df),'positives':int(y.sum()),**metrics(y,p),'ECE10':ece(y,p)}
        if valid_w.sum()>100:
            row.update({'weighted_'+k:val for k,val in metrics(y[valid_w],p[valid_w],w[valid_w]).items()})
            row['weighted_ECE10']=ece(y[valid_w],p[valid_w],w[valid_w])
        table.append(row)
    pd.DataFrame(table).to_csv(out/'Table_1_validation.csv',index=False)
    stress=pd.DataFrame(stress_rows)
    stress.groupby(['model','missingness'],as_index=False)[['AUROC','AUPRC','Brier']].mean().to_csv(out/'Table_2_stress.csv',index=False)
    pd.DataFrame(rule_rows).to_csv(out/'rule_diagnostics.csv',index=False)
    # Paired country bootstrap; country clusters, not independent rows.
    rng=np.random.default_rng(123);unique=np.unique(groups);ids_by={g:np.flatnonzero(groups==g) for g in unique}
    boot=[]
    for _ in range(500):
        ids=np.concatenate([ids_by[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ids]))<2:continue
        boot.append(roc_auc_score(y[ids],pred['constrained'][ids])-roc_auc_score(y[ids],pred['fusion'][ids]))
    delta=float(roc_auc_score(y,pred['constrained'])-roc_auc_score(y,pred['fusion']))
    # Leave-one-country-out evaluation for every country with both classes.
    for country in unique:
        te=np.flatnonzero(groups==country);tr=np.flatnonzero(groups!=country)
        if len(np.unique(y[te]))<2 or len(np.unique(y[tr]))<2:continue
        for v in variants:
            p,_=fit_predict(df.iloc[tr],df.iloc[te],v)
            country_rows.append({'country':country,'n':len(te),'positives':int(y[te].sum()),'model':v,**metrics(y[te],p)})
    pd.DataFrame(country_rows).to_csv(out/'country_holdout.csv',index=False)
    summary={'n':len(df),'positives':int(y.sum()),'groups':len(unique),'weighted_rows':int(valid_w.sum()),'AUROC_constrained_minus_fusion':delta,'country_bootstrap_95CI':[float(x) for x in np.quantile(boot,[.025,.975])], 'hypothesis_only_constraints':RULE_FEATURES,'outcome':'Observed advanced marketing analytics adoption, NOT prospective readiness','limitations':['Survey weights used only in metric sensitivity, not model fitting','No strata/PSU design variance; country bootstrap is not design-based inference','Constraint satisfaction is tautological, not evidence of domain validity','No nested tuning; model settings fixed','Survey year not reliably available for true firm age']}
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    # Exactly two figures; all visualized values derive from observed predictions.
    fig,ax=plt.subplots(figsize=(9,2.7));ax.axis('off');labels=['FAT firm capabilities','Technology inputs','Train-only preprocessing','Constrained logistic fit','Country-held-out evaluation']
    for i,label in enumerate(labels):
        x=.10+.2*i;ax.text(x,.52,label.replace(' ','\n',1),transform=ax.transAxes,ha='center',va='center',fontsize=9,bbox={'boxstyle':'round','facecolor':'#eaf1f6','edgecolor':'#486782'})
        if i<4:ax.annotate('',xy=(x+.14,.52),xytext=(x+.07,.52),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
    ax.text(.5,.13,'Hypothesized nonnegative internet/computer coefficients; adoption outcome, not readiness',transform=ax.transAxes,ha='center',fontsize=9)
    fig.savefig(out/'Figure_1_validation_architecture.pdf',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5));t=pd.DataFrame(table)
    axs[0].bar(t.model,t.AUROC);axs[0].set_ylim(0,1);axs[0].set_title('(a) Country-held-out AUROC');axs[0].tick_params(axis='x',rotation=25)
    for v in variants:
        pp=pred[v];order=np.argsort(pp);bins=np.array_split(order,10)
        axs[1].plot([np.mean(pp[idx]) for idx in bins],[np.mean(y[idx]) for idx in bins],marker='o',label=v)
    axs[1].plot([0,1],[0,1],ls=':',color='gray');axs[1].set_xlim(0,.25);axs[1].set_ylim(0,.25);axs[1].set_title('(b) Calibration, equal-count bins');axs[1].set_xlabel('Predicted');axs[1].set_ylabel('Observed')
    s=stress.groupby(['model','missingness'],as_index=False)['Brier'].mean()
    for v in variants:
        a=s[s.model==v];axs[2].plot(a.missingness,a.Brier,marker='o',label=v)
    axs[2].set_title('(c) Missingness stress, fold means');axs[2].set_xlabel('Test missingness');axs[2].set_ylabel('Brier');axs[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'Figure_2_validation_results.pdf',bbox_inches='tight');plt.close(fig)
    print(pd.DataFrame(table).to_string(index=False));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,default=Path('validation_results'));a=parser.parse_args()
    df,raw_n=load(a.data);print('Eligible records',raw_n);evaluate(df,a.out)
