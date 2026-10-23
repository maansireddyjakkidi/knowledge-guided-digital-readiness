"""Executed World Bank FAT cross-sectional adoption benchmark. Outcome db11a5 is NOT prospective readiness."""
import argparse,json,zipfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

OUTCOME="db11a5"
CAP=["a7a","age_2023","s1b","manager_tertiary"]
TECH=["internet_yes","b4a"]
ALL=CAP+TECH

def load(path):
    with zipfile.ZipFile(path) as z:
        name=next(n for n in z.namelist() if n.endswith("fat0_raw_data_qje.csv"))
        with z.open(name) as f:return pd.read_csv(f,low_memory=False)

def clean(raw):
    cols=["country",OUTCOME,"a7a","a7c","b5a","b4a","a6","s1b"]
    if not set(cols).issubset(raw):raise ValueError("Missing: "+str(set(cols)-set(raw)))
    d=raw[cols].copy()
    for c in cols[1:]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
        d.loc[d[c]<0,c]=np.nan
    d=d[d[OUTCOME].isin([0,1])].copy()
    d["age_2023"]=2023-d["a6"]
    d.loc[~d.age_2023.between(0,150),"age_2023"]=np.nan
    d["internet_yes"]=d["b5a"].map({1:1.,2:0.})
    d.loc[~d["a7c"].isin([1,2,3,4,5,6]),"a7c"]=np.nan
    d.loc[~d["s1b"].isin([1,2,3]),"s1b"]=np.nan
    d.loc[~d["b4a"].between(0,50000),"b4a"]=np.nan
    d["b4a"]=np.log1p(d["b4a"])
    d.loc[~d["a7a"].between(0,80),"a7a"]=np.nan
    d["manager_tertiary"]=d["a7c"].map({1:0,2:0,3:0,4:1,5:1,6:0})
    return d[["country",OUTCOME]+ALL].reset_index(drop=True)

def features(d,v):
    a=d[CAP if v in ("base","knowledge") else ALL].copy()
    if v in ("knowledge","full"):
        a["knowledge_manager_experience"]=d["manager_tertiary"]*(d["a7a"]>10).astype(float)
        if v=="full":
            a["knowledge_internet_manager"]=d["internet_yes"]*d["manager_tertiary"]
            a["knowledge_internet_computers"]=d["internet_yes"]*(d["b4a"]>0).astype(float)
    return a

def model(v):
    if v=="boost":
        return make_pipeline(SimpleImputer(strategy="median",add_indicator=True),HistGradientBoostingClassifier(max_iter=100,l2_regularization=1,random_state=41))
    return make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler(),LogisticRegression(max_iter=1000,C=1,random_state=41))

def scores(y,p):
    return {"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p),"Brier":brier_score_loss(y,p)}

def run(path,out):
    out.mkdir(parents=True,exist_ok=True)
    raw=load(path);d=clean(raw);y=d[OUTCOME].astype(int).to_numpy();groups=d.country.astype(str)
    folds=list(GroupKFold(n_splits=5).split(d,y,groups))
    for i,(tr,te) in enumerate(folds):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2:raise ValueError(f"Fold {i} has only one class")
    variants=["base","fusion","knowledge","full","boost"]
    predictions={v:np.full(len(d),np.nan) for v in variants};stress=[]
    for fold,(tr,te) in enumerate(folds):
        for v in variants:
            m=model(v);m.fit(features(d.iloc[tr],v),y[tr])
            predictions[v][te]=m.predict_proba(features(d.iloc[te],v))[:,1]
            if v!="boost":
                for rate in (0,.15,.30):
                    test=d.iloc[te].copy();rng=np.random.default_rng(5000+fold)
                    for col in ALL:
                        test.loc[rng.random(len(test))<rate,col]=np.nan
                    p=m.predict_proba(features(test,v))[:,1]
                    stress.append({"fold":fold,"model":v,"missingness":rate,"AUROC":roc_auc_score(y[te],p),"Brier":brier_score_loss(y[te],p),"n":len(te)})
    table1=pd.DataFrame([{"model":v,**scores(y,predictions[v])} for v in variants])
    table1.to_csv(out/"Table_1_real_benchmark.csv",index=False)
    table2=pd.DataFrame(stress).groupby(["model","missingness"],as_index=False)[["AUROC","Brier"]].mean()
    table2.to_csv(out/"Table_2_real_ablation_stress.csv",index=False)
    # Prediction-level data may be confidential: save locally, never commit to GitHub.
    pd.DataFrame({"country":groups,"y":y,**predictions}).to_csv(out/"out_of_fold_predictions.csv",index=False)
    rng=np.random.default_rng(23);uniq=groups.unique();diffs=[]
    for _ in range(1000):
        chosen=rng.choice(uniq,len(uniq),replace=True)
        ids=np.concatenate([np.flatnonzero(groups.to_numpy()==c) for c in chosen])
        if len(np.unique(y[ids]))<2:continue
        diffs.append(roc_auc_score(y[ids],predictions["full"][ids])-roc_auc_score(y[ids],predictions["fusion"][ids]))
    delta={"full_minus_fusion_AUROC":float(roc_auc_score(y,predictions["full"])-roc_auc_score(y,predictions["fusion"])),"country_bootstrap_95CI":[float(v) for v in np.quantile(diffs,[.025,.975])]}
    (out/"paired_country_bootstrap.json").write_text(json.dumps(delta,indent=2))
    fig,ax=plt.subplots(figsize=(10,3));ax.axis("off")
    labels=["Survey firm\ncapabilities","Technology\nenablers","Train-fold\nimputation","Knowledge\ninteractions","Country-held-out\nprediction"]
    for j,t in enumerate(labels):
        x=.1+j*.2
        ax.text(x,.53,t,ha="center",va="center",transform=ax.transAxes,bbox={"boxstyle":"round,pad=.55","facecolor":"#e8eef7","edgecolor":"#456"})
        if j<4:ax.annotate("",xy=(x+.17,.53),xytext=(x+.095,.53),xycoords="axes fraction",arrowprops={"arrowstyle":"->"})
    ax.text(.5,.13,"Cross-sectional advanced-marketing adoption, NOT future readiness",ha="center",transform=ax.transAxes)
    fig.savefig(out/"Figure_1_architecture.pdf",bbox_inches="tight");plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    axs[0].bar(table1.model,table1.AUROC);axs[0].set_ylim(0,1);axs[0].set_ylabel("Country-held-out AUROC");axs[0].tick_params(axis="x",rotation=35)
    for v in ["base","fusion","knowledge","full"]:
        t=table2[table2.model==v];axs[1].plot(t.missingness,t.Brier,marker="o",label=v)
    axs[1].set_xlabel("Synthetic missingness fraction");axs[1].set_ylabel("Brier (lower is better)");axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"Figure_2_empirical.pdf",bbox_inches="tight");plt.close(fig)
    manifest={"source":"World Bank FAT 2019-2023","data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"raw_rows":len(raw),"analysis_rows":len(d),"outcome":"db11a5: big-data analytics / ML in customer marketing (0/1)","positive":int(y.sum()),"countries":int(groups.nunique()),"cv":"5-fold country GroupKFold","predictors":ALL,"caveats":["Cross-sectional adoption, not prospective readiness","Unweighted estimates","Hypothesized knowledge interactions","Country-wave grouping requires sensitivity analysis","Stress only on held-out predictors"]}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(table1.to_string(index=False));print(delta)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,default=Path("real_results"));args=p.parse_args();run(args.data,args.out)
