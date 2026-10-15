"""Estimate heterogeneous treatment effects with an honest randomized holdout."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True);p.add_argument("--treatment",required=True);p.add_argument("--outcome",required=True);p.add_argument("--out",default="outputs");a=p.parse_args()
 d=pd.read_csv(a.input).dropna(subset=[a.treatment,a.outcome]);t=d[a.treatment].astype(int);y=d[a.outcome].astype(int);X=d.drop(columns=[a.treatment,a.outcome])
 if set(t.unique())!={0,1}:raise SystemExit("Treatment must be randomized and coded 0/1.")
 tr,te=train_test_split(np.arange(len(d)),test_size=.25,random_state=42,stratify=t.astype(str)+"_"+y.astype(str))
 def fit(mask):
  z=X.iloc[mask];num=z.select_dtypes(include="number").columns;cat=z.columns.difference(num)
  prep=ColumnTransformer([("num",make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler()),num),("cat",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),cat)])
  m=Pipeline([("features",prep),("outcome",LogisticRegression(max_iter=1200,class_weight="balanced"))]);m.fit(z,y.iloc[mask]);return m
 m0=fit(tr[t.iloc[tr].to_numpy()==0]);m1=fit(tr[t.iloc[tr].to_numpy()==1]);u=m1.predict_proba(X.iloc[te])[:,1]-m0.predict_proba(X.iloc[te])[:,1]
 # Qini-like uplift gain: cumulative observed treatment-control response difference in score-ranked buckets.
 ranks=np.argsort(-u);gain=[];cum_t=cum_c=0
 tt=t.iloc[te].to_numpy()[ranks]; yy=y.iloc[te].to_numpy()[ranks]
 for i,(ti,yi) in enumerate(zip(tt,yy),1):
  if ti:cum_t+=yi
  else:cum_c+=yi
  gain.append((cum_t/max(1,(tt[:i]==1).sum()) - cum_c/max(1,(tt[:i]==0).sum()))*i if (tt[:i]==0).sum() and (tt[:i]==1).sum() else 0.0)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pd.DataFrame({"row":te[ranks],"uplift":u[ranks],"treatment":tt,"outcome":yy,"cumulative_uplift_gain":gain}).to_csv(out/"ranked_uplift.csv",index=False)
 report={"train_rows":len(tr),"holdout_rows":len(te),"treatment_rate":float(t.iloc[te].mean()),"holdout_observed_lift":float(y.iloc[te][t.iloc[te].to_numpy()==1].mean()-y.iloc[te][t.iloc[te].to_numpy()==0].mean()),"note":"Uplift estimates require randomized treatment assignment; this is not causal with observational treatment."}
 (out/"metrics.json").write_text(json.dumps(report,indent=2));joblib.dump({"control_model":m0,"treatment_model":m1},out/"models.joblib");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
