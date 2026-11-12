import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from src.metrics import bias_report, regression_report

tr = pd.read_parquet("data/train.parquet")
va = pd.read_parquet("data/val.parquet")
te = pd.read_parquet("data/test.parquet")

vec = TfidfVectorizer(max_features=200_000, ngram_range=(1, 2),
                      min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(tr["comment_text"])
reg = Ridge(alpha=1.0).fit(Xtr, tr["target"])

for d in (va, te):
    d["pred"] = np.clip(reg.predict(vec.transform(d["comment_text"])), 0, 1)

reg_rep, bias_rep = regression_report(te), bias_report(te)
print("== REGRESIÓN POR GRUPO ==\n", reg_rep.to_string())
print("\n== SESGO (binario, thr=0.5) ==\n", bias_rep.to_string())

te.to_parquet("outputs/baseline_test_preds.parquet")
reg_rep.to_csv("outputs/baseline_regression_report.csv")
bias_rep.to_csv("outputs/baseline_bias_report.csv")
joblib.dump((vec, reg), "outputs/baseline_tfidf_ridge.joblib")