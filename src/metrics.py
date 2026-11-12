import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

from src.data import IDENTITIES


def _auc(y, p):
    """AUC solo si hay las dos clases; si no, NaN."""
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def regression_report(df, pred_col="pred", target_col="target"):
    """Error de regresión por grupo de identidad.
    mean_error > 0 significa que el modelo sobrestima la toxicidad."""
    rows = []
    for g in IDENTITIES + ["OVERALL"]:
        d = df if g == "OVERALL" else df[df[f"has_{g}"]]
        err = d[pred_col] - d[target_col]
        rows.append({
            "group": g,
            "n": len(d),
            "mean_target": d[target_col].mean(),
            "mean_pred": d[pred_col].mean(),
            "mean_error": err.mean(),
            "MAE": mean_absolute_error(d[target_col], d[pred_col]),
            "RMSE": mean_squared_error(d[target_col], d[pred_col]) ** 0.5,
            "spearman": spearmanr(d[target_col], d[pred_col])[0],
        })
    return pd.DataFrame(rows).set_index("group").round(4)


def bias_report(df, pred_col="pred", label_col="label", thr=0.5):
    """Métricas de sesgo con etiqueta binaria (target >= 0.5).
    FPR: no tóxicos del grupo marcados como tóxicos.
    BPSN: negativos del grupo + positivos del resto.
    BNSP: positivos del grupo + negativos del resto."""
    rows = []
    for g in IDENTITIES:
        m = df[f"has_{g}"].astype(bool)
        sub, bg = df[m], df[~m]

        bpsn = pd.concat([sub[sub[label_col] == 0], bg[bg[label_col] == 1]])
        bnsp = pd.concat([sub[sub[label_col] == 1], bg[bg[label_col] == 0]])
        neg = sub[sub[label_col] == 0]
        pos = sub[sub[label_col] == 1]

        rows.append({
            "group": g,
            "n": int(m.sum()),
            "n_toxic": int(len(pos)),
            "toxic_rate": sub[label_col].mean(),
            "pred_toxic_rate": (sub[pred_col] >= thr).mean(),
            "FPR": (neg[pred_col] >= thr).mean() if len(neg) else np.nan,
            "FNR": (pos[pred_col] < thr).mean() if len(pos) else np.nan,
            "subgroup_auc": _auc(sub[label_col], sub[pred_col]),
            "bpsn_auc": _auc(bpsn[label_col], bpsn[pred_col]),
            "bnsp_auc": _auc(bnsp[label_col], bnsp[pred_col]),
        })

    neg_all = df[df[label_col] == 0]
    pos_all = df[df[label_col] == 1]
    rows.append({
        "group": "OVERALL",
        "n": len(df),
        "n_toxic": int(len(pos_all)),
        "toxic_rate": df[label_col].mean(),
        "pred_toxic_rate": (df[pred_col] >= thr).mean(),
        "FPR": (neg_all[pred_col] >= thr).mean(),
        "FNR": (pos_all[pred_col] < thr).mean(),
        "subgroup_auc": _auc(df[label_col], df[pred_col]),
        "bpsn_auc": np.nan,
        "bnsp_auc": np.nan,
    })
    return pd.DataFrame(rows).set_index("group").round(4)