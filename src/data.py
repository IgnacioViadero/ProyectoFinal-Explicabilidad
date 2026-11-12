import pandas as pd
from sklearn.model_selection import train_test_split

IDENTITIES = [
    "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish",
    "muslim", "black", "white", "psychiatric_or_mental_illness",
]
SEED = 42


def load_raw(path="data/data.csv"):
    cols = ["id", "comment_text", "target"] + IDENTITIES
    return pd.read_csv(path, usecols=cols)


def preprocess(df, n_general=150_000, seed=SEED):
    df = df.dropna(subset=["comment_text", "target"]).copy()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()

    annotated = df[IDENTITIES].notna().any(axis=1)   # filas con anotación de identidad
    for g in IDENTITIES:
        df[f"has_{g}"] = (df[g].fillna(0) >= 0.5)
    df["label"] = (df["target"] >= 0.5).astype(int)  # solo para métricas de sesgo
    df["any_identity"] = df[[f"has_{g}" for g in IDENTITIES]].any(axis=1)

    # todas las filas anotadas + muestra del resto
    rest = df[~annotated].sample(min(n_general, (~annotated).sum()), random_state=seed)
    df = pd.concat([df[annotated], rest]).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)


def split(df, seed=SEED):
    strat = df["label"].astype(str) + df["any_identity"].astype(str)
    tr, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat)
    strat_tmp = tmp["label"].astype(str) + tmp["any_identity"].astype(str)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=strat_tmp)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


if __name__ == "__main__":
    df = preprocess(load_raw())
    tr, va, te = split(df)
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        d.to_parquet(f"data/{name}.parquet")
        print(name, len(d), "| target medio:", round(d["target"].mean(), 4),
              "| toxic:", round(d["label"].mean(), 4))
    print(te[[f"has_{g}" for g in IDENTITIES]].sum().to_string())