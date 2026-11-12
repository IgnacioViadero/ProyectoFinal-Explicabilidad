import argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("--name", default="distilbert_base")
p.add_argument("--model", default="distilbert-base-uncased")
p.add_argument("--n_train", type=int, default=150_000)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--weights_col", default=None)   # para la mitigación (paso 6)
a = p.parse_args()

torch.manual_seed(42); np.random.seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = pd.read_parquet("data/train.parquet")
if a.n_train < len(tr):
    tr = tr.sample(a.n_train, random_state=42)
va = pd.read_parquet("data/val.parquet")
te = pd.read_parquet("data/test.parquet")
tok = AutoTokenizer.from_pretrained(a.model)

class DS(Dataset):
    def __init__(s, df):
        s.t = df["comment_text"].tolist(); s.y = df["target"].values.astype("float32")
        s.w = df[a.weights_col].values.astype("float32") if a.weights_col else np.ones(len(df), "float32")
    def __len__(s): return len(s.t)
    def __getitem__(s, i): return s.t[i], s.y[i], s.w[i]

def collate(b):
    t, y, w = zip(*b)
    enc = tok(list(t), truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
    return enc, torch.tensor(y), torch.tensor(w)

dl = DataLoader(DS(tr), batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=2)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = len(dl) * a.epochs
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
bce = torch.nn.BCEWithLogitsLoss(reduction="none")

@torch.no_grad()
def predict(df, bs=128):
    model.eval(); out = []
    texts = df["comment_text"].tolist()
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(dev)
        with torch.autocast(device_type=dev, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu())
    return torch.cat(out).numpy()

for ep in range(a.epochs):
    model.train(); tot = 0
    for i, (enc, y, w) in enumerate(dl):
        enc, y, w = enc.to(dev), y.to(dev), w.to(dev)
        with torch.autocast(device_type=dev, enabled=dev == "cuda"):
            logit = model(**enc).logits.squeeze(-1)
        loss = (bce(logit.float(), y) * w).sum() / w.sum()
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        tot += loss.item()
        if i % 500 == 0: print(f"ep{ep} it{i}/{len(dl)} loss {tot/(i+1):.4f}", flush=True)
    va["pred"] = predict(va)
    print(f"== epoch {ep} | val spearman {spearmanr(va['target'], va['pred'])[0]:.4f}", flush=True)

te["pred"] = predict(te)
te.to_parquet(f"outputs/{a.name}_test_preds.parquet")
model.save_pretrained(f"outputs/{a.name}_model"); tok.save_pretrained(f"outputs/{a.name}_model")
print("Guardado.")