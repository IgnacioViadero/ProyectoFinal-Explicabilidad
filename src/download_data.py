# download_data.py
from datasets import load_dataset

ds = load_dataset("google/civil_comments")
print(ds)
print(ds["train"].column_names)
ds["train"].to_parquet("data/train_raw.parquet")
ds["validation"].to_parquet("data/val_raw.parquet")
ds["test"].to_parquet("data/test_raw.parquet")