"""
讀取 violations.jsonl,切分 train/val/test,存成 csv。
目標:用違規廣告文字(product + violation)預測罰款金額(log尺度迴歸)。
"""
import json
import math
import random
import csv
import re

random.seed(42)

SRC = "../data/violations.jsonl"
OUT_DIR = "../data"

rows = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        text = (d.get("product") or "") + " " + (d.get("violation") or "")
        text = re.sub(r"\s+", " ", text).strip()
        penalty = d.get("penalty_twd")
        law = d.get("law_cited", "")
        m = re.search(r"第(\d+)條第(\d+)項", law)
        label = f"第{m.group(1)}條第{m.group(2)}項" if m else "other"
        if text and penalty:
            rows.append({
                "text": text,
                "penalty_twd": penalty,
                "log_penalty": math.log(penalty),
                "law_label": label,
            })

print(f"總筆數: {len(rows)}")

random.shuffle(rows)
n = len(rows)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

splits = {
    "train": rows[:n_train],
    "val": rows[n_train:n_train + n_val],
    "test": rows[n_train + n_val:],
}

for name, data in splits.items():
    path = f"{OUT_DIR}/{name}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "penalty_twd", "log_penalty", "law_label"])
        writer.writeheader()
        writer.writerows(data)
    print(f"{name}: {len(data)} 筆 -> {path}")
