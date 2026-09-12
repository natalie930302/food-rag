"""
延伸實驗:原本的 bert-base-chinese(~102M參數)微調在280筆資料上輸給TF-IDF+Ridge
baseline(test R² -0.278 vs 0.391)。這支腳本驗證README「後續可以做的改進」提到
的方向:換成參數量更小的中文模型,並加強正則化,能不能縮小差距。

改動:
1. 模型換成 hfl/rbt3(RoBERTa-wwm-ext 的3層蒸餾版,~38M參數,約bert-base-chinese
   的1/3),層數少代表可訓練參數少,理論上在小資料下更不容易overfit
2. Dropout 從 0.1 提高到 0.3
3. AdamW 加上 weight_decay=0.01(原本是0)
4. 用 val R² 做 early stopping / best checkpoint 選擇,而不是直接用最後一個epoch
   的權重去評估test——原本的版本容易被訓練後期的overfit epoch拖累

其餘設定(資料切分、log轉換、評估指標)跟 train_bert.py 完全一致,只變動上面
4點,才能把差異歸因到「模型更小+正則化更強」這個假設上。
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import copy

MODEL_NAME = "hfl/rbt3"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {DEVICE}, 模型: {MODEL_NAME}")

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")
test_df = pd.read_csv("../data/test.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class ViolationDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["log_penalty"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class BertRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(self.dropout(cls)).squeeze(-1)


model = BertRegressor().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"總參數量: {n_params:,}(bert-base-chinese 約 102,000,000)")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = torch.nn.MSELoss()

train_loader = DataLoader(ViolationDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ViolationDataset(val_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(ViolationDataset(test_df), batch_size=BATCH_SIZE)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        tok_type = batch.get("token_type_ids")
        tok_type = tok_type.to(DEVICE) if tok_type is not None else None
        labels = batch["label"].to(DEVICE)

        if train:
            optimizer.zero_grad()
            preds = model(input_ids, attn, tok_type)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(input_ids, attn, tok_type)
                loss = loss_fn(preds, labels)

        total_loss += loss.item() * len(labels)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    return total_loss / len(all_labels), np.array(all_preds), np.array(all_labels)


print(f"=== 開始微調 {MODEL_NAME},最多 {EPOCHS} epochs,{len(train_df)} 筆訓練資料,用val R2挑best checkpoint ===")
t0 = time.time()
best_val_r2 = -1e9
best_state = None
best_epoch = 0
for epoch in range(1, EPOCHS + 1):
    train_loss, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_preds, val_labels = run_epoch(val_loader, train=False)
    val_r2 = r2_score(val_labels, val_preds)
    print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_R2={val_r2:.3f}  耗時={time.time()-t0:.0f}s")
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch

print(f"\n最佳 val R2 在 epoch {best_epoch}(val_R2={best_val_r2:.3f}),載回該checkpoint做最終評估")
model.load_state_dict(best_state)


def report(name, loader):
    _, preds, labels = run_epoch(loader, train=False)
    mae = mean_absolute_error(labels, preds)
    rmse = mean_squared_error(labels, preds, squared=False)
    r2 = r2_score(labels, preds)
    mae_twd = np.mean(np.abs(np.expm1(preds) - np.expm1(labels)))
    print(f"[{name}] MAE(log)={mae:.3f}  RMSE(log)={rmse:.3f}  R2={r2:.3f}  約當金額誤差(TWD)={mae_twd:,.0f}")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mae_twd": float(mae_twd)}


print("\n=== 最終評估(best val checkpoint)===")
train_result = report("train", train_loader)
val_result = report("val", val_loader)
test_result = report("test", test_loader)

import json
with open("../data/results_small_regularized.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "n_params": n_params,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_val_r2": float(best_val_r2),
        "train": train_result,
        "val": val_result,
        "test": test_result,
    }, f, ensure_ascii=False, indent=2)
print("\n已儲存至 data/results_small_regularized.json")
