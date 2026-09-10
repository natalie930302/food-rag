"""
微調中文 BERT(bert-base-chinese)做迴歸,預測 log(罰款金額)。
跟 baseline_tfidf.py 的結果比較,驗證微調是否真的帶來進步。
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 4
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {DEVICE}")

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
        self.dropout = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        return self.head(self.dropout(cls)).squeeze(-1)

model = BertRegressor().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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

print(f"=== 開始微調 {MODEL_NAME},{EPOCHS} epochs,{len(train_df)} 筆訓練資料 ===")
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    train_loss, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_preds, val_labels = run_epoch(val_loader, train=False)
    val_r2 = r2_score(val_labels, val_preds)
    print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_R2={val_r2:.3f}  耗時={time.time()-t0:.0f}s")

def report(name, loader):
    _, preds, labels = run_epoch(loader, train=False)
    mae = mean_absolute_error(labels, preds)
    rmse = mean_squared_error(labels, preds, squared=False)
    r2 = r2_score(labels, preds)
    mae_twd = np.mean(np.abs(np.expm1(preds) - np.expm1(labels)))
    print(f"[{name}] MAE(log)={mae:.3f}  RMSE(log)={rmse:.3f}  R2={r2:.3f}  約當金額誤差(TWD)={mae_twd:,.0f}")

print("\n=== 最終評估 ===")
report("train", train_loader)
report("val", val_loader)
report("test", test_loader)

torch.save(model.state_dict(), "../data/bert_regressor.pt")
print("已儲存微調後模型")
