"""
Baseline: 字元級 TF-IDF + Ridge 迴歸,預測 log(罰款金額)。
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = pd.read_csv("../data/train.csv")
val = pd.read_csv("../data/val.csv")
test = pd.read_csv("../data/test.csv")

# 中文沒有天然分詞,用字元 n-gram(2-4字)當特徵,避免額外依賴分詞工具
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=5000)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])
X_test = vectorizer.transform(test["text"])

y_train = train["log_penalty"].values
y_val = val["log_penalty"].values
y_test = test["log_penalty"].values

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

def evaluate(name, X, y):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred, squared=False)
    r2 = r2_score(y, pred)
    # 換算回原始金額尺度的平均誤差(元)
    mae_twd = np.mean(np.abs(np.expm1(pred) - np.expm1(y)))
    print(f"[{name}] MAE(log)={mae:.3f}  RMSE(log)={rmse:.3f}  R2={r2:.3f}  約當金額誤差(TWD)={mae_twd:,.0f}")
    return mae, r2

print("=== Baseline: TF-IDF(char 2-4gram) + Ridge ===")
evaluate("train", X_train, y_train)
evaluate("val", X_val, y_val)
evaluate("test", X_test, y_test)

joblib.dump(model, "../data/baseline_model.joblib")
joblib.dump(vectorizer, "../data/baseline_vectorizer.joblib")
print("已儲存 baseline 模型")
