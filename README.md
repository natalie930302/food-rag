# 食品法規 RAG 問答系統

基於食藥署法規與北市違規廣告案例的問答 API,使用 LlamaIndex + FAISS + SQLite + OpenAI。

## 特色

- 🌐 **混合架構**:本地 embedding(BGE-M3)+ 雲端 LLM(GPT-4o)
- 📂 **全格式支援**:PDF / DOC / DOCX / TXT 都能處理
- 🔍 **OCR 處理掃描檔**:Tesseract 中文繁體
- 📊 **表格抽取與檢查**:PDF 表格自動轉 Markdown
- 🕸️ **法條網狀關聯**:支援「重組肉 Q1 一次引用 4 條法條」這種聯合查詢
- ❌ **失敗檔追蹤**:無法處理的檔案會標註但不丟失,API 可告知使用者

## 系統需求

| 項目 | 最低 | 建議 |
|---|---|---|
| Python | 3.11 | 3.11 / 3.12 |
| RAM | 8 GB | 16 GB |
| 磁碟 | 5 GB | 10 GB |
| OS | Linux / macOS / WSL2 | 同左 |

## 安裝

### 1. 系統依賴

**Ubuntu / WSL2**:
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv \
    tesseract-ocr tesseract-ocr-chi-tra \
    libreoffice poppler-utils
```

**macOS**:
```bash
brew install python@3.11 tesseract tesseract-lang libreoffice poppler
```

### 2. 專案安裝

```bash
git clone <your-repo> food-rag
cd food-rag
make install
source .venv/bin/activate

cp .env.example .env
# 編輯 .env 填入 OPENAI_API_KEY
```

## 使用

### 1. 準備資料

把三個 zip 檔放到 `data/raw/zips/`:
```
data/raw/zips/
├── 食藥署.zip
├── 台北市政府公告114年違規廣告.zip
└── 台北市政府公告115年違規廣告.zip
```

執行:
```bash
make unzip
```

### 2. 跑 Ingest

```bash
make ingest
```

完成後 `data/index/` 下會有:
- `chunks.db`(SQLite)
- `faiss_chunks.index`(法規向量)
- `faiss_cases.index`(案例向量)

第一次跑會下載 BGE-M3 模型(~2.3 GB),需時較久。

### 3. 啟動 API

```bash
make run
```

瀏覽器開 `http://localhost:8000/docs` 看互動文件。

## API 範例

```bash
# 一般問答
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "真空包裝豆干要符合什麼規定?"}'

# 廣告審稿
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"ad_text": "本產品有效改善高血壓"}'

# 法條關聯
curl http://localhost:8000/laws/食安法第28條/related

# 失敗檔清單
curl http://localhost:8000/failed
```

## 專案結構

```
food-rag/
├── config/          # 設定
├── data/            # 資料(raw/converted/processed/index)
├── ingest/          # 攝取 pipeline
│   ├── parsers/     # 各格式解析器
│   ├── chunker/     # 切碎策略
│   └── extractors/  # 法條偵測等
├── app/             # FastAPI 應用
├── eval/            # 檢索品質評估(見下方)
├── prompts/         # Prompt 模板
├── scripts/         # 一次性工具
└── tests/           # 測試
```

## 檢索品質評估(2026/09 新增)

`tests/` 底下原本只有 API 層的 plumbing 測試(狀態碼、回傳格式對不對),沒有量測過「檢索有沒有真的撈到對的內容」——這是 RAG 系統最關鍵、卻最容易被跳過驗證的一環。

- `eval/eval_questions.json`:從真實已索引的 17,152 個 chunks 裡,挑 24 題涵蓋不同主題(標示規定、添加物登錄、檢驗週期、追溯系統、裁罰基準等)的內容,**用改寫過的自然提問方式**(不是直接複製索引文字)當查詢,避免文字表面重疊讓 Recall 虛高
- `eval/eval_retrieval.py`:直接呼叫 `app/retrieval.py` 裡正式環境在用的 `retrieve_chunks()`,量到的數字反映的是真實部署的檢索品質,不是另外寫一套簡化邏輯

### Baseline 結果(BGE-M3 dense retrieval,無 metadata 過濾)

| Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| 0.708 | 0.958 | 1.000 | 0.830 |

### 加上 Cross-Encoder Reranking 後

`eval/eval_reranking.py` 在 dense retrieval 的初篩結果(top-10)上,加一層 `BAAI/bge-reranker-base`(跟現有的 `BAAI/bge-m3` embedding 同團隊發布)重新排序——這是近年 production RAG 系統的標準兩階段做法:bi-encoder(query 和文件各自獨立編碼)負責快速從全庫撈候選,cross-encoder(query 和文件當同一個輸入一起編碼,能做 token 級別的交互注意力)負責在小範圍候選裡精排,犧牲不能預先索引全庫的代價換取更高的排序精度。

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Baseline(僅dense retrieval) | 0.708 | 0.958 | 1.000 | 0.830 |
| **+ Cross-Encoder Reranking** | **0.750** | **1.000** | 1.000 | **0.861** |

真實、正向的結果:Recall@1 提升 4.2 個百分點、Recall@3 到 100%、MRR 提升 3.1 個百分點。這跟 Portfolio 裡其他幾個「微調反而讓結果變差」的負向案例不同——這裡驗證的是「在已經很強的 baseline 之上,加一個現在業界/學界標準的兩階段檢索架構是否真的有幫助」,結果是肯定的。

### 如何重現

```bash
cd eval
python eval_retrieval.py    # baseline 檢索評估
python eval_reranking.py    # baseline vs. reranking 比較(會下載bge-reranker-base)
```

## 文件

詳細技術說明見規劃文件。
