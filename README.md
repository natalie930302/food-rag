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
├── prompts/         # Prompt 模板
├── scripts/         # 一次性工具
└── tests/           # 測試
```

## 文件

詳細技術說明見規劃文件。
