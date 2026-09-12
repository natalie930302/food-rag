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

### Corrective RAG:檢索結果不夠相關時,誠實拒答而不是硬答

參考 Yan et al., *"Corrective Retrieval Augmented Generation"*(arXiv:2401.15884, 2024)的核心概念:傳統 RAG 不管檢索品質好壞,一律把 top-k 結果塞給 LLM 生成答案,這是常見的幻覺(hallucination)成因之一——檢索到不相關的內容,LLM 還是會努力「掰」出一個看似合理的答案。`app/corrective_retrieval.py` 用 cross-encoder reranker 的分數當簡化版的「相關性評分器」,分數低於信心閾值就回傳「沒有足夠可信的檢索結果」,而不是硬塞低相關內容給 LLM。

**閾值是實測校準出來的,不是猜的**:`eval/tune_confidence_threshold.py` 跑了24題真實in-domain問題跟6題明顯跟食品法規無關的問題(所得稅申報、Unity動畫設定、颱風居家安全等),量到兩組分數有清楚間隔——

| | 分數範圍 |
|---|---|
| In-domain(24題) | 0.994 ~ 1.000 |
| Out-of-domain(6題) | 0.0006 ~ 0.8056 |

取中點訂閾值為 0.90。`eval/eval_corrective.py` 驗證加了這道信心閘門後:

| | 結果 |
|---|---|
| In-domain 維持信心且答對 | **24/24**(沒有因為加了把關機制而誤傷) |
| Out-of-domain 正確拒答 | **6/6**(全部正確識別為「不該自信回答」) |

這是簡化版的 CRAG(用reranker分數當評分器,沒有CRAG論文完整的網路搜尋fallback機制),但核心的「檢索結果品質把關」邏輯是一致的,而且是用真實跑出來的數字驗證過,不是紙上假設。

### 如何重現

```bash
cd eval
python eval_retrieval.py              # baseline 檢索評估
python eval_reranking.py               # baseline vs. reranking 比較
python tune_confidence_threshold.py    # 實測校準信心閾值
python eval_corrective.py              # 驗證 corrective retrieval 的把關效果
python eval_agentic.py                 # 驗證 agentic query reformulation 的效果
```

### Agentic Query Reformulation:誠實的負向/中性結果

**Agentic RAG** 是目前(2025-2026)最主流的研究方向之一(ICML 2026 workshop 光是標題含「agentic」的投稿就有60+篇)。`corrective_retrieval.py` 原本只做「被動把關」(信心不夠就拒答),`app/agentic_retrieval.py` 把它升級成「主動採取行動」:信心不足時,用 LLM(`gpt-4o-mini`)把問題換一種更正式的說法重新表述,再檢索一次——這是 Agentic RAG 最基礎的一種行為模式(query reformulation + retry),不是完整的 multi-agent 系統,誠實地說是「最小可行版本」。

用 24 題原本的評估集(baseline已經24/24信心且答對)+ 新增 8 題刻意用更口語、跟法規原文用詞差距更大的「hard」問題集測試:

| 問題集 | 正確 | 觸發retry | 因retry救回 |
|---|---|---|---|
| eval_questions.json(24題) | 24/24 | 0 | 0(預期內,確認沒有誤傷) |
| hard_questions.json(8題) | 6/8 | 1 | **0** |

**誠實記錄兩個發現,都不是我想要的結果,但都是真的跑出來的:**

1. **8題裡有1題(`如果我只是把東西重新分裝,不算是真正在做食品加工吧?`)是「confidently wrong」**——reranker給了很高的信心分數,但答案是錯的。這代表信心閘門機制有一個本質限制:它評分的是「檢索到的內容看起來像不像相關」,不是「答案對不對」,兩者不完全等價。這個問題目前的機制**偵測不到**,agentic retry完全不會被觸發,不是這次改動能解決的。
2. 唯一真的觸發retry的那一題(`紅麴膠囊這種東西,官方是怎麼歸類的?`),LLM把它改寫成更正式的「紅麴膠囊在官方法規中屬於何種產品類別？」,但重新檢索後信心分數還是偏低,**沒有救回來**。診斷原因:gold chunk 的內容主體是「什麼是營養補充食品」的一般性定義,紅麴膠囊只是文中順帶提到的其中一個例子——問題不在於問法夠不夠正式,是這個chunk的語意重心本來就不在「紅麴膠囊」本身,單純換句話說沒辦法解決這種「答案藏在較大範疇定義裡的一個例子」的檢索粒度問題,需要更根本的作法(例如更細的chunking策略,或是先做entity extraction再檢索)。

跟 Portfolio 裡其他負向結果一樣的教訓:**不是每個聽起來合理的改進方向都真的有用,誠實驗證比預設會成功更重要**。這個方向本身(agentic retry)架構上是安全的(沒有誤傷原本答對的問題),但這次具體驗證的「LLM重新表述問題」這個corrective action,在小樣本測試中沒有展現出效果——如果要繼續往這個方向做,下一步應該是先解決chunking粒度問題,而不是繼續在同一個chunk結構上做更多次retry。

### 在改 chunker 之前,先量這個問題有多普遍

「答案藏在定義段落列舉的例子裡」聽起來像個值得解決的問題,但改 chunking 策略是整個 ingest pipeline 最貴的改動(要重新設計切分邏輯、重跑全部 16,119 個 chunks 的 ingest)。在動手之前,先用一個簡單的啟發式規則掃過現有語料庫,量測這個模式到底有多普遍,而不是憑一個案例就直接動工:抓「定義用語」(係指、所稱、指下列等)同時出現「長列舉」(如/例如/包括後面接4項以上用頓號分隔的例子)的 chunk。

**結果:16,119 個 chunks 裡只有 118 個(0.73%)符合這個模式**,而且這類 chunk 的平均長度(403字)跟全體平均(377字)差不多,不是特別長、特別容易「塞進太多資訊」的異常值。

**誠實的結論**:這不是一個結構性、大範圍的問題,是一個窄範圍的邊界案例——重新設計整個 chunking 策略去解決一個影響不到1%語料的問題,投報率不高。這個啟發式規則本身也有限制(只抓得到用頓號列舉例子這種表面形式,抓不到用其他句型「藏」例子的情況,所以0.73%可能是低估),但已經足以支持一個判斷:**現階段不值得為此重寫 chunker**,先記錄成已知限制,之後如果要花更多時間,更該做的是設計一組更大的 hard 問題集,先確認這類問題實際出現的頻率是不是真的邊緣案例,再決定要不要動 chunking 邏輯。

## 研究歷程

### 研究動機與路徑

`tests/` 底下原本只有 API 層的 plumbing 測試(狀態碼、回傳格式對不對),從沒量過「檢索有沒有真的撈到對的內容」——這是這輪補強的起點。先建立 24 題自然提問的評估集跟量測管線,量出 baseline 其實已經不差(Recall@1=0.708),於是研究路徑分三步往下走:

1. **在已經不錯的 baseline 上,現在業界標準的兩階段架構(dense retrieval + cross-encoder reranking)是否還有提升空間?**→ 驗證結果是肯定的,全面提升
2. **檢索到不相關內容時,系統該不該誠實拒答,而不是硬答?**→ 參考 Corrective RAG,用校準過的信心閾值做把關,兩組問題(in-domain/out-of-domain)都拿到滿分
3. **信心不夠時,系統能不能主動採取行動補救,而不是只會拒答?**→ 這是刻意呼應「Agentic AI 是目前最主流研究方向」這個時勢判斷去補的方向,做了 query reformulation + retry 的最小可行版本,新增 8 題刻意口語化的 hard 問題集去逼近系統的真實極限,而不是繼續在已經 24/24 的簡單題目上驗證

第 3 步的結果不如預期(0/1 成功救回),但診斷根因指向 chunk 語意粒度問題,這比「假裝有效」更有價值——它把下一步該往哪裡去(chunking 策略,而不是 retry 次數)講清楚了。

### 研究方法

- **兩階段檢索架構驗證**:比照 production RAG 系統的標準做法(bi-encoder 全庫召回 + cross-encoder 精排),用同一組真實 API 路徑(`retrieve_chunks()`)量測,避免另外寫一套簡化邏輯量出灌水的數字
- **閾值用分布間隔實測校準,不是猜的**:`tune_confidence_threshold.py` 分別測 in-domain(24題)跟 out-of-domain(6題)兩組分數分布,取有清楚間隔的中點,而非拍腦袋設一個看起來合理的數字
- **用「故意設計來考倒系統」的問題集驗證極限**:`hard_questions.json` 8題刻意用更口語、跟法規原文用詞差距更大的問法,目的是在系統已經對簡單題目滿分的情況下,找出它真正的失敗模式,而不是重複驗證已知會過的案例
- **失敗後往根因追,不是停在「沒救回來」**:唯一觸發retry的那題沒被救回來時,直接回頭比對 gold chunk 的實際內容,發現答案是嵌在較大範疇定義裡的一個例子,才確認問題出在chunk粒度而非問法正式與否

### 遇到的困難

- **信心閘門的本質限制,不是這次能解決的**:8題hard問題裡有1題是reranker給高分但答案錯的「confidently wrong」,retry完全不會被觸發——這暴露出「檢索內容看起來像不像相關」≠「答案對不對」,是confidence-based方法本身的天花板,誠實記錄下來而不是回頭修改評估方式讓數字好看
- **agentic retry沒有帶來預期中的提升**:一開始預期reformulation至少能救回一部分case,實際只有1題觸發、0題救回——沒有回頭調整hard_questions.json的題目難度讓結果好看,而是把「為什麼沒救回來」的根因分析寫清楚

### 時程(依實際執行順序)

1. 補 24 題評估集 + `retrieve_chunks()` baseline 檢索評估
2. Cross-Encoder Reranking 驗證(正向)
3. Corrective RAG:閾值校準 + in/out-of-domain 驗證(正向)
4. Agentic Query Reformulation:新增 8 題 hard 問題集 + 驗證(誠實負向,根因診斷出chunk粒度問題)
5. 動手改 chunker 前,先用啟發式規則量測 chunk 粒度問題的普遍程度(0.73%,窄範圍邊界案例)→ 判斷現階段不值得重寫 chunking 邏輯

## 文件

詳細技術說明見規劃文件。
