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
python ../scripts/extract_chunk_entities.py  # 從長列舉段落抽取實體詞,建立entity boost索引
python compare_reranker_models.py      # 比較 bge-reranker-base vs. v2-m3 的 rank-1 準確率
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

### 針對這個窄範圍問題做局部修補,不重寫 chunker(2026/09新增)

上面判斷「不值得重寫chunker」,不代表這個問題就放著不管——如果修補成本夠低,還是值得做。做法是 `scripts/extract_chunk_entities.py` + `app/entity_boost.py`:從長列舉段落(`如/例如/包括` 後接3項以上頓號分隔的例子)抽出列舉的具體名詞,建立「名詞→chunk_id」的關鍵詞對照表。檢索時如果問題包含這些名詞,就把對應的chunk額外加進候選池,交給既有的cross-encoder reranker跟信心閘門去判斷該不該用——**這一層只負責「不要漏掉候選」,不繞過任何品質把關**,即使關鍵詞比對抓到不相關的chunk,一樣會被reranker評低分。

**做的時候發現量測用的偵測規則本身低估了問題範圍**:原本量測「這個問題有多普遍」時,用「定義用語(係指/所稱等)+ 長列舉」兩個條件一起卡,量出0.73%(118個chunks)。但拿真正失敗的案例(紅麴膠囊,chunk 14031)回頭測,才發現它的原文用的是「包括**但不限於**」,不在原本設定的定義用語清單裡,被漏掉了——量測時的偵測條件其實是保守估計的下界。做修補時拿掉「定義用語」這個條件,只憑「長列舉」抽取,範圍變成 820 個chunks(約5.1%語料),抽出1,879個候選名詞。**這是先前用來量測「值不值得修」的規則,不等於用來「做修補」時該用的規則**——前者要保守以免高估問題嚴重性,後者可以偏寬鬆,因為抓錯了也有reranker把關,成本很低。

**結果:部分修復,誠實記錄修好的部分跟沒修好的部分**:

| 問題集 | 正確 | 觸發retry | 因retry救回 |
|---|---|---|---|
| eval_questions.json(24題) | 24/24(無變化) | 0 | 0 |
| hard_questions.json(8題) | **7/8**(原本6/8) | 1 | **1**(原本0) |

紅麴膠囊那題現在確實被「救回來」了,但精確追查機制發現這不是一個乾淨的勝利:entity boost 把 chunk 14031(正確答案)加進候選池後(原本連候選池的前10名都排不進去),搭配 agentic retry 的改寫問題,cross-encoder reranker 給 chunk 14031 的分數是 0.33,**仍然遠低於**另一個提到「紅麴」但實際上是講紅麴製品倉儲規範、答非所問的干擾chunk(分數0.99,排名第一)。系統回報「hit=True」的原因是評估用的是 **Recall@5**(gold chunk有沒有出現在回傳的前5筆裡),而不是「排名第一的是不是正確答案」——chunk 14031 排在第5名,勉強擠進回傳範圍,但reranker真正「相信」的答案排名第一其實還是錯的。

**誠實的定位**:這次修補解決的是「候選池根本找不到正確答案」這個問題(候選生成階段的瓶頸),但沒有解決「reranker能不能正確判斷哪個候選才是真正相關」這個更深層的問題(排序品質階段的瓶頸)。這個排序品質問題往下追,牽出了下一個章節的發現。

### reranker 模型升級:base 對這類問題有詞義混淆,換模型後全面改善(2026/09新增)

往下追查「排序品質」這個瓶頸,發現真正原因不是原本以為的chunk粒度/稀釋問題,而是現有的 `bge-reranker-base` 模型本身對這類查詢有詞義混淆:紅麴膠囊那題,正確答案(chunk 14031,講「營養補充食品」定義)只拿到0.07分,但一個講「紅麴製品應**分類**分區存放」的干擾chunk(講倉儲規範,答非所問)反而拿到0.84分——reranker把「分類存放」的「分類」跟查詢裡「怎麼**歸類**」的「分類」搞混了,這是語意層面的混淆,不是chunk切得不好。

換成更強的 `bge-reranker-v2-m3` 後,同一組候選,正確答案分數翻盤到0.05,干擾chunk掉到0.0005——順序整個對了。為了確認這不是單一案例湊巧,寫了 `eval/compare_reranker_models.py`,用完全相同的候選池,兩個模型分別對全部32題(24easy+8hard)做rank-1準確率比較:

| Reranker | Rank-1 準確率 |
|---|---|
| bge-reranker-base | 22/32 = 0.688 |
| **bge-reranker-v2-m3** | **28/32 = 0.875** |

7題從錯翻對,只有1題從對翻錯——是全面性的改善,不是單一案例的巧合。換模型後,重新校準信心閾值(`tune_confidence_threshold.py`):新模型的in-domain/out-of-domain分數間隔比舊模型更乾淨(gap=0.925,舊模型是0.188),閾值從0.90調整為0.52。全部下游評估也重新跑過:

| 評估 | 換模型前 | 換模型後 |
|---|---|---|
| Reranking(Recall@1 / MRR) | 0.750 / 0.861 | **0.917 / 0.948** |
| Corrective RAG(in-domain / out-of-domain) | 24/24 / 6/6 | 24/24 / 6/6(無退步) |
| Agentic RAG(hard問題正確數) | 6/8 | 6/8(組成改變,見下方) |

### 一個更深的發現:排序對了,信心閘門還是不一定會通過

換了reranker、確認排序邏輯修好之後,重新驗證紅麴膠囊這題,發現它**還是沒有被救回來**——但這次的原因跟之前完全不同,而且更精確。直接檢查發現:用新reranker,chunk 14031 確實在候選池裡排名第一(這點已經修好),但它的**絕對分數只有0.153**,還是低於重新校準後的閾值0.52。

這揭露了信心閘門機制一個結構性的盲點:**「候選池裡排名第一」不等於「絕對分數夠有信心」**。chunk 14031 是一個列了30幾種不同「營養補充食品」品項的大定義段落,紅麴膠囊只是其中一個例子——不管reranker多強,只要整個chunk的內容有95%在講其他不相關的品項,cross-encoder算出來的「這個chunk整體跟查詢的相關程度」分數就會被稀釋,即使它相對其他候選是最好的選擇,絕對分數還是拉不上來。換句話說,信心閾值假設的是「正確答案應該要有高分」,但對這種「正確資訊被淹沒在一個大雜燴列舉段落裡」的內容,這個假設本身就不成立。

原本猜測的根因是「chunk粒度」,追到這裡才發現更精確的描述是「**信心閘門用絕對分數判斷,沒辦法反映『這是候選裡最好的選擇』跟『這個選擇本身夠不夠格』是兩件不同的事**」——這是比原本的診斷更深一層,而且指出如果要真正解決,方向不是換reranker(已經換了、也確實有全面性幫助),而是信心判斷的機制本身需要改成相對排名(這個候選比其他候選好多少)而不是純絕對分數,或者還是要回到最早提過的細粒度chunking這條路。另一題「食品安全管制窗口」課程資格問題,反而在這次升級後意外被retry救回來(6/8整體持平,但組成不同)。confidently wrong的重新分裝案例仍未觸及,是完全不同的失效模式。

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
6. 針對這個窄範圍問題做低成本局部修補(entity boost)→ Recall@5 從6/8回升到7/8,但精確追查發現是候選池問題解決了、排序品質問題還在,誠實記錄部分修復的範圍
7. 往下追排序品質問題,診斷出是reranker模型本身的詞義混淆,不是chunk粒度問題 → 換更強的reranker(bge-reranker-v2-m3),32題rank-1準確率系統性驗證從0.688提升到0.875,重新校準信心閾值、重跑全部下游評估
8. 換了reranker後紅麴膠囊那題還是沒被救回來 → 精確診斷出更深一層的原因:排名對了(rank-1)但絕對分數還是太低,信心閘門機制本身無法反映「相對最好」跟「絕對夠格」的差異,把原本「chunk粒度問題」的診斷修正得更精確

## 文件

詳細技術說明見規劃文件。
