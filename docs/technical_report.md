# 食品法規領域的檢索增強生成:兩階段檢索、信心閘門與有邊界的 Tool-Calling Agent 之量化評估

**Retrieval-Augmented Generation for Taiwanese Food Regulations: A Quantitative Study of Two-Stage Retrieval, Confidence Gating, and a Bounded Tool-Calling Agent**

許安婷(An-Ting Hsu)· 2026 年 9 月 · 專案:github.com/natalie930302/food-rag

---

## 摘要

本報告以食藥署法規/指引/問答集(16,119 個 chunks)與台北市違規廣告裁罰公告(400 筆)為語料,建構一個食品法規問答系統,並針對三個常被「預設有效」的設計做量化驗證:(1) dense retrieval 之上加 cross-encoder reranking 的兩階段架構;(2) 以 reranker 分數做信心閘門的 Corrective RAG;(3) 讓 LLM 自主決定檢索策略的 tool-calling agent 及其 harness 邊界。評估集從 24 題手寫擴大到 108 題(84 題為 LLM 生成、人工審查),另有 8 題刻意口語化的 hard set、20 題 out-of-domain、20 題 multi-hop;所有指標附 95% bootstrap 信賴區間,配置間差距以 paired bootstrap 與精確符號檢定判定。主要發現:`bge-reranker-base` 在 24 題時看似提升 Recall@1(0.708→0.750),在 108 題上無顯著差異(Δ=+0.019,p=0.815);換成 `bge-reranker-v2-m3` 才有顯著提升(Δ=+0.120 [+0.046, +0.194],p=0.004)。信心閾值在 30 題上的「乾淨間隔」在 128 題上消失,沒有任何閾值能同時零誤拒、零誤放。tool-calling agent 在單跳問題上與固定重試打平(99 vs 100/108),在為其設計的 20 題多步問題上亦未勝出(8 vs 10,p=0.688),唯一無可取代的是關聯法條工具;harness 四道邊界以消融實驗各自量化,僅字面錨定 drift 檢查有可量測效果(關閉後 31→27/32),grounded 強制覆寫與答案層引用驗證在本評估集上未攔到任何案例。本報告刻意保留所有負向結果與被推翻的早期結論,並記錄由多步題組暴露的一個潛藏缺陷(案例檢索誤用索引)。

**關鍵字**:Retrieval-Augmented Generation、Cross-Encoder Reranking、Corrective RAG、Agentic RAG、Agent Harness、Bootstrap Confidence Interval、Evaluation Methodology

---

## 1. 緒論

### 1.1 動機

食品業者面對的法規文件龐雜:法條本文、施行細則、主管機關公告、指引、問答集,彼此互相引用,且大量以掃描 PDF 存在。RAG 是處理此類「答案存在於文件、但文件太多」問題的自然選擇,但近年 RAG 系統的改進方向(reranking、corrective gating、agentic retrieval)多以「聽起來合理」為前提被採用,實際在特定領域、特定語料上是否有效,往往缺乏對照與統計檢定。本專案的核心不是功能,而是**對每個設計決策做可重現的量化驗證,包括驗證失敗時如實記錄**。

### 1.2 研究問題

1. 在已具備一定水準的 dense retrieval 之上,cross-encoder reranking 是否有統計上可辨識的提升?提升來自「兩階段架構」還是「reranker 模型本身」?
2. 以 reranker 分數做信心閘門,能否可靠區分 in-domain 與 out-of-domain 問題?閾值校準對樣本數有多敏感?
3. 讓 LLM 自主決定檢索策略(tool-calling agent),相較於寫死的重試邏輯,在什麼條件下有幫助、什麼條件下沒有?
4. agent harness 的每道邊界(預算、grounded 強制、drift 檢查、引用驗證)各自擋掉了什麼?

### 1.3 貢獻

- 一套完整可重現的評估管線:108 題主評估集(手寫/合成分開報告)、hard set、out-of-domain、multi-hop 四組題目,`eval/stats.py` 提供 bootstrap CI、paired bootstrap、精確符號檢定,`make eval` 一鍵產出 `eval/RESULTS.md`
- 兩個被大樣本推翻的早期結論(reranker-base 的「提升」、信心閾值的「乾淨間隔」),以及推翻的過程
- 一個 tool-calling agent harness,其四道邊界皆以程式碼強制、皆可個別關閉以做消融
- 答案層引用驗證(citation grounding check):信心閘門之外的第二道防線,針對「檢索內容可信但 LLM 引錯條號」

---

## 2. 相關工作

**兩階段檢索。** Bi-encoder(如 BGE-M3 [1])將 query 與文件獨立編碼,可預先索引全庫;cross-encoder(如 bge-reranker [2])將 query 與文件串接後聯合編碼,能做 token 級交互注意力,精度較高但無法預先索引,故標準做法是 bi-encoder 召回、cross-encoder 精排。本報告的問題不是「該不該用」,而是「在這個語料上差多少、差距是否顯著」。

**Corrective RAG。** Yan et al. [3] 提出以輕量評估器對檢索結果評分,分為 Correct / Incorrect / Ambiguous 後採取不同行動(直接用、網路搜尋補救、混合)。本專案取其核心——「不把低品質檢索結果原封不動塞給 LLM」——以 reranker 分數做簡化版評估器,不含網路搜尋 fallback。

**Agentic RAG。** Singh et al. [4] 的 survey 將 agentic RAG 分為 single-agent、multi-agent、hierarchical 等模式。本專案的 `/ask_agent` 屬最基礎的 single-agent tool-calling loop;其重點在 harness 設計(邊界由程式碼強制)而非 agent 架構的複雜度。

**Self-consistency。** Wang et al. [5] 以多次取樣的一致性判斷答案可信度。本專案的「字面錨定 drift 檢查」借用同一精神,但比較的是「LLM 改寫的查詢」與「原始問題字面」兩條檢索路徑的 top-1 是否一致。

**引用驗證。** 對 LLM 輸出的引用做事後檢查是 attributed QA 的常見做法 [6];本專案限縮為可機械判定的「引用條號是否出現在提供的 context 中」,不引入第二個 LLM 當裁判。

---

## 3. 系統

### 3.1 資料與 ingest

語料來自食藥署公開文件(PDF / DOC / DOCX / TXT,含掃描檔)與台北市政府 114–115 年違規廣告裁罰公告。Ingest 管線:格式解析(pdfplumber、python-docx、LibreOffice 轉檔、Tesseract 繁中 OCR、表格轉 Markdown)→ 文字正規化 → 四策略 chunker 路由(`#AI資料庫` 分隔格式 / Q&A 格式 / 公文結構標題 / 長度切分)→ 法條引用偵測(多對多 `chunk_laws` 關聯,含 primary / penalty / definition / reference 角色)→ BGE-M3 編碼 → SQLite + FAISS `IndexFlatIP`。最終 16,119 個 chunks(平均 377 字;qa 2,495、guide 12,602、standard 989、law_text 33)與 400 筆案例。

### 3.2 `/ask`:確定性管線

1. **路由**:問題含廣告/宣稱關鍵字 → 鎖定食安法第 28 條範圍;否則用使用者 filters
2. **Dense retrieval**:SQL metadata 預過濾 → FAISS top-10
3. **Entity boost**:從長列舉段落抽出的 1,879 個名詞建立「名詞 → chunk」對照,問題命中即併入候選池(只擴池,不繞過後續把關)
4. **Cross-encoder rerank**:`bge-reranker-v2-m3`
5. **信心閘門**:top-1 分數 < 0.52 → 用 LLM 將問題改寫成正式用語重查一次;仍不足 → 回覆固定的誠實拒答訊息,不呼叫 LLM
6. **生成**:gpt-4o-mini
7. **答案層引用驗證**:答案裡每個「第 N 條」須出現在提供的 chunk 內文或 metadata;否則帶回饋重生成一次,仍不符則於 `meta.unsupported_citations` 如實回報

### 3.3 `/ask_agent`:tool-calling harness

同一套檢索包成三個工具——`search_regulations`(含信心閘門,回傳 `confident` 與分數)、`search_violation_cases`、`search_related_laws`(法條共現統計)——交給 OpenAI function-calling loop,由 LLM 決定查幾次、查什麼。Harness 以程式碼強制四道邊界,每一道皆可個別關閉以供消融:

| 邊界 | 機制 | 針對的失效模式 |
|---|---|---|
| 預算 | 工具呼叫次數、累計 token、牆鐘秒數任一超過 → 關閉工具、強制作答,`usage.stop_reason` 記錄原因 | 重複查詢燒費用、長尾延遲 |
| grounded 強制 | 全程無任何 `confident=True` → 不論 LLM 說什麼,覆寫為拒答 | LLM 無視 prompt 硬答 |
| 字面錨定 drift 檢查 | 每次 `search_regulations` 額外用原始問題字面檢索一次(純本地);兩者 top-1 不一致且字面版有信心 → 改用字面版 | LLM 濃縮的關鍵字查詢語意飄移、查到高信心但錯的內容 |
| 答案層引用驗證 | 同 3.2 第 7 步 | 檢索內容可信但引錯條號 |

決策 temperature 固定為 0(早期發現 0.1 會讓邊界名次附近的題目重跑結果不一致)。每步記錄為 `ToolCallRecord`(工具、參數、信心、chunk ids、是否觸發 drift 介入),`scripts/replay_trace.py` 可離線逐步回放。

### 3.4 `/query`:先分流,再決定用多重的機制

5.3 節顯示 agent 在單跳問題上與固定管線同樣準確但慢一倍,因此單一入口 `/query` 採 Adaptive-RAG [8] 的原則:先判斷意圖,只把需要多步檢索的問題交給 agent。路由分兩層——關鍵字規則(零成本、確定性,只在訊號很強時判定)→ gpt-4o-mini 結構化 JSON 分類(temperature 0,規則判不出來才呼叫)。四類意圖:`regulation_qa`(→ `/ask`)、`case_lookup`(→ `/ask` + 強制查案例)、`ad_review`(→ `/review`)、`multi_hop`(→ `/ask_agent`)。路由決定與理由隨回應回傳。

---

## 4. 評估方法

### 4.1 題目

| 題組 | n | 來源 | 用途 |
|---|---|---|---|
| 主評估集 manual | 24 | 手寫,改寫過的自然提問 | 檢索品質 |
| 主評估集 synthetic | 84 | gpt-4o-mini 從隨機 chunk 生成 140 題 → 自動過濾(長度、與原文最長共同子字串 ≤ 6 字、排除近似重複 chunk)→ 人工審查剔除 56 題 | 檢索品質;與 manual 分開報告 |
| hard | 8 | 手寫,刻意口語、與法規用詞差距大 | 找失效模式 |
| out-of-domain | 20 | 手寫,含 4 題「近域陷阱」(化妝品標示、寵物飼料標示等) | 信心閘門 |
| multi-hop | 20 | 手寫,需法規 + 案例或法規 + 關聯法條 | agent 何時有用 |

人工審查的每一個決定(保留哪幾題、剔除理由)寫死在 `eval/build_eval_v2.py`,可完整重現。

### 4.2 指標與統計

Recall@k、MRR;信心閘門用誤拒率/誤放率;agent 用「gold chunk 是否在最終拿去回答的 chunk 裡」(hit)。所有比例與平均附 95% percentile bootstrap CI(10,000 次重抽)。配置間比較在同一組題目上做 paired bootstrap(消除題目難度變異),二元結果另做精確符號檢定(只計不一致的題目)。

---

## 5. 結果

### 5.1 兩階段檢索(RQ1)

| 配置 | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| dense only | 0.713 [0.630, 0.796] | 0.861 [0.796, 0.926] | 0.796 [0.732, 0.858] |
| + reranker-base | 0.731 [0.648, 0.815] | 0.935 [0.889, 0.972] | 0.826 [0.767, 0.883] |
| + reranker-v2-m3 | **0.833 [0.759, 0.898]** | 0.907 [0.852, 0.954] | **0.874 [0.818, 0.926]** |

| 比較(Recall@1) | Δ [95% CI] | 翻對/翻錯 | p |
|---|---|---|---|
| dense → base | +0.019 [−0.056, +0.093] | 10 / 8 | 0.815 |
| base → v2-m3 | +0.102 [+0.037, +0.176] | 13 / 2 | 0.007 |
| dense → v2-m3 | +0.120 [+0.046, +0.194] | 16 / 3 | 0.004 |

「兩階段架構」本身不是提升來源;`reranker-base` 翻對 10 題、翻錯 8 題。提升來自 reranker 模型品質。這個結論與早期 24 題的紀錄相反——當時 0.708→0.750 被解讀為有效,實為 1 題之差。手寫題與合成題趨勢一致;合成題 Recall@1 較低(0.810 vs 0.917),表示 LLM 生成題並未讓評估變簡單。

### 5.2 信心閘門(RQ2)

24+6 題校準時 in-domain 最低分 0.984、out-of-domain 最高分 0.059,gap +0.925,閾值取中點 0.52。108+20 題重跑:in-domain 最低 0.068、out-of-domain 最高 0.400,gap −0.332。

| 閾值 | in-domain 誤拒 | out-of-domain 誤放 |
|---|---|---|
| 0.52 | 8/108 | 0/20 |
| 0.068 | 0/108 | 2/20 |

誤拒的 8 題中 4 題答案位於表格列(農藥殘留、檢驗費用),cross-encoder 對表格片段的相關性評分本來就偏低;誤放的 2 題為近域陷阱(房屋租賃契約公證 0.40、化妝品標示 0.24)。「乾淨間隔」是 30 題的抽樣假象。現行保留 0.52 為「寧拒答勿硬答」的產品取向。

端對端(閾值 0.52):in-domain 維持信心且撈到 gold 96/108 = 0.889 [0.824, 0.944](手寫 24/24、合成 72/84);誤拒 8 題;confidently wrong 4 題(閘門給高分但未撈到正確 chunk,confidence-based 方法的結構性盲點)。out-of-domain 20/20 正確拒答,含 4 題近域陷阱。

### 5.3 固定重試 vs. tool-calling agent(RQ3)

| 問題集 | 固定重試 | tool-calling agent | 翻對/翻錯 | p |
|---|---|---|---|---|
| 主評估集(108) | 100/108 = 0.926 [0.870, 0.972] | 99/108 = 0.917 [0.861, 0.963] | 0/1 | 1.000 |
| hard(8) | 6/8 | 6/8 | 0/0 | 1.000 |

固定重試在 108 題上 8 題觸發、4 題救回,推翻早期 8 題 hard set「0 題救回」的負向結論。agent 在單跳問題上與固定重試無差異(108 題僅 3 題使用 >1 次工具;drift 檢查介入 25 題),延遲 31.5 秒/題(固定管線約 13 秒)。早期 32 題上「31/32 > 30/32」的結論在 108 題上不成立。兩系統各有 confidently wrong 案例(固定重試 5+2 題)。

### 5.4 Harness 消融(RQ4)

| 配置 | in-domain 命中(32) | OOD 沒依據卻硬答(20) | 秒/題 |
|---|---|---|---|
| full | 31/32 = 0.969 [0.906, 1.000] | 0/20 | 18.6 |
| no_grounding | 30/32 = 0.938 [0.844, 1.000] | 0/20 | 19.3 |
| no_drift_check | 27/32 = 0.844 [0.719, 0.969] | 0/20 | 11.4 |
| temp_0.1 | 31/32 = 0.969 [0.906, 1.000] | 0/20 | 18.3 |

字面錨定 drift 檢查是唯一在本組題目上有可量測效果的邊界(關掉掉 4 題,代價每題約 7 秒)。grounded 強制覆寫在本組題目上多餘:關掉後 gpt-4o-mini 對 20 題 OOD 全部自行以不同措辭拒答(第一版腳本以字面比對拒答句而誤計 19/20,改為語意判斷後為 0/20);此邊界的價值是保證而非可量測的提升,保留但如實標示。temperature=0 的效果需重複執行才能觀察,單次消融無差異。

### 5.5 Multi-hop

20 題需要「法規 + 案例」或「法規 + 關聯法條」的問題,命中拆為三元件:

| 元件 | baseline(固定管線 + 關鍵字觸發查案例) | tool-calling agent |
|---|---|---|
| reg_hit | 14/20 = 0.700 [0.500, 0.900] | 10/20 = 0.500 [0.300, 0.700] |
| case_hit(17 題要求) | 16/17 = 0.941 [0.824, 1.000] | 15/17 = 0.882 [0.706, 1.000] |
| related_hit(3 題要求) | 0/3 | 3/3 |
| full_hit | 10/20 = 0.500 [0.300, 0.700] | 8/20 = 0.400 [0.200, 0.600] |

翻對 2 / 翻錯 4,p = 0.688;agent 平均 2.15 次工具呼叫、18/20 題 >1 次。即使在為 agent 設計的題組上,agent 未勝過固定管線加關鍵字規則:案例一跳兩者相當;法規一跳 agent 輸 4 題(多步時 LLM 濃縮查詢的語意飄移放大,且 2 題跳過法規檢索);唯一無可取代者為關聯法條工具(3/3)。本題組首次執行時案例命中 0/17,追查發現 `search_violation_cases` 誤用法規 chunk 的 FAISS 索引——潛藏數月的真實缺陷,由多步題組暴露,已修正並加入回歸測試。

### 5.6 答案層引用驗證

108 題中 100 題通過信心閘門並生成答案,89 題答案含條號;驗證前引用 context 中不存在條號者 0/100 = 0.000 [0.000, 0.000],重生成機制未被觸發。與 5.4 的 grounded 強制覆寫一致:在 gpt-4o-mini 與現行 prompt 下,兩道程式碼層邊界均未攔到任何案例,其價值為保證而非可量測的提升;四道邊界中僅字面錨定 drift 檢查在本評估集上有可量測效果。

### 5.7 路由器

166 題標籤集(108+8 單跳、20 multi-hop、30 題手寫審稿/案例/邊界題):整體 162/166 = 0.976 [0.952, 0.994];規則層 61 題(37%)100%,LLM 層 105 題 96.2%。代價不對稱的錯誤分開計:multi_hop 被判成單跳(會漏案例/關聯法條)0 題;單跳被判成 multi_hop(只是變慢)2 題。第一版規則層 90%,錯誤集中在把泛用罰則字眼當成案例訊號;收窄規則、其餘交 LLM 後整體由 94.0% 升至 97.6%。

---

## 6. 討論

**小樣本會製造假結論。** 本專案兩個最重要的發現都是「24 題時的結論在 108 題上不成立」。在 n=24 下,1 題之差就是 4.2 個百分點,任何「提升」都在雜訊範圍內;bootstrap CI 讓這件事變得無法忽視。

**負向結果的價值在於指向下一步。** reranker-base 無效 → 追單一失敗案例 → 發現詞義混淆 → 換模型 → 顯著提升,這條路徑比「加了 reranking 就變好」更能說明系統為什麼是現在這個樣子。

**agent 的價值要用對的題目才量得到——量了之後,答案是「在這個領域沒有」。** 單跳問題上 agent 平均只呼叫 1 次工具,與固定重試打平;多步題組上 harness 確實被用到(2.15 次呼叫),但 full_hit 仍未勝過「固定管線 + 關鍵字觸發查案例」(8 vs 10)。LLM 自主決定查詢字串的自由度在多步場景放大了 query drift,且會跳過必要的法規檢索。agent 唯一無可取代的是關聯法條工具。合理的下一步不是讓 agent 更聰明,而是把案例與關聯法條做成規則觸發的確定性多步管線再比一次。

**邊界要靠程式碼,但要誠實說哪些邊界真的被用到。** 消融顯示只有 drift 檢查有可量測效果;grounded 強制覆寫與引用驗證在 gpt-4o-mini + 現行 prompt 下一個案例都沒攔到——prompt 這次守住了。它們的價值是換模型或改 prompt 時的保險,而非本評估集上的提升;把這件事寫清楚,比宣稱「四道邊界都重要」更有用,因為它說明了 harness 的成本(每題多一次 rerank、多一次驗證)換到的到底是什麼。

## 7. 限制

合成題偏差、confidently wrong 不可偵測、案例庫 391/400 集中在第 28 條、法條偵測不認中文數字條號、`第15條之一` 解析 bug 修正後索引尚未重建、OpenAI API 非完全確定性——詳見 README「誠實的限制」。

## 8. 結論

一個領域 RAG 系統的可信度不來自它用了多少時髦元件,而來自每個元件都經過對照、每個數字都帶著不確定性、每個失敗都被追到根因。本專案在食品法規語料上證實:reranker 模型品質才是提升來源,信心閾值必須用足夠大的樣本校準並誠實揭露取捨,tool-calling agent 的價值取決於問題是否真的需要多步,而 harness 的邊界必須由程式碼保證。

## 參考文獻

[1] J. Chen et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation," arXiv:2402.03216, 2024.
[2] S. Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding," arXiv:2309.07597, 2023.
[3] S.-Q. Yan et al., "Corrective Retrieval Augmented Generation," arXiv:2401.15884, 2024.
[4] A. Singh et al., "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG," arXiv:2501.09136, 2025.
[5] X. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models," ICLR 2023.
[6] B. Bohnet et al., "Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models," arXiv:2212.08037, 2022.
[7] B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
[8] S. Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity," NAACL 2024.
