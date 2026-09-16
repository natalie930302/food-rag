"""
給 tool-calling agent(app/agent.py)用的工具定義。

跟 app/agentic_retrieval.py 的差異:agentic_retrieval 是「固定動作」——信心不夠
就一定重新表述問題再查一次,LLM 只負責改寫問題本身,不負責決定「該做什麼」。
這裡把決定權交給 LLM:LLM 自己看第一次檢索的信心分數,決定要不要重新表述問題、
要不要查違規案例佐證、要不要查關聯法條擴大範圍——這才是 tool-calling agent
harness 的核心差異(LLM 驅動控制流程,不是寫死的 if/else)。

三個工具都是既有檢索邏輯的薄包裝,不是重新實作:
  - search_regulations   包 app/corrective_retrieval.py 的信心閘門檢索
  - search_violation_cases 包 app/retrieval.py 的案例檢索
  - search_related_laws  包 app/retrieval.py 的法條共現統計

信心分數(confident/top_score)如實回傳給 LLM,而不是包裝成「找到了/沒找到」的
二元訊號——這樣 system prompt 才能要求 LLM 在 confident=false 時不要把這批
內容當可信依據使用,把 corrective RAG「誠實拒答」的紀律從單次檢索延伸到
多步驟的 agent 迴圈裡。

## 字面錨定一致性檢查(2026/09 新增,見 README「Query Drift 緩解」章節)

早期 tool-agent 評估(見 docs/RESEARCH_LOG.md)實測發現一個新失效模式:LLM 幫 search_regulations 選的
查詢字串(通常是濃縮過的關鍵字,不是原始問題全文)有時候會語意飄移,檢索到
完全不相關的候選集,而且信心分數比用原始問題字面文字查還高(「食品添加物
輸入登記」那題:字面問題查詢 gold chunk 排名第一分數0.974,LLM濃縮的關鍵字
查詢卻查到不相關候選、分數還更高的0.978)——corrective RAG 的信心閘門評的是
「這批結果看起來像不像相關」,不是「查詢改寫有沒有偏離原意」,兩者不等價,
原本的機制對這種漂移完全沒有防禦力。

緩解做法:每次 search_regulations 呼叫,除了用 LLM 選的查詢字串檢索,**免費
多做一次**用原始問題字面文字的檢索(純本地運算,不呼叫 LLM,不增加 API 費用)
當一致性錨點。兩者如果對「哪個 chunk 最相關」(top-1)看法一致,信任 LLM 的
版本;意見不一致時,改用字面問題的檢索結果——這個概念上類似 self-consistency
(Wang et al., 2022, "Self-Consistency Improves Chain of Thought Reasoning")
的精神:單一次生成/檢索路徑不可靠時,用多條獨立路徑是否一致來判斷可信度,
只是這裡比較的是「兩種查詢措辭的檢索結果」而不是「多次取樣的推理路徑」。

刻意保守的設計:只在兩者 top-1 不一致時介入,一致時完全不動——這是因為
diagnostic 發現「外銷登錄」那題屬於另一種不同的失效模式(gold chunk本來就卡在
Recall@5邊界,兩種查詢的候選集高度重疊,只是邊界名次因為LLM生成的微小用詞
差異而偶爾被擠出前5名),這個一致性檢查解決不了、也不該嘗試解決邊界排名
抖動的問題,誤把兩種不同根因的失效模式用同一個機制處理,會讓後續診斷更混亂。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.agentic_retrieval import reformulate_query
from app.corrective_retrieval import retrieve_with_confidence_gate
from app.decompose import is_compound, keyword_to_question, looks_like_keywords
from app.retrieval import count_chunks_for_law, get_co_cited_laws, retrieve_cases

CASE_GROUNDING_THRESHOLD = 0.58   # 案例向量相似度(cosine)門檻,見 execute_tool 內註解

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": (
                "檢索食品法規/指引/QA 條文。回傳的 confident 欄位代表這批結果"
                "夠不夠可信——confident=false 時,這些內容不能當作回答依據使用,"
                "應該換一種說法重新查詢,或改用其他工具找佐證。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "完整的自然語言問句(例如「減肥廣告違反食安法哪一條?」),不要只給關鍵字;一次問好幾件事時,一個子問題查一次"},
                    "law_article": {
                        "type": "string",
                        "description": "可選,鎖定特定法條範圍,例如「食安法第28條」",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_violation_cases",
            "description": "檢索過去的違規裁罰案例,用來佐證罰則、金額或實際執法案例。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要找的案例類型,用一句話描述(例如「宣稱減肥的食品廣告被裁罰的案例」)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_related_laws",
            "description": (
                "查一個法條常常跟哪些其他法條一起被引用。當 search_regulations "
                "信心不足、懷疑問題其實該問另一條法規時,用這個工具找方向,"
                "而不是盲目重試同一個查詢。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_full": {
                        "type": "string",
                        "description": "法條號碼,例如「28」或「15之一」(不含「第」「條」字)",
                    },
                },
                "required": ["article_full"],
            },
        },
    },
]


@dataclass
class ToolCallRecord:
    """記錄一次工具呼叫,供 API 回傳的 trace 跟 eval 腳本使用。"""
    name: str
    arguments: dict
    confident: bool | None = None
    top_score: float | None = None
    chunk_ids: list[int] = field(default_factory=list)
    result_summary: str = ""
    query_drift_detected: bool = False
    retry_used: bool = False          # 工具內建的 LLM 改寫重試有沒有被觸發(2026/09 修正,見 execute_tool)
    # 給答案層引用驗證用:每個 chunk 的 text / primary_law(只在 search_regulations 有值)
    chunks: list[dict] = field(default_factory=list)


def execute_tool(
    db, embed_model, faiss_index, reranker, call_name: str, args: dict, original_question: str = "",
    drift_check: bool = True, faiss_cases=None, client=None, tool_retry: bool = True,
) -> tuple[str, ToolCallRecord]:
    """執行一次工具呼叫,回傳 (要塞回 messages 的 JSON 字串, 給 trace 用的記錄)。

    original_question:使用者最原始的問題字面文字(不是 LLM 改寫過的查詢),只有
    search_regulations 會用到,做字面錨定一致性檢查(見上方模組說明)。
    drift_check=False 只給 eval/eval_harness_ablation.py 量「這道檢查擋掉了什麼」用。
    client / tool_retry:search_regulations 信心不足時,用跟固定管線同一套「LLM 改寫問題重查一次」
    (app/agentic_retrieval.reformulate_query)。2026/09 multi-hop 評估追出 agent 輸給固定管線的根因之一
    就是這裡:固定管線有這次重試、agent 的工具沒有,LLM 又常常不自己重試——工具先天少一次機會。
    """
    if call_name == "search_regulations":
        filters = {"law_article": args["law_article"]} if args.get("law_article") else None
        llm_query = args["query"]
        # 工具端保證:LLM 給關鍵字(「減肥廣告」)就補成問句再查。cross-encoder 對關鍵字打分極低
        # (實測 0.27 vs 問句 0.91),這件事不能只在 prompt 裡拜託它。
        query_fixed = False
        if looks_like_keywords(llm_query):
            llm_query = keyword_to_question(llm_query)
            query_fixed = True
        result = retrieve_with_confidence_gate(
            db, embed_model, faiss_index, reranker, llm_query, filters=filters,
        )

        query_drift_detected = False
        if drift_check and original_question and llm_query.strip() != original_question.strip():
            literal_result = retrieve_with_confidence_gate(
                db, embed_model, faiss_index, reranker, original_question, filters=filters,
            )
            llm_top1 = result.chunks[0].chunk_id if result.chunks else None
            literal_top1 = literal_result.chunks[0].chunk_id if literal_result.chunks else None
            # 只在字面問題本身也有信心、且兩者top-1看法不一致時介入——字面問題
            # 沒信心時保留LLM改寫查詢的結果(這是agentic retry原本就有效的情境)
            if literal_result.confident and llm_top1 != literal_top1:
                query_drift_detected = True
                result = literal_result

        retry_used = False
        if not result.confident and tool_retry and client is not None:
            # 原始問題是複合句(多個問號)時,整句本身就是壞查詢,改寫的底稿用工具這次的子問句
            base = llm_query if (not original_question or is_compound(original_question)) else original_question
            reformulated = reformulate_query(client, base)
            retry_result = retrieve_with_confidence_gate(
                db, embed_model, faiss_index, reranker, reformulated, filters=filters,
            )
            retry_used = True
            if retry_result.confident:
                result = retry_result

        payload = {
            "confident": result.confident,
            "top_score": result.top_score,
            "chunks": [
                {"chunk_id": c.chunk_id, "law": c.primary_law, "document": c.document, "text": c.text}
                for c in result.chunks
            ],
        }
        if query_fixed:
            payload["query_used"] = llm_query
            payload["note_query"] = f"你給的是關鍵字,系統已改成問句「{llm_query}」再查;之後請直接給完整問句。"
        if query_drift_detected:
            payload["note"] = "你的查詢字串跟原始問題字面文字檢索結果不一致,已改用原始問題字面文字的檢索結果。"
        elif retry_used and result.confident:
            payload["note"] = "第一次檢索信心不足,系統已自動改寫問題重查一次並找到可信內容。"
        elif not result.confident:
            payload["note"] = "信心分數過低,不可當作回答依據,請換句話說重新查詢或改用其他工具。"
        record = ToolCallRecord(
            name=call_name, arguments={**args, "query_used": llm_query} if query_fixed else args,
            confident=result.confident, top_score=result.top_score,
            chunk_ids=[c.chunk_id for c in result.chunks],
            result_summary=f"{len(result.chunks)} chunks, confident={result.confident}, score={result.top_score}"
                           + (" [關鍵字→問句]" if query_fixed else ""),
            query_drift_detected=query_drift_detected, retry_used=retry_used,
            chunks=[{"chunk_id": c.chunk_id, "text": c.text, "primary_law": c.primary_law} for c in result.chunks],
        )
        return json.dumps(payload, ensure_ascii=False), record

    if call_name == "search_violation_cases":
        # 案例有自己的 FAISS 索引(faiss_cases.index),不能拿法規 chunk 的索引去查——
        # 2026/09 eval/eval_multihop.py 量到 agent 案例命中 0/17 才發現原本傳錯索引,
        # 查到的是法規向量空間裡的鄰居再去 violations 表撈 id,結果全錯。
        if faiss_cases is None:
            from app.deps import get_faiss_cases
            faiss_cases = get_faiss_cases()
        cases = retrieve_cases(db, embed_model, faiss_cases, args["query"], top_k=3)
        # FAISS 永遠回傳最近鄰,所以「查到 3 筆」不等於「有相關案例」。實測相關問題 0.60–0.66、
        # 無關問題(捷運票價、天氣)0.45–0.52,以 0.58 為界;有信心的案例才算回答依據(grounded)。
        top = cases[0].score if cases and cases[0].score is not None else None
        cases_confident = top is not None and top >= CASE_GROUNDING_THRESHOLD
        payload = {
            "confident": cases_confident,
            "top_score": top,
            "cases": [
                {
                    "id": c.id, "year": c.year, "month": c.month,
                    "product": c.product, "company": c.company,
                    "penalty_twd": c.penalty_twd, "law_cited": c.law_cited,
                    "violation": (c.violation or "")[:300],
                }
                for c in cases
            ]
        }
        if not cases_confident:
            payload["note"] = "這些案例與問題相似度不足,可能無關,不可當作回答依據。"
        record = ToolCallRecord(
            name=call_name, arguments=args, confident=cases_confident, top_score=top,
            chunk_ids=[c.id for c in cases],
            result_summary=f"{len(cases)} cases, confident={cases_confident}, score={top}",
            # 有信心的案例也是引用驗證的證據:答案引用案例的法條(「第28條第1項」)不該被判成瞎掰
            chunks=[{"chunk_id": -c.id, "primary_law": c.law_cited,
                     "text": f"{c.year}年{c.month}月 {c.company or ''}「{c.product or ''}」違反{c.law_cited or ''},"
                             f"裁處 {c.penalty_twd or 0} 元。{(c.violation or '')[:120]}"}
                    for c in cases] if cases_confident else [],
        )
        return json.dumps(payload, ensure_ascii=False), record

    if call_name == "search_related_laws":
        rows = get_co_cited_laws(db, args["article_full"])
        total = count_chunks_for_law(db, args["article_full"])
        payload = {
            "total_chunks_for_this_article": total,
            "co_cited_laws": [
                {"law_name": r["related_law_name"], "article": r["related_article"], "co_count": r["co_count"]}
                for r in rows
            ],
        }
        record = ToolCallRecord(
            name=call_name, arguments=args, confident=False,
            result_summary=f"total={total}, {len(rows)} related laws",
            # 關聯法條是「條文共現統計」,不是針對使用者問題的證據:任何條號都查得到共現,所以它不能單獨
            # 讓整題算 grounded(2026/09 e2e 抓到 OOD 題「登山失溫」因此被放行)。但它列出的是真實條號,
            # 仍放進 chunks 供引用驗證,答案列出關聯條文時才不會被驗證器當成瞎掰。
            chunks=[{"chunk_id": -1, "primary_law": f"{r['related_law_name']}第{r['related_article']}條",
                     "text": f"{r['related_law_name']}第{r['related_article']}條(與第{args['article_full']}條共同引用 {r['co_count']} 次)"}
                    for r in rows],
        )
        return json.dumps(payload, ensure_ascii=False), record

    raise ValueError(f"未知工具: {call_name}")
