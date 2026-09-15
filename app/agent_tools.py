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

eval/eval_tool_agent.py 實測發現一個新失效模式:LLM 幫 search_regulations 選的
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

from app.corrective_retrieval import retrieve_with_confidence_gate
from app.retrieval import count_chunks_for_law, get_co_cited_laws, retrieve_cases

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
                    "query": {"type": "string", "description": "檢索用的問題或關鍵字"},
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
                    "query": {"type": "string", "description": "案例檢索關鍵字或問題描述"},
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
    # 給答案層引用驗證用:每個 chunk 的 text / primary_law(只在 search_regulations 有值)
    chunks: list[dict] = field(default_factory=list)


def execute_tool(
    db, embed_model, faiss_index, reranker, call_name: str, args: dict, original_question: str = "",
    drift_check: bool = True, faiss_cases=None,
) -> tuple[str, ToolCallRecord]:
    """執行一次工具呼叫,回傳 (要塞回 messages 的 JSON 字串, 給 trace 用的記錄)。

    original_question:使用者最原始的問題字面文字(不是 LLM 改寫過的查詢),只有
    search_regulations 會用到,做字面錨定一致性檢查(見上方模組說明)。
    drift_check=False 只給 eval/eval_harness_ablation.py 量「這道檢查擋掉了什麼」用。
    """
    if call_name == "search_regulations":
        filters = {"law_article": args["law_article"]} if args.get("law_article") else None
        llm_query = args["query"]
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

        payload = {
            "confident": result.confident,
            "top_score": result.top_score,
            "chunks": [
                {"chunk_id": c.chunk_id, "law": c.primary_law, "document": c.document, "text": c.text}
                for c in result.chunks
            ],
        }
        if query_drift_detected:
            payload["note"] = "你的查詢字串跟原始問題字面文字檢索結果不一致,已改用原始問題字面文字的檢索結果。"
        elif not result.confident:
            payload["note"] = "信心分數過低,不可當作回答依據,請換句話說重新查詢或改用其他工具。"
        record = ToolCallRecord(
            name=call_name, arguments=args,
            confident=result.confident, top_score=result.top_score,
            chunk_ids=[c.chunk_id for c in result.chunks],
            result_summary=f"{len(result.chunks)} chunks, confident={result.confident}, score={result.top_score}",
            query_drift_detected=query_drift_detected,
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
        payload = {
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
        record = ToolCallRecord(
            name=call_name, arguments=args,
            chunk_ids=[c.id for c in cases],
            result_summary=f"{len(cases)} cases",
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
            name=call_name, arguments=args,
            result_summary=f"total={total}, {len(rows)} related laws",
        )
        return json.dumps(payload, ensure_ascii=False), record

    raise ValueError(f"未知工具: {call_name}")
