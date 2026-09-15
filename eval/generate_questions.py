"""
把評估集從 24 題擴大到 ~100 題:用 LLM 從隨機抽樣的 chunk 生成自然提問。

為什麼要做:原本 24 題手寫評估集太小——Recall@1 從 0.708 到 0.750 其實只是 24 題裡
多對 1 題,這種差距在 n=24 下跟雜訊分不開(見 eval/stats.py 的 bootstrap CI)。
手寫 100 題要花好幾天,用 LLM 生成 + 自動過濾 + 人工抽查是務實的折衷。

誠實說明合成題目的偏差:LLM 看著 chunk 出題,問法會比真實使用者更「貼近原文」,
即使已經用「禁止複製原文片段」的規則過濾,合成題整體仍可能比手寫題簡單。所以
輸出檔會標記 source=synthetic / manual,所有評估都會分開報告兩組數字,不混在一起。

過濾規則(自動):
  1. 題目長度 12~60 字
  2. 題目跟 chunk 的最長共同子字串 <= MAX_COMMON_SUBSTR 字(避免文字重疊灌水 Recall)
  3. 抽樣時排除跟其他 chunk 幾乎重複的內容(FAISS 最近鄰 cosine >= DUP_THRESHOLD),
     否則 gold chunk 不唯一,評估會誤判
  4. 排除已存在於 eval_questions.json / hard_questions.json 的 gold chunk

用法:
  python generate_questions.py --n 80 --seed 42
  → 寫入 eval/generated_questions.json(候選,含 answer_span 供人工抽查)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from app.deps import get_db, get_faiss_chunks, get_openai_client
from config.settings import settings

HERE = Path(__file__).parent
MIN_CHUNK_LEN, MAX_CHUNK_LEN = 120, 700
MIN_Q_LEN, MAX_Q_LEN = 12, 60
MAX_COMMON_SUBSTR = 6
DUP_THRESHOLD = 0.95
KIND_QUOTA = {"qa": 0.5, "guide": 0.35, "standard": 0.15}

PROMPT = """你是在幫一個「食品法規問答系統」建立評估資料。下面是資料庫裡的一段內容。

請寫「一個」台灣的食品業者或一般民眾可能會問的問題,答案必須能從這段內容找到。

要求:
- 用口語、自然的問法,像真人在問客服,不要像考題
- 不可以直接複製內容裡的詞句(超過 4 個字連續相同就算複製),要換自己的說法
- 不要提到文件名稱、題號、「本文」「這段」之類的字眼
- 問題長度 15~45 字
- 同時給出答案在內容裡的關鍵句(answer_span,原文照抄即可,10~60 字)

只回傳 JSON,格式:{{"question": "...", "answer_span": "..."}}

內容:
\"\"\"
{chunk}
\"\"\""""


def longest_common_substring(a: str, b: str) -> int:
    """O(len(a)*len(b)) 的 DP,題目短所以夠快。"""
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def existing_gold_ids() -> set[int]:
    ids = set()
    # 也排除先前批次已生成的 gold,避免不同 seed 抽到同一個 chunk(第一、二批就撞過一次)
    for p in [HERE / "eval_questions.json", HERE / "hard_questions.json", *HERE.glob("generated_questions*.json")]:
        if p.exists() and not p.name.endswith("_rejected.json"):
            ids |= {q["gold_chunk_id"] for q in json.loads(p.read_text(encoding="utf-8"))}
    return ids


def has_near_duplicate(index, embedding_id: int) -> bool:
    vec = index.reconstruct(int(embedding_id)).reshape(1, -1).astype(np.float32)
    scores, idxs = index.search(vec, 3)
    for s, i in zip(scores[0], idxs[0]):
        if i != embedding_id and s >= DUP_THRESHOLD:
            return True
    return False


def sample_chunks(db, index, n: int, rng: random.Random, exclude: set[int]) -> list[dict]:
    rows = db.execute(
        """SELECT id, text, kind, document, embedding_id FROM chunks
           WHERE char_len BETWEEN ? AND ? AND kind IN ('qa','guide','standard')
             AND embedding_id IS NOT NULL""",
        (MIN_CHUNK_LEN, MAX_CHUNK_LEN),
    ).fetchall()
    by_kind: dict[str, list] = {}
    for r in rows:
        if r["id"] in exclude:
            continue
        by_kind.setdefault(r["kind"], []).append(r)

    picked = []
    for kind, frac in KIND_QUOTA.items():
        pool = by_kind.get(kind, [])
        rng.shuffle(pool)
        want = round(n * frac)
        got = 0
        for r in pool:
            if got >= want:
                break
            if has_near_duplicate(index, r["embedding_id"]):
                continue
            picked.append(dict(r))
            got += 1
        print(f"  {kind}: 抽 {got}/{want}(池子 {len(pool)})")
    rng.shuffle(picked)
    return picked


def generate_one(client, chunk_text: str) -> dict | None:
    resp = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.8,
        max_tokens=200,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(chunk=chunk_text)}],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data if "question" in data else None
    except (json.JSONDecodeError, TypeError):
        return None


def passes_filters(question: str, chunk_text: str) -> tuple[bool, str]:
    q = question.strip()
    if not (MIN_Q_LEN <= len(q) <= MAX_Q_LEN):
        return False, f"長度 {len(q)}"
    lcs = longest_common_substring(q, chunk_text)
    if lcs > MAX_COMMON_SUBSTR:
        return False, f"與原文共同子字串 {lcs} 字"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="目標題數(會多抽 30% 當過濾餘裕)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="generated_questions.json")
    ap.add_argument("--quota", default=None,
                    help="各 kind 抽樣比例,例如 qa=0.7,guide=0.2,standard=0.1(第二批用:表格/檢驗方法類人工審查淘汰率高)")
    args = ap.parse_args()
    if args.quota:
        KIND_QUOTA.clear()
        KIND_QUOTA.update({k: float(v) for k, v in (kv.split("=") for kv in args.quota.split(","))})

    rng = random.Random(args.seed)
    db = get_db()
    index = get_faiss_chunks()
    client = get_openai_client()

    print("抽樣 chunks(排除近似重複、排除既有評估題)...")
    candidates = sample_chunks(db, index, int(args.n * 1.3), rng, existing_gold_ids())

    accepted, rejected = [], []
    for i, c in enumerate(candidates, 1):
        if len(accepted) >= args.n:
            break
        gen = generate_one(client, c["text"])
        if gen is None:
            rejected.append({"chunk_id": c["id"], "reason": "LLM 回傳格式錯誤"})
            continue
        ok, why = passes_filters(gen["question"], c["text"])
        if not ok:
            # 再給一次機會,加強「不要複製原文」的指示
            gen2 = generate_one(client, c["text"] + "\n\n(注意:上一版問題跟原文重疊太多,請完全換一種說法)")
            if gen2 and passes_filters(gen2["question"], c["text"])[0]:
                gen, ok = gen2, True
        if not ok:
            rejected.append({"chunk_id": c["id"], "reason": why, "question": gen["question"]})
            continue
        accepted.append({
            "question": gen["question"].strip(),
            "gold_chunk_id": c["id"],
            "kind": c["kind"],
            "document": c["document"],
            "answer_span": gen.get("answer_span", ""),
            "source": "synthetic",
            "lcs_with_chunk": longest_common_substring(gen["question"], c["text"]),
        })
        print(f"[{len(accepted):3d}/{args.n}] ({c['kind']}) {gen['question']}")

    out = HERE / args.out
    out.write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / (Path(args.out).stem + "_rejected.json")).write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n接受 {len(accepted)} 題、過濾掉 {len(rejected)} 題 → {out}")
    print("下一步:人工抽查 answer_span 是否真的回答了 question,再 merge 進 eval_questions_v2.json")


if __name__ == "__main__":
    main()
