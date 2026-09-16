"""
重建索引後,把評估集裡的 gold_chunk_id 從舊索引對照到新索引。

為什麼需要:ingest/step1_parse.py 用 ProcessPool 平行解析、依完成順序寫 chunks.jsonl,
所以重建後 chunk id 會重排。評估集(eval_questions*.json、hard_questions.json)存的是
舊 id,不對照就全部失效。

做法:舊 DB 取出 gold chunk 的 text → 在新 DB 找 text 完全相同的 chunk → 新 id。
找不到完全相同的(例如 OCR 或解析結果有差)就用字元 3-gram Jaccard 找最像的,
並印出相似度讓人檢查;相似度 < 0.9 的一律列出來、不自動改。

用法:
  python scripts/remap_eval_gold_ids.py data/index_backup_20260916/chunks.db data/index/chunks.db
  → 直接改寫 eval/*.json(改之前會存 .bak)
"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
EVAL_FILES = ["eval/eval_questions.json", "eval/eval_questions_v2.json", "eval/hard_questions.json",
              "eval/generated_questions.json", "eval/generated_questions_b2.json"]


def ngrams(s: str, n: int = 3) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def main(old_db_path: str, new_db_path: str):
    old = sqlite3.connect(old_db_path)
    new = sqlite3.connect(new_db_path)
    new_by_text = {}
    for cid, text in new.execute("SELECT id, text FROM chunks"):
        new_by_text.setdefault(text, cid)
    new_rows = list(new.execute("SELECT id, text FROM chunks"))
    print(f"舊索引 {old.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} chunks,新索引 {len(new_rows)} chunks")

    cache: dict[int, tuple[int | None, float]] = {}

    def remap(old_id: int) -> tuple[int | None, float]:
        if old_id in cache:
            return cache[old_id]
        row = old.execute("SELECT text FROM chunks WHERE id=?", (old_id,)).fetchone()
        if row is None:
            cache[old_id] = (None, 0.0)
            return cache[old_id]
        text = row[0]
        if text in new_by_text:
            cache[old_id] = (new_by_text[text], 1.0)
            return cache[old_id]
        g = ngrams(text)
        best_id, best = None, 0.0
        for cid, t in new_rows:
            s = jaccard(g, ngrams(t))
            if s > best:
                best_id, best = cid, s
        cache[old_id] = (best_id, best)
        return cache[old_id]

    total = exact = fuzzy = unresolved = 0
    for rel in EVAL_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        qs = json.loads(p.read_text(encoding="utf-8"))
        changed = False
        for q in qs:
            old_id = q["gold_chunk_id"]
            new_id, sim = remap(old_id)
            total += 1
            if new_id is None or sim < 0.9:
                unresolved += 1
                print(f"  ⚠️ 未解決 {rel} old={old_id} best={new_id} sim={sim:.3f}  {q['question'][:40]}")
                continue
            if sim < 1.0:
                fuzzy += 1
                print(f"  ~ 模糊對照 {rel} {old_id} → {new_id} sim={sim:.3f}")
            else:
                exact += 1
            if new_id != old_id:
                q["gold_chunk_id_old"] = old_id
                q["gold_chunk_id"] = new_id
                changed = True
        if changed:
            p.with_suffix(".json.bak").write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
            p.write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已改寫 {rel}(備份 .bak)")
    print(f"\n總計 {total} 個 gold:完全相同 {exact}、模糊對照 {fuzzy}、未解決 {unresolved}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
