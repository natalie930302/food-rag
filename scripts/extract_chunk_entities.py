"""
針對 README「延伸驗證」章節量測 chunking 粒度問題時用的偵測規則,抽出長列舉
段落裡的具體名詞(例如「紅麴膠囊」),建立 entity -> chunk_id 的對照表。這是
解決 Agentic RAG 誠實負向結果(紅麴膠囊問題檢索不到)的目標修補:不重寫
chunker、不重跑整個 ingest,只針對長列舉段落做局部索引補強。

**跟量測腳本(diagnostic)的偵測條件不同,這裡刻意放寬**:量測「這個問題有多
普遍」時用了「定義用語+長列舉」兩個條件一起卡,是為了得到一個保守、不誇大
的普遍程度估計(0.73%)。但拿真正失敗的案例(chunk 14031,紅麴膠囊)回頭測
才發現它用的是「包括但不限於」,不在原本設定的定義用語清單(係指/所稱/指
下列/包括下列/之定義)裡,被漏掉了——這代表量測時用的偵測條件本身就是保守
估計的下界,不是精確值。修補的目的是把候選chunk找出來讓reranker去判斷相不
相關,即使多抓一些无關的候選也不影響安全性(見 entity_boost.py 說明),所以
這裡拿掉「定義用語」這個條件,只憑「長列舉」就抽取,寧可多抓一點。

用法:python extract_chunk_entities.py,輸出 ../data/index/chunk_entities.json
"""
from __future__ import annotations

import json
import re
import sqlite3

ENUM_PATTERN = re.compile(r"(如|例如|包括)[^。]{1,80}")
MIN_TERM_LEN = 3
MAX_TERM_LEN = 12

conn = sqlite3.connect("../data/index/chunks.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT id, text FROM chunks").fetchall()

entity_to_chunks: dict[str, list[int]] = {}
flagged_chunk_count = 0

for row in rows:
    cid, text = row["id"], row["text"]
    if not text:
        continue
    matches = [m for m in ENUM_PATTERN.finditer(text) if m.group(0).count("、") >= 3]
    if not matches:
        continue

    flagged_chunk_count += 1
    for m in matches:
        span = m.group(0)[len(m.group(1)):]  # 去掉開頭的 如/例如/包括
        for item in span.split("、"):
            item = re.sub(r"等[^、]*$", "", item).strip()
            if MIN_TERM_LEN <= len(item) <= MAX_TERM_LEN:
                entity_to_chunks.setdefault(item, [])
                if cid not in entity_to_chunks[item]:
                    entity_to_chunks[item].append(cid)

print(f"符合「長列舉(>=3項例子)」模式的chunks數: {flagged_chunk_count}")
print(f"抽出的候選實體詞數: {len(entity_to_chunks)}")

with open("../data/index/chunk_entities.json", "w", encoding="utf-8") as f:
    json.dump(entity_to_chunks, f, ensure_ascii=False, indent=2)
print("已儲存至 data/index/chunk_entities.json")
