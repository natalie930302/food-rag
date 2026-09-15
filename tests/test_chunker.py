"""Chunker 單元測試。

fixture 依照各 chunker 模組 docstring 記載的格式約定撰寫:
  - hash 格式:`@文件名 章節標題` 獨立一行,內容在下一行,`#` 分隔
  - QA 格式:Q 編號只在「行首」才算分割點(避免把表格內的 Q25 誤判成新題目)
"""
from ingest.chunker.hash_delimiter import is_hash_format, split_by_hash
from ingest.chunker.header_chunker import has_structural_headers, split_by_headers
from ingest.chunker.qa_chunker import is_qa_format, split_by_qa
from ingest.chunker.router import chunk_document

LONG = "內容" * 20  # 40 字,超過預設 chunk_min_len=30


class TestHashChunker:
    def test_basic_split(self):
        text = f"@文件A 章節1\n{LONG}\n#\n@文件A 章節2\n又是{LONG}\n#"
        chunks = split_by_hash(text)
        assert len(chunks) == 2
        assert chunks[1].startswith("又是")
        # @ 標頭行(含章節標題)應整行移除,不混進內容
        assert "@" not in chunks[0]
        assert "章節1" not in chunks[0]

    def test_empty_blocks_filtered(self):
        text = "@a\n短\n#\n@b\n還是很短\n#\n@c\n" + ("長" * 50) + "\n#"
        chunks = split_by_hash(text, min_len=30)
        assert len(chunks) == 1
        assert chunks[0] == "長" * 50

    def test_image_placeholder_removed(self):
        # `@#` 是圖片佔位符:`#` 仍是分隔符,但 `@` 不能殘留在內容裡
        text = f"@a\n{LONG}\n@#\n{LONG}\n#"
        chunks = split_by_hash(text)
        assert len(chunks) == 2
        assert all("@" not in c for c in chunks)

    def test_is_hash_format(self):
        assert is_hash_format("液蛋指引#AI資料庫.txt")
        assert is_hash_format("某指引AI整理.txt")
        assert not is_hash_format("食品衛生標準.pdf")


class TestQAChunker:
    def test_basic_qa(self):
        text = (
            "Q1:本準則是什麼?\nA1:這是某個準則的說明,內容相當完整。\n"
            "Q2:適用範圍?\nA2:適用於所有食品業者,範圍很廣。\n"
            "Q3:罰則?\nA3:違反者依法第28條處罰。"
        )
        chunks = split_by_qa(text, min_len=10)
        assert len(chunks) == 3
        assert chunks[0].startswith("Q1")
        assert "A1" in chunks[0]

    def test_qa_no_punctuation(self):
        text = (
            "Q1本規定的法源依據為何?\nA1依據食安法第9條第1項規定辦理,適用於所有業者。\n"
            "Q2何謂收貨?\nA2指實際收受貨物之動作,包含驗收與入庫作業。"
        )
        chunks = split_by_qa(text, min_len=10)
        assert len(chunks) == 2

    def test_q_inside_table_not_split(self):
        # 表格行內的 Q25 不是新題目,不該被切成一個 chunk
        text = (
            "| 項目 | 說明 |\n| Q25 | 表格內的偽編號 |\n"
            "Q1:問題一?\nA1:答案一,內容夠長夠長夠長。\n"
            "Q2:問題二?\nA2:答案二,內容夠長夠長夠長。"
        )
        chunks = split_by_qa(text, min_len=10)
        assert len(chunks) == 2
        assert all(c.startswith("Q") and "Q25" not in c for c in chunks)

    def test_long_qa_block_split_by_sentence(self):
        text = "Q1:問題?\nA1:" + "這是一句話。" * 40 + "\nQ2:問題二?\nA2:答案二夠長夠長。"
        chunks = split_by_qa(text, min_len=10, max_len=100)
        assert len(chunks) > 2
        assert all(len(c) <= 100 for c in chunks)

    def test_is_qa_format(self):
        assert is_qa_format("", "食品衛生標準Q&A.pdf")
        assert is_qa_format("\n".join(f"Q{i}:..." for i in range(1, 7)), "")
        # 單行內連續 Q 編號不算(只認行首),避免誤判
        assert not is_qa_format("Q1:... Q2:... Q3:... Q4:... Q5:... Q6:...", "")
        assert not is_qa_format("普通段落內容", "普通文件.pdf")


class TestHeaderChunker:
    def test_chinese_numerals(self):
        text = (
            "壹、前言\n此處說明前言,這是一段很長的前言內容,說明整份文件的目的與背景。\n"
            "貳、適用範圍\n本準則適用於所有食品業者,包含製造、加工、販售等各類型業者。\n"
            "參、規範內容\n各種規範的詳細內容在此說明,包含衛生、標示與檢驗等項目。"
        )
        assert has_structural_headers(text)
        chunks = split_by_headers(text, max_len=2000, min_len=10)
        assert len(chunks) == 3
        assert chunks[0].startswith("壹")
        assert chunks[1].startswith("貳")

    def test_short_sections_merged_forward(self):
        text = "一、甲\n二、乙\n三、丙\n" + "四、丁\n" + "很長的內容" * 10
        chunks = split_by_headers(text, max_len=2000, min_len=30)
        # 太短的段落應往下合併,而不是被丟掉
        assert "".join(chunks).count("一、") == 1

    def test_no_headers(self):
        assert not has_structural_headers("沒有任何結構標題的段落。")
        assert split_by_headers("沒有任何結構標題的段落。") == []


class TestRouter:
    def test_router_picks_hash(self):
        text = f"@a 章節\n{LONG}\n#"
        chunks, strategy = chunk_document(text, "test#AI資料庫.txt")
        assert strategy == "hash"
        assert len(chunks) == 1

    def test_router_picks_qa(self):
        text = "\n".join(f"Q{i}:這是問題{i}?\nA{i}:這是答案{i}," + "內容夠長" * 5 for i in range(1, 8))
        chunks, strategy = chunk_document(text, "問答集.pdf")
        assert strategy == "qa"
        assert len(chunks) == 7

    def test_router_falls_back_to_length(self):
        chunks, strategy = chunk_document("平鋪直敘的內容。" * 100, "plain.txt")
        assert strategy == "length"
        assert chunks

    def test_empty_input(self):
        chunks, strategy = chunk_document("", "x.txt")
        assert chunks == []
        assert strategy == "empty"
