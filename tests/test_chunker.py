"""Chunker 單元測試。"""
import pytest

from ingest.chunker.hash_delimiter import split_by_hash, is_hash_format
from ingest.chunker.qa_chunker import split_by_qa, is_qa_format
from ingest.chunker.header_chunker import split_by_headers, has_structural_headers
from ingest.chunker.router import chunk_document


class TestHashChunker:
    def test_basic_split(self):
        text = "@文件A 章節1 內容內容內容內容內容內容內容內容內容內容內容內容內容內容內容內容內容\n#\n@文件A 章節2 又是內容又是內容又是內容又是內容又是內容又是內容又是內容\n#"
        chunks = split_by_hash(text)
        assert len(chunks) == 2
        assert "章節1" in chunks[0]
        assert "@" not in chunks[0]  # @ 標頭應已移除

    def test_empty_blocks_filtered(self):
        text = "@a 短\n#\n@b 還是很短\n#\n@c " + ("長" * 50) + "\n#"
        chunks = split_by_hash(text, min_len=30)
        # 只有 c 夠長
        assert len(chunks) == 1

    def test_is_hash_format(self):
        assert is_hash_format("液蛋指引#AI資料庫.txt")
        assert is_hash_format("某指引AI整理.txt")
        assert not is_hash_format("食品衛生標準.pdf")


class TestQAChunker:
    def test_basic_qa(self):
        text = """
Q1:本準則是什麼? A1:這是某個準則的說明,內容相當完整。
Q2:適用範圍? A2:適用於所有食品業者,範圍很廣。
Q3:罰則? A3:違反者依法第28條處罰。
"""
        chunks = split_by_qa(text, min_len=10)
        assert len(chunks) >= 2

    def test_qa_no_punctuation(self):
        # 沒標點的 Q1
        text = "Q1本規定的法源依據為何?A1依據食安法第9條第1項。Q2何謂收貨?A2指實際收受貨物之動作。"
        chunks = split_by_qa(text, min_len=10)
        assert len(chunks) == 2

    def test_is_qa_format(self):
        assert is_qa_format("", "食品衛生標準Q&A.pdf")
        assert is_qa_format("Q1:... Q2:... Q3:... Q4:... Q5:... Q6:...", "")
        assert not is_qa_format("普通段落內容", "普通文件.pdf")


class TestHeaderChunker:
    def test_chinese_numerals(self):
        text = "壹、前言 此處說明前言。" * 5 + "\n貳、適用範圍 範圍說明。" * 5 + "\n參、規範內容 各種規範。" * 5
        chunks = split_by_headers(text, max_len=2000, min_len=10)
        assert len(chunks) == 3
        assert chunks[0].startswith("壹")
        assert chunks[1].startswith("貳")


class TestRouter:
    def test_router_picks_hash(self):
        text = "@a 內容多多多多多多多多多多多多多多多多多多多多多多多多多多多多多多多\n#"
        chunks, strategy = chunk_document(text, "test#AI資料庫.txt")
        assert strategy == "hash"

    def test_router_picks_qa(self):
        text = "\n".join([f"Q{i}:這是問題{i}? A{i}:這是答案{i},內容夠長。" for i in range(1, 8)])
        chunks, strategy = chunk_document(text, "問答集.pdf")
        assert strategy == "qa"

    def test_empty_input(self):
        chunks, strategy = chunk_document("", "x.txt")
        assert chunks == []
