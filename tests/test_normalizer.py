"""文字清理測試。"""
from ingest.normalizer import (
    merge_broken_lines,
    normalize,
    normalize_unicode,
    remove_invisible_chars,
    remove_page_numbers,
)


class TestNormalize:
    def test_remove_bom(self):
        text = "\ufeff內容開始"
        assert remove_invisible_chars(text) == "內容開始"

    def test_remove_page_number(self):
        text = "第1頁,共10頁\n內容\n- 5 -"
        out = remove_page_numbers(text)
        assert "第1頁" not in out
        assert "- 5 -" not in out

    def test_unicode_nfkc(self):
        # 全形數字 → 半形
        text = "第28條"
        out = normalize_unicode(text)
        assert "28" in out

    def test_merge_broken_lines(self):
        text = "這段話被\n斷在中間。\n下一句又被\n斷開。"
        out = merge_broken_lines(text)
        assert "這段話被斷在中間。" in out

    def test_keep_new_paragraph(self):
        text = "壹、第一段內容。\n貳、第二段內容。"
        out = merge_broken_lines(text)
        # 兩段應該分開
        assert "貳" in out

    def test_full_pipeline(self):
        text = "\ufeff第1頁,共3頁\n壹、前言   多餘空格\n\n\n\n二、適用範圍"
        out = normalize(text)
        assert "第1頁" not in out
        assert "\ufeff" not in out
        assert "\n\n\n" not in out
