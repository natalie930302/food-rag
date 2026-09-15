"""app/verifier.py:答案引用的法條有沒有真的出現在檢索內容裡。"""
from app.verifier import Citation, extract_citations, feedback_for_regeneration, verify_citations
from tests.conftest import make_chunk


def test_extracts_arabic_and_chinese_numerals_and_dedups():
    answer = "依食安法第28條第1項規定……另依同法第二十八條……以及健康食品管理法第14條、第15條之一。"
    cits = extract_citations(answer)
    assert {(c.article, c.suffix) for c in cits} == {(28, None), (14, None), (15, "一")}
    assert next(c for c in cits if c.article == 28).law == "食安法"


def test_supported_by_chunk_text():
    chunks = [make_chunk(1, "依食品安全衛生管理法第 28 條規定,食品廣告不得誇張。")]
    check = verify_citations("依食安法第28條,不可以。", chunks)
    assert check.supported and check.unsupported == []


def test_supported_by_chinese_numeral_in_chunk():
    chunks = [make_chunk(1, "第二十八條 食品之標示、宣傳或廣告,不得有不實、誇張……")]
    assert verify_citations("依第28條", chunks).supported


def test_supported_by_primary_law_metadata():
    c = make_chunk(1, "廣告不得宣稱醫療效能。")
    c.primary_law = "食安法第28條"
    assert verify_citations("依食安法第28條", [c]).supported


def test_unsupported_citation_is_flagged():
    c = make_chunk(1, "依食安法第22條,包裝食品應標示品名。")
    c.primary_law = "食安法第22條"
    check = verify_citations("依食安法第28條第1項,這是違規廣告。", [c])
    assert not check.supported
    assert [c.label for c in check.unsupported] == ["食安法第28條"]


def test_suffix_article_not_confused_with_base_article():
    # 檢索到的是第15條,答案卻引第15條之一 → 不算有依據
    c = make_chunk(1, "食安法第15條:食品不得有下列情形……")
    c.primary_law = "食安法第15條"
    check = verify_citations("依食安法第15條之一,應建立追溯系統。", [c])
    assert not check.supported


def test_suffix_both_spellings_supported():
    assert verify_citations("依第15條之一", [make_chunk(1, "依第15之一條規定")]).supported
    assert verify_citations("依第15條之一", [make_chunk(1, "依第15條之一規定")]).supported


def test_no_citations_is_trivially_supported():
    check = verify_citations("資料庫裡沒有相關依據。", [])
    assert check.supported and check.citations == []


def test_accepts_dict_chunks():
    assert verify_citations("依第8條", [{"text": "第8條 食品良好衛生規範", "primary_law": None}]).supported


def test_feedback_lists_unsupported_labels():
    c = make_chunk(1, "第22條")
    c.primary_law = None
    check = verify_citations("依第28條與第45條", [c])
    msg = feedback_for_regeneration(check)
    assert "第28條" in msg and "第45條" in msg and "重新回答" in msg


def test_citation_label():
    assert Citation(28, None, "食安法").label == "食安法第28條"
    assert Citation(15, "一", None).label == "第15條之一"
