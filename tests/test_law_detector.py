"""法條偵測測試。

設計上的已知邊界(測試如實反映,不是遺漏):
  - 沒有法律名稱前綴的「第25條」不會被歸屬到任何法律(避免誤歸屬),
    只有「本法/同法」這類相對指稱在有 primary_law_hint 時才會被推導。
"""
from ingest.extractors.law_detector import detect_laws, primary_law_to_ref


class TestLawDetector:
    def test_simple_food_safety_law(self):
        text = "依食品安全衛生管理法第28條第1項規定..."
        refs = detect_laws(text)
        assert len(refs) >= 1
        assert refs[0].law_name == "食安法"
        assert refs[0].article_no == 28
        assert refs[0].paragraph == "1"

    def test_multiple_articles(self):
        text = """
本問答依食品安全衛生管理法第22條規定辦理。
若違反同法第25條第2項,處新台幣3萬元以上罰鍰。
另依食安法第28條相關規定...
"""
        refs = detect_laws(text, primary_law_hint="食安法第22條")
        articles = {r.article_no for r in refs}
        assert {22, 25, 28} <= articles

    def test_bare_article_not_attributed(self):
        # 沒有法律名稱、也沒有相對指稱 → 不猜,不產生引用
        refs = detect_laws("若違反第25條第2項,處罰鍰。")
        assert refs == []

    def test_role_detection(self):
        text = "違反食安法第28條者,處新臺幣4萬元以上400萬元以下罰鍰。"
        refs = detect_laws(text)
        assert any(r.article_no == 28 and r.role == "penalty" for r in refs)

    def test_definition_role(self):
        refs = detect_laws("食安法第3條所稱食品,係指供人飲食或咀嚼之產品。")
        assert any(r.article_no == 3 and r.role == "definition" for r in refs)

    def test_relative_reference_with_hint(self):
        text = "本法第10條第1項規定:食品業者之設廠登記..."
        refs = detect_laws(text, primary_law_hint="食安法第10條")
        assert any(r.article_no == 10 and r.law_name == "食安法" for r in refs)

    def test_relative_reference_without_hint_ignored(self):
        assert detect_laws("本法第10條第1項規定...") == []

    def test_article_with_suffix(self):
        refs = detect_laws("依食安法第15條之一規定,食品業者應建立追溯系統。")
        assert any(r.article_full == "15之一" and r.article_no == 15 for r in refs)

    def test_health_food_law(self):
        text = "依健康食品管理法第14條規定..."
        refs = detect_laws(text)
        assert any(r.law_name == "健康食品管理法" and r.article_no == 14 for r in refs)

    def test_dedup_same_reference(self):
        text = "依食安法第28條規定;另依食安法第28條規定。"
        refs = detect_laws(text)
        assert len([r for r in refs if r.article_no == 28]) == 1


class TestPrimaryLawToRef:
    def test_food_safety(self):
        ref = primary_law_to_ref("食安法第28條")
        assert ref is not None
        assert ref.law_name == "食安法"
        assert ref.article_no == 28
        assert ref.article_full == "28"

    def test_with_suffix_metadata_style(self):
        # data/ 的 primary_law metadata 實際寫法
        ref = primary_law_to_ref("食安法第15條之一")
        assert ref is not None
        assert ref.article_no == 15
        assert ref.article_full == "15之一"

    def test_with_suffix_statute_style(self):
        ref = primary_law_to_ref("食安法第15之一條")
        assert ref is not None
        assert ref.article_full == "15之一"

    def test_health_food(self):
        ref = primary_law_to_ref("健康食品管理法")
        assert ref is not None
        assert ref.law_name == "健康食品管理法"

    def test_none(self):
        assert primary_law_to_ref("") is None
        assert primary_law_to_ref("非標準格式") is None
