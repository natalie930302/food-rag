"""法條偵測測試。"""
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
若違反第25條第2項,處新台幣3萬元以上罰鍰。
另依第28條相關規定...
"""
        refs = detect_laws(text)
        articles = {r.article_no for r in refs}
        assert 22 in articles
        assert 25 in articles
        assert 28 in articles

    def test_role_detection(self):
        text = "違反第28條者,處新臺幣4萬元以上400萬元以下罰鍰。"
        refs = detect_laws(text)
        # 28 條此處角色應為 penalty
        assert any(r.article_no == 28 and r.role == "penalty" for r in refs)

    def test_relative_reference_with_hint(self):
        text = "本法第10條第1項規定:食品業者之設廠登記..."
        refs = detect_laws(text, primary_law_hint="食安法第10條")
        assert any(r.article_no == 10 for r in refs)

    def test_health_food_law(self):
        text = "依健康食品管理法第14條規定..."
        refs = detect_laws(text)
        assert any(r.law_name == "健康食品管理法" for r in refs)


class TestPrimaryLawToRef:
    def test_food_safety(self):
        ref = primary_law_to_ref("食安法第28條")
        assert ref is not None
        assert ref.law_name == "食安法"
        assert ref.article_no == 28

    def test_with_suffix(self):
        ref = primary_law_to_ref("食安法第15條之一")
        assert ref is not None
        assert ref.article_full == "15之一"

    def test_health_food(self):
        ref = primary_law_to_ref("健康食品管理法")
        assert ref is not None
        assert ref.law_name == "健康食品管理法"

    def test_none(self):
        assert primary_law_to_ref("") is None
        assert primary_law_to_ref("非標準格式") is None
