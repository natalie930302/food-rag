"""Metadata 推導測試。"""
from ingest.extractors.metadata import derive_metadata, infer_kind


class TestDeriveMetadata:
    def test_with_law_article(self):
        path = "data/raw/食藥署/食安法/食安法第28條/「準則」Q&A.pdf"
        md = derive_metadata(path, root="data/raw")
        assert md["category"] == "食藥署"
        assert md["primary_law"] == "食安法第28條"

    def test_with_subtopic(self):
        path = "data/raw/食藥署/食安法/食安法第8條/GHP/GHP 液蛋/液蛋指引.txt"
        md = derive_metadata(path, root="data/raw")
        assert md["primary_law"] == "食安法第8條"
        assert "GHP" in md["subtopic"]

    def test_health_food_law(self):
        path = "data/raw/食藥署/健康食品管理法/健康食品管理法.pdf"
        md = derive_metadata(path, root="data/raw")
        assert md["primary_law"] == "健康食品管理法"

    def test_no_law(self):
        path = "data/raw/食藥署/額外補充、指引/包裝食品宣稱素食.pdf"
        md = derive_metadata(path, root="data/raw")
        assert md["primary_law"] is None
        assert "額外補充" in (md["subtopic"] or "")

    def test_with_suffix_law(self):
        path = "data/raw/食藥署/食安法/食安法第15條之一/原料規定.pdf"
        md = derive_metadata(path, root="data/raw")
        assert md["primary_law"] == "食安法第15條之一"


class TestInferKind:
    def test_qa(self):
        assert infer_kind("食品衛生標準Q&A.pdf", None) == "qa"
        assert infer_kind("某問答集.pdf", None) == "qa"

    def test_standard(self):
        assert infer_kind("食品中微生物衛生標準.pdf", "食安法第17條") == "standard"

    def test_law_text(self):
        assert infer_kind("健康食品管理法.pdf", "健康食品管理法") == "law_text"

    def test_guide(self):
        assert infer_kind("GHP 液蛋指引.pdf", "食安法第8條") == "guide"
