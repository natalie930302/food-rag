"""app/llm.py infer_verdict:等級必須跟報告內文一致,不能只靠關鍵字表。"""
import pytest

from app.llm import infer_verdict


@pytest.mark.parametrize("answer, expected", [
    ("一、法規說明 …涉及醫療效能… \n\nVERDICT: 高風險", "high"),
    ("**VERDICT:** 高風險", "high"),                       # 粗體
    ("VERDICT：有疑慮", "medium"),                         # 全形冒號
    ("文案符合規定,無須修改。VERDICT 合規", "low"),         # 沒冒號
])
def test_verdict_line_formats(answer, expected):
    assert infer_verdict([], answer, "") == expected


def test_report_wording_fallback_when_no_verdict_line():
    # 2026/09 實測案例:「改善高血壓」不在關鍵字表,LLM 又沒吐 VERDICT 行 → 以前會判 low,跟內文矛盾
    answer = "一、法規說明\n該文案宣稱「有效改善高血壓」涉及醫療效能,違反《食品安全衛生管理法》第28條。"
    assert infer_verdict([], answer, "本產品有效改善高血壓,三天見效") == "high"


def test_keyword_layer_still_raises_floor():
    assert infer_verdict(["降血糖"], "VERDICT: 合規", "") == "high"      # 關鍵字命中 medical → 至少 high
    assert infer_verdict(["豐胸"], "VERDICT: 合規", "") == "medium"      # function 類 → medium


def test_compliant_phrase_downgrades_only_without_keywords():
    ad = "補充葉黃素,有助於維持正常視覺功能"
    assert infer_verdict([], "VERDICT: 有疑慮", ad) == "low"            # 無關鍵字 + 白名單用語 → low
    assert infer_verdict(["恢復視力"], "VERDICT: 有疑慮", ad) == "high"  # 有 medical 關鍵字就不降
