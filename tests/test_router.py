"""app/router.py:規則層與 LLM 層的意圖路由。"""
from types import SimpleNamespace

import pytest

from app import router as rt


class ScriptedClient:
    def __init__(self, content):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


@pytest.mark.parametrize("question, intent", [
    ("網路上賣益生菌說能「提升免疫力」,違反哪一條?過去有人因此被罰嗎、罰多少?", "multi_hop"),
    ("食安法第28條常常跟哪些其他條文一起被引用?", "multi_hop"),
    ("幫我審這段文案:本產品有效改善高血壓", "ad_review"),
    ("「補腦益智、增強記憶力」這句放在包裝上可以嗎?", "ad_review"),
    ("天然色素算是食品添加物嗎?需要辦理登錄嗎?", "regulation_qa"),
])
def test_rules_route_clear_signals(question, intent):
    d = rt.route_by_rules(question)
    assert d is not None and d.intent == intent and d.source == "rules"


@pytest.mark.parametrize("question", [
    "本產品有效改善高血壓,三天見效",            # 純文案、沒有問句也沒有審稿提示
    "最近有哪些葉黃素廣告被罰?",                # 只提案例:純找案例還是順帶問規定?規則不猜
    "請問這次的罰款標準是怎麼制定的呢?",        # 有罰則字眼但其實是法規問題
    "裁罰基準裡的資力加權倍數是指什麼?",
])
def test_rules_return_none_when_ambiguous(question):
    assert rt.route_by_rules(question) is None


def test_llm_route_parses_json_and_uses_temperature_zero():
    client = ScriptedClient('{"intent": "ad_review", "reason": "像文案"}')
    d = rt.route_by_llm(client, "本產品有效改善高血壓,三天見效")
    assert d.intent == "ad_review" and d.source == "llm" and d.reason == "像文案"
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("bad", ["not json", '{"intent": "nonsense"}', "", None])
def test_llm_route_falls_back_to_cheapest_path_on_bad_output(bad):
    d = rt.route_by_llm(ScriptedClient(bad), "q")
    assert d.intent == rt.DEFAULT_INTENT and d.source == "fallback"


def test_route_prefers_rules_and_skips_llm():
    client = ScriptedClient('{"intent": "multi_hop"}')
    d = rt.route("食安法第28條常常跟哪些其他條文一起被引用?", client)
    assert d.source == "rules" and client.calls == []


def test_route_uses_llm_only_when_rules_undecided():
    client = ScriptedClient('{"intent": "ad_review"}')
    d = rt.route("本產品有效改善高血壓,三天見效", client)
    assert d.source == "llm" and len(client.calls) == 1


def test_route_without_llm_falls_back():
    d = rt.route("本產品有效改善高血壓,三天見效", client=None)
    assert d.intent == rt.DEFAULT_INTENT and d.source == "fallback"
    d2 = rt.route("本產品有效改善高血壓,三天見效", client=ScriptedClient("{}"), use_llm=False)
    assert d2.source == "fallback"
