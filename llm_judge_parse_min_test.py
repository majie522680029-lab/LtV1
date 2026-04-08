import sys
import types


if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - test stub only
        pass

    openai_stub.OpenAI = OpenAI
    sys.modules["openai"] = openai_stub

from llm_judge import _parse_llm_judge_result


def run_llm_judge_parse_min_test():
    cases = [
        (
            "strict_json",
            '{"is_similar": true, "confidence": 0.91, "reason": "same intent"}',
            True
        ),
        (
            "json_with_alias_keys",
            '{"similar": false, "score": 0.35, "explanation": "different scope"}',
            False
        ),
        (
            "plain_text_with_fields",
            '是否相似: 是; 置信度: 0.88; 理由: 两个问题意图一致',
            True
        ),
    ]
    for name, text, expected in cases:
        parsed = _parse_llm_judge_result(text)
        assert parsed["is_similar"] is expected, f"{name} failed: {parsed}"
        assert 0.0 <= parsed["confidence"] <= 1.0, f"{name} confidence invalid: {parsed}"
        assert parsed["reason"] != "parse failure", f"{name} parse failure: {parsed}"
    print("llm_judge_parse_min_test passed")


if __name__ == "__main__":
    run_llm_judge_parse_min_test()
