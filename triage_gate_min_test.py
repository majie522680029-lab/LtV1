import sys
import types


if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - 仅用于测试时绕过第三方依赖
        pass

    openai_stub.OpenAI = OpenAI
    sys.modules["openai"] = openai_stub

from llm_judge import triage_gate_decision


def simulate_candidate(sim, t_low, t_high, llm_judge_fn):
    gate = triage_gate_decision(sim, t_low, t_high)
    llm_called = gate["llm_called"]
    if llm_called:
        llm_result = llm_judge_fn()
        final_decision = bool(llm_result["is_similar"])
    else:
        final_decision = bool(gate["final_decision"])
    return {
        "triage": gate["triage"],
        "llm_called": llm_called,
        "final_decision": final_decision
    }


def run_triage_gate_min_test():
    t_low = 0.80
    t_high = 0.88
    epsilon = 1e-6

    high_case = simulate_candidate(
        sim=t_high + epsilon,
        t_low=t_low,
        t_high=t_high,
        llm_judge_fn=lambda: {"is_similar": False}
    )
    assert high_case["triage"] == "HIGH"
    assert high_case["llm_called"] is False
    assert high_case["final_decision"] is True

    low_case = simulate_candidate(
        sim=t_low - epsilon,
        t_low=t_low,
        t_high=t_high,
        llm_judge_fn=lambda: {"is_similar": True}
    )
    assert low_case["triage"] == "LOW"
    assert low_case["llm_called"] is False
    assert low_case["final_decision"] is False

    mid_case = simulate_candidate(
        sim=(t_low + t_high) / 2.0,
        t_low=t_low,
        t_high=t_high,
        llm_judge_fn=lambda: {"is_similar": True}
    )
    assert mid_case["triage"] == "MID"
    assert mid_case["llm_called"] is True
    assert mid_case["final_decision"] is True

    print("triage_gate_min_test passed")
    print({"high_case": high_case, "low_case": low_case, "mid_case": mid_case})


if __name__ == "__main__":
    run_triage_gate_min_test()
