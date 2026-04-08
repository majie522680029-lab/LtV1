import hashlib
import os
import re, json
import time
import httpx
from openai import OpenAI


def build_deepseek_client(api_key, base_url="https://api.deepseek.com"):
    """构建 DeepSeek OpenAI 兼容客户端"""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False)
    )


def _classify_judge_exception(e):
    """将请求异常归因成可读原因，便于定位网络/代理/鉴权问题。"""
    err_name = type(e).__name__
    err_text = str(e)
    err_text_l = err_text.lower()

    if err_name in ("AuthenticationError", "PermissionDeniedError") or "401" in err_text_l:
        return "judge request error: 401 unauthorized (api key 无效/过期)"
    if err_name == "RateLimitError" or "429" in err_text_l:
        return "judge request error: 429 rate_limited (请求过快或额度不足)"
    if err_name in ("APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout") or "timed out" in err_text_l:
        return "judge request error: timeout (网络超时，请稍后重试)"
    if "proxy" in err_text_l or "tunnel" in err_text_l:
        return "judge request error: proxy (代理链路异常)"
    if (
        "name or service not known" in err_text_l
        or "temporary failure in name resolution" in err_text_l
        or "nodename nor servname provided" in err_text_l
    ):
        return "judge request error: dns (域名解析失败)"
    if err_name in ("APIConnectionError", "APIError", "ConnectError", "ConnectionError"):
        return "judge request error: connection (无法连接到 API)"
    return f"judge request error: {err_name}"


def _is_retryable_judge_exception(e):
    """仅对网络瞬时异常重试，避免对 401/429 盲目重试。"""
    err_name = type(e).__name__
    err_text_l = str(e).lower()
    if "401" in err_text_l or "429" in err_text_l:
        return False
    return (
        err_name in ("APIConnectionError", "APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout", "ConnectError")
        or "timed out" in err_text_l
        or "connection reset" in err_text_l
        or "temporarily unavailable" in err_text_l
    )


def triage_gate_decision(sim, t_low, t_high):
    """双阈值三分流决策：HIGH 直接相似，LOW 直接不相似，MID 调用 LLM。"""
    if t_high <= t_low:
        raise ValueError(f"invalid triage thresholds: T_HIGH({t_high}) must be greater than T_LOW({t_low})")

    if sim >= t_high:
        return {
            "triage": "HIGH",
            "llm_called": False,
            "final_decision": True,
            "reason": "sim >= T_HIGH"
        }
    if sim < t_low:
        return {
            "triage": "LOW",
            "llm_called": False,
            "final_decision": False,
            "reason": "sim < T_LOW"
        }
    return {
        "triage": "MID",
        "llm_called": True,
        "final_decision": None,
        "reason": "T_LOW <= sim < T_HIGH"
    }


def _parse_llm_judge_result(raw_text: str):
    default_result = {"is_similar": False, "confidence": 0.0, "reason": "parse failure"}

    try:
        text = (raw_text or "").strip()

        # 1) 去掉 ```json ... ``` 代码块包装
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        plain_text = text

        # 2) 截取第一个 { 到最后一个 }（处理前后夹带解释文字）
        l, r = text.find("{"), text.rfind("}")
        json_text = text[l:r+1] if (l != -1 and r != -1 and r > l) else None

        def _normalize_dict(data_obj):
            if not isinstance(data_obj, dict):
                return None
            normalized = dict(data_obj)
            if "is_similar" not in normalized and "similar" in normalized:
                normalized["is_similar"] = normalized["similar"]
            if "confidence" not in normalized and "score" in normalized:
                normalized["confidence"] = normalized["score"]
            if "reason" not in normalized and "explanation" in normalized:
                normalized["reason"] = normalized["explanation"]
            return normalized

        data = None
        if json_text is not None:
            try:
                parsed = json.loads(json_text)
                data = _normalize_dict(parsed)
            except Exception:
                data = None

        if data is None:
            # 兜底：允许模型输出中带自然语言，尽量从文本中抽取字段
            is_sim_match = re.search(
                r"(?:\"?is_similar\"?\s*[:=]\s*|是否相似[:：]?\s*)(true|false|1|0|yes|no|是|否|相似|不相似)",
                plain_text,
                flags=re.IGNORECASE
            )
            conf_match = re.search(
                r"(?:\"?confidence\"?\s*[:=]\s*|置信度[:：]?\s*)([0-9]+(?:\.[0-9]+)?)",
                plain_text,
                flags=re.IGNORECASE
            )
            reason_match = re.search(
                r"(?:\"?reason\"?\s*[:=]\s*|理由[:：]?\s*)(.+)$",
                plain_text,
                flags=re.IGNORECASE | re.DOTALL
            )
            if is_sim_match:
                is_sim_raw = is_sim_match.group(1).strip().lower()
                if is_sim_raw in ("true", "1", "yes", "是", "相似"):
                    is_sim = True
                elif is_sim_raw in ("false", "0", "no", "否", "不相似"):
                    is_sim = False
                else:
                    return default_result
                conf = float(conf_match.group(1)) if conf_match else (0.8 if is_sim else 0.2)
                if conf > 1.0:
                    conf = conf / 100.0 if conf <= 100 else 1.0
                conf = min(max(conf, 0.0), 1.0)
                reason = (reason_match.group(1).strip() if reason_match else "parsed from plain text")
                reason = reason.strip("`\"' ")
                reason_words = reason.split()
                if len(reason_words) > 30:
                    reason = " ".join(reason_words[:30])
                return {"is_similar": is_sim, "confidence": conf, "reason": reason}
            return default_result

        # 3) 必要字段检查
        for k in ("is_similar", "confidence", "reason"):
            if k not in data:
                return default_result

        # 4) 类型纠正：允许 "true"/"false"、"0.83"
        is_sim = data["is_similar"]
        if isinstance(is_sim, str):
            if is_sim.lower() in ("true", "yes", "1"):
                is_sim = True
            elif is_sim.lower() in ("false", "no", "0"):
                is_sim = False
            else:
                return default_result
        if not isinstance(is_sim, bool):
            return default_result

        conf = data["confidence"]
        if isinstance(conf, str):
            conf = conf.strip()
            conf = float(conf)  # 允许 "0.83"
        if not isinstance(conf, (int, float)):
            return default_result
        conf = float(conf)
        if not (0.0 <= conf <= 1.0):
            return default_result

        reason = data["reason"]
        if not isinstance(reason, str):
            reason = str(reason)

        # 你原来限制 30 个词：中文会不分词，这里保留你的逻辑
        reason_words = reason.split()
        if len(reason_words) > 30:
            reason = " ".join(reason_words[:30])

        return {"is_similar": is_sim, "confidence": conf, "reason": reason}

    except Exception:
        return default_result


def llm_judge_is_similar(client, rep_question, rep_answer, cand_question, cand_answer, model, base_url=None):
    """调用 LLM Judge 判断是否相似"""
    system_prompt = (
        "你是客服知识库去重的严格裁判，返回的原因必须要是中文。"
        "请保守判断：不确定时判定为不相似。"
        "你必须只输出一行严格JSON，禁止任何解释、前后缀、Markdown代码块。"
        "输出格式必须完全等于："
        "{\"is_similar\": false, \"confidence\": 0.0, \"reason\": \"...\"}"
    )
    user_prompt = (
        "[PAIR A - 新的代表性问答]\n"
        f"Q: {rep_question}\n"
        f"A: {rep_answer}\n\n"
        "[PAIR B - 知识库已有候选问答]\n"
        f"Q: {cand_question}\n"
        f"A: {cand_answer}\n\n"
        "只返回 JSON，包含以下键：\n"
        "is_similar (bool),\n"
        "confidence (0-1 数值),\n"
        "reason (不超过 30 个词，对于问题的描述，主语必须是“新问题”或“旧问题”)\n\n"
        "规则：\n"
        "- 仅当两者表达同一用户意图，且多数场景可复用同一知识库条目时，is_similar 才为 true。\n"
        "- 如果只是相关但非同一意图，is_similar=false。\n"
        "- 不确定时，is_similar=false 且 confidence<=0.6。"
    )

    default_result = {"is_similar": False, "confidence": 0.0, "reason": "parse failure"}
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                timeout=30
            )
            content = response.choices[0].message.content.strip()
            parsed = _parse_llm_judge_result(content)
            if parsed.get("reason") == "parse failure":
                print(f"[llm_judge] parse_failure raw_content={content[:500]}")
            return parsed
        except Exception as e:
            if _is_retryable_judge_exception(e) and attempt < max_attempts:
                print(f"[llm_judge] transient_error attempt={attempt}/{max_attempts}: {type(e).__name__}, retrying...")
                time.sleep(0.8 * attempt)
                continue
            reason = _classify_judge_exception(e)
            proxy_info = (
                f"http_proxy={os.getenv('http_proxy') or os.getenv('HTTP_PROXY')}, "
                f"https_proxy={os.getenv('https_proxy') or os.getenv('HTTPS_PROXY')}, "
                f"base_url={base_url or 'N/A'}"
            )
            print(f"[llm_judge] request_error={type(e).__name__}: {str(e)} | {proxy_info}")
            return {"is_similar": False, "confidence": 0.0, "reason": reason}


def judge_with_cache(
    st_session_state,
    rep_question,
    rep_answer,
    cand_question,
    cand_answer,
    model,
    api_key,
    client=None,
    base_url="https://api.deepseek.com"
):
    """使用缓存的 LLM Judge 判断，避免重复调用"""
    if not api_key:
        return {"is_similar": False, "confidence": 0.0, "reason": "missing api key"}

    if "judge_cache" not in st_session_state:
        st_session_state["judge_cache"] = {}

    cache_key_payload = {
        "rep_question": rep_question,
        "rep_answer": rep_answer,
        "cand_question": cand_question,
        "cand_answer": cand_answer,
        "model": model
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if cache_key in st_session_state["judge_cache"]:
        return st_session_state["judge_cache"][cache_key]

    if client is None:
        client = build_deepseek_client(api_key, base_url=base_url)

    result = llm_judge_is_similar(
        client,
        rep_question,
        rep_answer,
        cand_question,
        cand_answer,
        model,
        base_url=base_url
    )
    # 网络/鉴权/限流等请求异常不缓存，避免一次异常导致本轮后续一直命中失败缓存
    if isinstance(result.get("reason"), str) and result["reason"].startswith("judge request error:"):
        return result
    st_session_state["judge_cache"][cache_key] = result
    return result
