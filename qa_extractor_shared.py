import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Callable


def normalize_compare_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def strip_model_thinking(text: str) -> str:
    return re.sub(
        r"(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)",
        "",
        text or "",
    ).strip()


def print_model_raw_output(scene: str, output_text: str) -> None:
    text = (output_text or "").strip()
    if not text:
        print(f"[{scene}] raw_output_empty")
        return
    print(f"[{scene}] raw_output_begin")
    print(text)
    print(f"[{scene}] raw_output_end")


class QAPairExtractor:
    def __init__(
        self,
        client: Any,
        config: Any,
        retryable_exception_checker: Callable[[Exception], bool] | None = None,
        exception_reporter: Callable[[str, Exception], None] | None = None,
    ):
        self.client = client
        self.config = config
        self.retryable_exception_checker = retryable_exception_checker
        self.exception_reporter = exception_reporter

        shared_extract_rules = (
            "1. 只保留脱离当前会话也能成立、且最终可以写入知识库的知识型 QA：业务规则、办理条件、失败原因、查询结论、费用口径、功能点处理依据、是否可办理/续约/顺延。\n"
            "2. 答案必须包含明确业务知识，例如原因、规则、条件、结论、依据；不能只有处理动作。\n"
            "3. 如果答案只是当前会话中的客服反馈、临时状态、查询过程或处理动作，一律删除。例如：我这边查不到、没有短信、请稍等、已转接、需后台处理、需派单核查。\n"
            "4. 答案可综合整段对话中的有效回复，不要求紧跟在问题后；但只能基于原对话，不得臆造。\n"
            "5. 删除补资料、提单、关注工单、已处理请查看、等待转接、内部协同、纯确认、纯重复客户原话等低价值内容。\n"
            "6. 删除手机号、姓名、工号、具体客户、具体当前订单等个案标识；产品名、业务名、规则条件、报错文案、功能点可以保留。\n"
            "7. 如果答案没有新增知识，或者这条问答不能帮助其他用户解决同类问题，则不保留；没有合格结果返回 []。\n"
            "8. 只能输出最终 JSON 数组，禁止输出思考过程、解释、分析、<think> 标签或 Markdown。"
        )

        self.extract_prompt = """你需要从以下客服对话记录中提取**信息咨询类问答对**，并严格遵守以下规则：
1.  仅提取客户提出的**信息寻求类问题**，忽略寒暄、闲聊、个人隐私内容（如手机号、姓名）。
2.  答案必须来自客服的回复，可适当改写口语化表述，但不能添加额外信息。
3.  排除时效性内容（如仅限某日有效）和特定客户的专属内容。
4.  若问题涉及产品，必须在问答对中明确提及产品名称。
5.  输出格式为JSON数组，每个元素包含"question"和"answer"两个字段，不要添加任何其他内容。

对话记录：
{transcript}

提取的问答对："""

        self.local_direct_system_prompt = (
            "你是客服知识库 QA 高精度抽取助手。\n"
            "你的任务是从单条客服对话中直接提取最终可以写入知识库的知识型问答。\n"
            "你必须优先保证精度，宁可少抽，也不要保留不具备知识库价值的问答。\n"
            "直接输出最终结果，不要输出思考过程。\n\n"
            "规则：\n"
            + shared_extract_rules
        )

        self.local_direct_user_prompt = (
            "请直接输出最终可入库的知识库问答。\n"
            "只保留能够脱离当前会话、单独写入知识库、并帮助其他用户解决同类问题的知识型 QA。\n"
            "优先保留：失败原因、办理条件、业务规则、查询结论、费用口径、功能点处理依据、是否可办理/续约/顺延。\n"
            "答案必须包含明确业务知识，例如原因、规则、条件、结论、处理依据。\n"
            "以下内容一律删除：\n"
            "1. 当前客服查询反馈，如“我这边查不到”“没有短信”“系统没显示”；\n"
            "2. 当前会话临时状态，如“请稍等”“已转接”“待核实”；\n"
            "3. 只有处理动作、没有知识解释的话，如“需后台处理”“需派单核查”“联系某部门处理”；\n"
            "4. 仅重复客户原话、没有新增知识的内容。\n"
            "问题整理成用户可直接检索的清晰问句；答案只能保留客服已明确表达的知识结论，不能补充臆测。\n"
            "如果没有合格结果，输出 []。\n"
            "禁止输出思考过程、解释、分析，只能输出 JSON 数组。\n\n"
            "示例1：\n"
            "输入：\n"
            "CUSTOMER: 发票账务打印设置失败怎么处理？\n"
            "SERVICE: 发票账务打印设置失败可能是由于预存款未转分月，需获取{{CBSS-发票账务打印设置}}功能点将订单转分月，增加用户可打金额。\n"
            "输出：\n"
            "[{{\"question\": \"发票账务打印设置失败怎么处理？\", \"answer\": \"发票账务打印设置失败可能是由于预存款未转分月，需获取{{CBSS-发票账务打印设置}}功能点将订单转分月，增加用户可打金额。\"}}]\n\n"
            "示例2：\n"
            "输入：\n"
            "CUSTOMER: 查不到宽带号码怎么办？\n"
            "SERVICE: 我这边查的话，两个号码没有短信。\n"
            "输出：\n"
            "[]\n\n"
            "示例3：\n"
            "输入：\n"
            "CUSTOMER: 固网解约报错提示活动已经进行过转兑，不允许返销如何处理？\n"
            "SERVICE: 需派单核查处理。\n"
            "输出：\n"
            "[]\n\n"
            "现在处理以下对话：\n{transcript}\n\n只输出 JSON 数组："
        )

        self.local_extract_system_prompt = self.local_direct_system_prompt
        self.local_extract_user_prompt = self.local_direct_user_prompt

        self.local_rewrite_system_prompt = """你是客服知识库 QA 整理助手。
你只负责把已经合格的知识库问答整理成表达更清晰、更规范的最终问答。

硬性规则：
1. 只能基于已有问答内容做整理，不能补充常识、不能臆造。
2. 不要改变原有业务结论。
3. 不要合并多个不同问题。
4. 只能输出严格 JSON 数组，禁止解释和 Markdown 代码块。"""

        self.local_rewrite_user_prompt = """请将下面问答整理成更清晰的最终知识库问答。

要求：
- 问题整理成完整、自然、可检索的句子。
- 如果问题本身已经清楚，尽量少改。
- 可保留必要的具体业务场景，如套餐名、产品名、报错文案、功能点名称。
- 答案只能做轻量整理，不能补充新知识。
- 不要重新判断业务规则，只做表述优化。
- 如果全部内容都不适合作为知识库问答，返回 []。

示例1
输入：
[{{"question": "副卡办理不了", "answer": "通过 APP 办理副卡报错，建议咨询公客销进一步核实。"}}]
输出：
[{{"question": "通过 APP 办理副卡报错怎么办？", "answer": "通过 APP 办理副卡报错时，建议咨询公客销进一步核实。"}}]

示例2
输入：
[{{"question": "郊区专属FTTR两年期赠费包到期后能否自动顺延", "answer": "该赠费包到期后自动顺延"}}]
输出：
[{{"question": "郊区专属FTTR两年期赠费包到期后是否自动顺延？", "answer": "该赠费包到期后自动顺延。"}}]

现在处理以下问答：
{candidate_qa_text}

只输出 JSON 数组："""

        self.local_generic_system_prompt = """你是客服知识库 QA 通用化助手。
你只负责把已经合格的知识库问答改写成更通用、更适合沉淀到知识库的标准问答。

硬性规则：
1. 删除具体号码、具体员工、具体工号、具体客户、具体订单等个案信息。
2. 保留业务核心结论、规则条件、失败原因、功能点依据。
3. 如果问答本身已经足够通用，就少改。
4. 不得删除产品名、业务名、套餐名、报错文案等关键业务标识。
5. 只能输出严格 JSON 数组，禁止解释和 Markdown 代码块。"""

        self.local_generic_user_prompt = """请把下面问答改写成更通用的知识库问答。

改写要求：
- 保留业务核心结论，去掉号码、姓名、工号、具体员工名、具体客户、具体订单等个案细节。
- 如果问题是追问式、碎片式，要整理成完整标准问句。
- 如果问题涉及产品、套餐、业务名称或报错文案，要保留这些关键信息。
- 如果答案本身已经是通用的，不要过度改写。
- 不要合并多个不同问答。

示例1
输入：
[{{"question": "这个号码开户部门是哪里，当时开户是谁的名字，期间有没有过户记录", "answer": "开户员工：周伟，开户部门：北京市分公司社会渠道中心.资源管理中心，暂未查询到过户记录"}}]
输出：
[{{"question": "如何查询号码的开户部门和是否有过户记录？", "answer": "可查询号码的开户部门和是否存在过户记录。"}}]

示例2
输入：
[{{"question": "郊区专属FTTR两年期赠费包（600元两年）到期后能否自动顺延？", "answer": "该赠费包到期后自动顺延。"}}]
输出：
[{{"question": "郊区专属FTTR两年期赠费包（600元两年）到期后是否自动顺延？", "answer": "该赠费包到期后自动顺延。"}}]

现在处理以下问答：
{qa_text}

只输出 JSON 数组："""

    def _parse_qa_output(self, output_text: str) -> list[dict[str, str]]:
        text = (output_text or "").strip()
        if not text:
            return []

        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

        candidate_texts = [text]
        array_start, array_end = text.find("["), text.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            candidate_texts.append(text[array_start:array_end + 1])
        obj_start, obj_end = text.find("{"), text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidate_texts.append(text[obj_start:obj_end + 1])

        for candidate in candidate_texts:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue

            qa_list = parsed
            if isinstance(parsed, dict):
                for key in ("qa_pairs", "items", "data", "result"):
                    if isinstance(parsed.get(key), list):
                        qa_list = parsed.get(key)
                        break

            if not isinstance(qa_list, list):
                continue

            valid_pairs = []
            for qa in qa_list:
                if (
                    isinstance(qa, dict)
                    and "question" in qa
                    and "answer" in qa
                    and isinstance(qa["question"], str)
                    and isinstance(qa["answer"], str)
                    and len(qa["question"].strip()) >= self.config.min_question_length
                ):
                    valid_pairs.append(
                        {
                            "question": qa["question"].strip(),
                            "answer": qa["answer"].strip(),
                        }
                    )

            if valid_pairs or qa_list == []:
                return valid_pairs

        return []

    def _normalize_local_qa_item(self, item: Any) -> dict[str, str] | None:
        if not isinstance(item, dict):
            return None

        question_keys = ("question", "query", "q", "question_text", "问题")
        answer_keys = ("answer", "reply", "a", "answer_text", "答案", "response")
        question = next((item.get(key) for key in question_keys if isinstance(item.get(key), str)), None)
        answer = next((item.get(key) for key in answer_keys if isinstance(item.get(key), str)), None)
        if question is None or answer is None:
            return None

        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return None
        return {"question": question, "answer": answer}

    def _is_low_value_local_question(self, question: str, strict: bool = True) -> bool:
        text = normalize_compare_text(question)
        min_len = max(2, self.config.min_question_length) if not strict else max(4, self.config.min_question_length)
        if len(text) < min_len:
            return True

        low_value_patterns = (
            r"^(你好|您好|在吗|谢谢|好的|请稍等|稍等)$",
        )
        if any(re.search(pattern, text) for pattern in low_value_patterns):
            return True

        intent_signals = (
            "如何", "怎么", "为什么", "为何", "是否", "能否", "可以", "怎么办", "查询", "办理", "开具",
            "打印", "失败", "异常", "套餐", "流量", "账号", "发票", "宽带", "合约", "抵扣", "自动",
            "员工", "工号", "部门", "提醒", "通知", "续约", "上架", "到期", "费用",
        )
        if strict and len(text) < 6 and not any(signal in text for signal in intent_signals):
            return True
        return False

    def _is_low_value_local_answer(self, answer: str, strict: bool = True) -> bool:
        text = normalize_compare_text(answer)
        if len(text) < 2:
            return True

        low_value_patterns = (
            r"^.*已处理.*请查看.*$",
            r"^.*关注.*工单.*$",
            r"^.*提.*工单.*$",
            r"^.*请提供.*$",
            r"^.*请补充.*$",
            r"^.*请上传.*$",
            r"^.*已反馈.*$",
            r"^(请稍等|稍等|请您稍等|好的，请稍等)[。！!？?]*$",
            r"^.*已为您转接.*$",
            r"^.*转接.*请稍等.*$",
        )
        if any(re.search(pattern, text) for pattern in low_value_patterns):
            return True

        if strict:
            strict_patterns = (
                r"^.*请关注.*工单.*$",
            )
            if any(re.search(pattern, text) for pattern in strict_patterns):
                return True
        return False

    def _qa_has_low_information_gain(self, question: str, answer: str) -> bool:
        question_text = re.sub(r"[？?。！，,：:；;\s]", "", normalize_compare_text(question))
        answer_text = re.sub(r"[？?。！，,：:；;\s]", "", normalize_compare_text(answer))
        if not question_text or not answer_text:
            return True
        if question_text == answer_text:
            return True

        ratio = SequenceMatcher(None, question_text, answer_text).ratio()
        knowledge_signals = (
            "可以", "不能", "无法", "需要", "建议", "由于", "因为", "需", "应", "自动", "不会",
            "联系", "咨询", "查询", "显示", "正常", "异常", "报错", "失败", "成功", "顺延",
            "续约", "处理", "规则", "优先级", "费用", "原因",
        )
        if ratio >= 0.82 and not any(signal in answer_text for signal in knowledge_signals):
            return True
        if answer_text in question_text or question_text in answer_text:
            if not any(signal in answer_text for signal in knowledge_signals):
                return True
        return False

    def _filter_local_qa_pairs(self, qa_pairs: list[dict[str, Any]], strict: bool = True) -> list[dict[str, str]]:
        filtered_pairs = []
        seen = set()
        for qa in qa_pairs:
            item = self._normalize_local_qa_item(qa)
            if item is None:
                continue
            if self._is_low_value_local_question(item["question"], strict=strict) or self._is_low_value_local_answer(item["answer"], strict=strict):
                continue
            if self._qa_has_low_information_gain(item["question"], item["answer"]):
                continue
            key = (normalize_compare_text(item["question"]), normalize_compare_text(item["answer"]))
            if key in seen:
                continue
            seen.add(key)
            filtered_pairs.append(item)
        return filtered_pairs

    def _merge_local_qa_pairs(self, *qa_pair_groups: list[dict[str, str]]) -> list[dict[str, str]]:
        merged: dict[str, dict[str, str]] = {}
        for qa_pairs in qa_pair_groups:
            for qa in qa_pairs or []:
                item = self._normalize_local_qa_item(qa)
                if item is None:
                    continue
                key = normalize_compare_text(item["question"])
                if not key:
                    continue
                existing = merged.get(key)
                if existing is None:
                    merged[key] = item
                    continue
                existing_score = len(existing["question"]) + len(existing["answer"])
                item_score = len(item["question"]) + len(item["answer"])
                if item_score > existing_score:
                    merged[key] = item
        return list(merged.values())

    def _repair_local_json_candidate(self, text: str) -> str:
        repaired = (text or "").strip()
        repaired = repaired.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")
        if open_brackets > close_brackets and open_brackets - close_brackets <= 2:
            repaired += "]" * (open_brackets - close_brackets)

        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces and open_braces - close_braces <= 2:
            repaired += "}" * (open_braces - close_braces)

        return repaired

    def _extract_local_json_candidates(self, text: str) -> list[str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []

        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        candidates = [cleaned]
        array_start, array_end = cleaned.find("["), cleaned.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            candidates.append(cleaned[array_start:array_end + 1])

        obj_start, obj_end = cleaned.find("{"), cleaned.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            obj_text = cleaned[obj_start:obj_end + 1]
            candidates.append(obj_text)
            candidates.append(f"[{obj_text}]")

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            repaired = self._repair_local_json_candidate(candidate)
            if repaired and repaired not in seen:
                seen.add(repaired)
                unique_candidates.append(repaired)
        return unique_candidates

    def _loads_local_json_candidate(self, candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except Exception:
            pass

        try:
            import ast
            return ast.literal_eval(candidate)
        except Exception:
            return None

    def _parse_local_qa_output(self, output_text: str) -> list[dict[str, str]]:
        text = (output_text or "").strip()
        if not text:
            return []

        for candidate in self._extract_local_json_candidates(text):
            parsed = self._loads_local_json_candidate(candidate)
            if parsed is None:
                continue

            qa_list = parsed
            if isinstance(parsed, dict):
                for key in ("qa_pairs", "items", "data", "result", "qa_list"):
                    if isinstance(parsed.get(key), list):
                        qa_list = parsed.get(key)
                        break

            if isinstance(qa_list, dict):
                qa_list = [qa_list]
            if not isinstance(qa_list, list):
                continue

            normalized_items = []
            for qa in qa_list:
                item = self._normalize_local_qa_item(qa)
                if item is not None:
                    normalized_items.append(item)

            if normalized_items or qa_list == []:
                return normalized_items

        return []

    def _is_retryable(self, exc: Exception) -> bool:
        return bool(self.retryable_exception_checker and self.retryable_exception_checker(exc))

    def _report_exception(self, scene: str, exc: Exception, request_url: str) -> None:
        if self.exception_reporter is not None:
            self.exception_reporter(scene, exc, request_url=request_url)
            return
        raise exc

    def _call_local_model(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str | None:
        for attempt in range(1, self.config.local_max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.qa_extract_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.local_temperature,
                    max_tokens=max_tokens or self.config.local_max_tokens,
                    top_p=self.config.local_top_p,
                    presence_penalty=self.config.local_presence_penalty,
                    frequency_penalty=self.config.local_frequency_penalty,
                    timeout=self.config.local_timeout,
                )
                output = response.choices[0].message.content.strip()
                print_model_raw_output("shared_local_qa_extract", output)
                return strip_model_thinking(output)
            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.config.local_max_attempts:
                    time.sleep(0.8 * attempt)
                    continue
                self._report_exception(
                    "本地 QA 抽取 API 调用",
                    exc,
                    request_url=f"{self.config.qa_extract_base_url.strip()}/chat/completions",
                )
                return None
        return None

    def _finalize_local_qa_pairs(self, transcript: str, parsed_pairs: list[dict[str, str]], output: str | None = None) -> list[dict[str, str]]:
        if not parsed_pairs:
            print(
                "[local_qa_extract] parse_empty "
                + json.dumps(
                    {
                        "transcript_preview": transcript[:200],
                        "output_preview": (output or "")[:500],
                    },
                    ensure_ascii=False,
                )
            )
            return []

        strict_pairs = self._filter_local_qa_pairs(parsed_pairs, strict=True)
        if strict_pairs:
            return strict_pairs

        relaxed_pairs = self._filter_local_qa_pairs(parsed_pairs, strict=False)
        if relaxed_pairs:
            print(
                "[local_qa_extract] strict_filtered_all_fallback_relaxed "
                + json.dumps(
                    {
                        "transcript_preview": transcript[:200],
                        "parsed_count": len(parsed_pairs),
                        "relaxed_count": len(relaxed_pairs),
                        "parsed_preview": parsed_pairs[:2],
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(
                "[local_qa_extract] filtered_all "
                + json.dumps(
                    {
                        "transcript_preview": transcript[:200],
                        "parsed_count": len(parsed_pairs),
                        "parsed_preview": parsed_pairs[:2],
                    },
                    ensure_ascii=False,
                )
            )
        return relaxed_pairs

    def _extract_high_value_pairs_local(self, transcript: str) -> list[dict[str, str]]:
        prompt = self.local_direct_user_prompt.format(transcript=transcript.strip())
        output = self._call_local_model(self.local_direct_system_prompt, prompt)
        if output is None:
            return []
        parsed_pairs = self._parse_local_qa_output(output)
        return self._finalize_local_qa_pairs(transcript, parsed_pairs, output)

    def _rewrite_pairs_local(self, qa_pairs: list[dict[str, str]]) -> list[dict[str, str]]:
        if not qa_pairs:
            return []
        candidate_qa_text = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
        output = self._call_local_model(
            self.local_rewrite_system_prompt,
            self.local_rewrite_user_prompt.format(candidate_qa_text=candidate_qa_text),
        )
        if output is None:
            return qa_pairs
        rewritten_pairs = []
        seen = set()
        for qa in self._parse_local_qa_output(output):
            item = self._normalize_local_qa_item(qa)
            if item is None:
                continue
            key = (normalize_compare_text(item["question"]), normalize_compare_text(item["answer"]))
            if key in seen:
                continue
            seen.add(key)
            rewritten_pairs.append(item)
        return rewritten_pairs or qa_pairs

    def _genericize_pairs_local(self, qa_pairs: list[dict[str, str]]) -> list[dict[str, str]]:
        if not qa_pairs:
            return []
        genericized_pairs = []
        for qa in qa_pairs:
            qa_text = json.dumps([qa], ensure_ascii=False, indent=2)
            output = self._call_local_model(
                self.local_generic_system_prompt,
                self.local_generic_user_prompt.format(qa_text=qa_text),
                max_tokens=196,
            )
            if output is None:
                genericized_pairs.append(qa)
                continue
            parsed_pairs = []
            seen = set()
            for item in self._parse_local_qa_output(output):
                normalized = self._normalize_local_qa_item(item)
                if normalized is None:
                    continue
                key = (normalize_compare_text(normalized["question"]), normalize_compare_text(normalized["answer"]))
                if key in seen:
                    continue
                seen.add(key)
                parsed_pairs.append(normalized)
            if parsed_pairs:
                genericized_pairs.extend(parsed_pairs)
            else:
                genericized_pairs.append(qa)
        return self._merge_local_qa_pairs(genericized_pairs)

    def _extract_final_local_qa(self, transcript: str) -> list[dict[str, str]]:
        strict_pairs = self._extract_high_value_pairs_local(transcript)
        filtered_pairs = self._filter_local_qa_pairs(strict_pairs, strict=True)
        rewritten_pairs = self._rewrite_pairs_local(filtered_pairs)
        return self._genericize_pairs_local(rewritten_pairs)

    def extract_qa_from_transcript(self, transcript: str) -> list[dict[str, str]]:
        if not transcript or not isinstance(transcript, str) or len(transcript.strip()) < self.config.min_dialog_length:
            return []

        if self.config.use_local_model_branch:
            return self._extract_final_local_qa(transcript)

        formatted_prompt = self.extract_prompt.format(transcript=transcript.strip())

        try:
            response = self.client.chat.completions.create(
                model=self.config.qa_extract_model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )

            raw_output = (response.choices[0].message.content or "").strip()
            print_model_raw_output("shared_online_qa_extract", raw_output)
            output = strip_model_thinking(raw_output)
            return self._parse_qa_output(output)
        except Exception as exc:
            self._report_exception(
                "QA 抽取 API 调用",
                exc,
                request_url=f"{self.config.qa_extract_base_url.strip()}/chat/completions",
            )
            return []
