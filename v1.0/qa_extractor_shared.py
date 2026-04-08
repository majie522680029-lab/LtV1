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
            "1. 只保留脱离当前会话也能成立的知识型 QA：业务规则、办理条件、失败原因、处理路径、查询结论、费用/优先级口径。\n"
            "2. 答案可综合整段对话中的有效回复，不要求紧跟在问题后；但只能基于原对话，不得臆造。\n"
            "3. 删除补资料、提单、关注工单、已处理请查看、等待转接、内部协同、纯确认等低价值话术。\n"
            "4. 删除手机号、姓名、工号、具体客户等个案标识；产品名、业务名、规则条件可以保留。\n"
            "5. 如果答案没有新增知识，或问答只是重复/推进会话，不保留；没有合格结果返回 []。\n"
            "6. 不要输出思考过程、分析过程、<think> 标签、解释、Markdown；只输出最终 JSON 数组。"
        )

        local_extract_examples = """示例1
输入对话：
CUSTOMER: 副卡办理不了
SERVICE: 如果通过 APP 办理副卡报错，建议咨询公客销进一步核实。
正确输出：
[{{"question": "通过 APP 办理副卡报错怎么办？", "answer": "如果通过 APP 办理副卡报错，建议咨询公客销进一步核实。"}}]

示例2
输入对话：
CUSTOMER: 欠费前有没有提醒
SERVICE: 这个问题需要后台专家协助核查，已为您转接，请稍等。
正确输出：
[]"""

        self.extract_prompt = """你需要从以下客服对话记录中提取**信息咨询类问答对**，并严格遵守以下规则：
1.  仅提取客户提出的**信息寻求类问题**，忽略寒暄、闲聊、个人隐私内容（如手机号、姓名）。
2.  答案必须来自客服的回复，可适当改写口语化表述，但不能添加额外信息。
3.  排除时效性内容（如仅限某日有效）和特定客户的专属内容。
4.  若问题涉及产品，必须在问答对中明确提及产品名称。
5.  输出格式为JSON数组，每个元素包含"question"和"answer"两个字段，不要添加任何其他内容。

对话记录：
{transcript}

提取的问答对："""

        self.local_extract_system_prompt = (
            "你是客服知识库 QA 抽取助手。\n"
            "你的任务是从单条客服对话中直接提取最终可用的知识库问答。\n"
            "直接输出最终结果，不要输出思考过程。\n\n"
            "规则：\n"
            + shared_extract_rules
        )

        self.local_extract_user_prompt = (
            "请直接输出可复用的知识库问答。\n"
            "只保留脱离会话也成立的知识；答案可来自整段对话，不要求紧跟问题。\n"
            "删除这类内容：补资料、提单、关注工单、已处理请查看、等待转接、内部协同、仅重复客户原话。\n"
            "问题尽量改写成清晰问句，但产品或业务名称是关键信息时必须保留。\n"
            "如果答案没有新增知识，或没有合格结果，输出 []。\n"
            "禁止输出思考过程、解释、分析，只能输出 JSON 数组。\n\n"
            "输出格式：\n"
            "[{{\"question\": \"...\", \"answer\": \"...\"}}]\n\n"
            + local_extract_examples
            + "\n\n现在处理以下对话：\n{transcript}\n\n只输出 JSON 数组："
        )

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

    def _extract_final_local_qa(self, transcript: str) -> list[dict[str, str]]:
        prompt = self.local_extract_user_prompt.format(transcript=transcript.strip())
        output = self._call_local_model(self.local_extract_system_prompt, prompt)
        if output is None:
            return []
        parsed_pairs = self._parse_local_qa_output(output)
        return self._finalize_local_qa_pairs(transcript, parsed_pairs, output)

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

            output = strip_model_thinking(response.choices[0].message.content or "")
            return self._parse_qa_output(output)
        except Exception as exc:
            self._report_exception(
                "QA 抽取 API 调用",
                exc,
                request_url=f"{self.config.qa_extract_base_url.strip()}/chat/completions",
            )
            return []
