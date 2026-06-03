import json
import os
import re
import time
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except OSError:
        return
    except Exception:
        return


def _env_flag_enabled(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _get_debug_trace_file_path():
    explicit_path = os.getenv("QA_DEBUG_TRACE_FILE", "").strip()
    if explicit_path:
        return os.path.abspath(explicit_path)
    return os.path.abspath(os.path.join(os.getcwd(), "qa_debug_trace.log"))


def _write_debug_trace_file(text):
    try:
        trace_file_path = _get_debug_trace_file_path()
        trace_dir = os.path.dirname(trace_file_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        with open(trace_file_path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
    except Exception:
        return


def _format_debug_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def _format_count_line(payload, *keys):
    parts = []
    for key in keys:
        if key in payload:
            parts.append(f"{key}={payload.get(key)}")
    return "；".join(parts)


def _format_readable_debug_trace(data):
    stage = data.get("stage", "")
    payload = data.get("payload") or {}
    title_map = {
        "dialog_input": "原始对话",
        "dialog_skipped": "跳过对话",
        "local_model_request": "抽取模型 Prompt",
        "local_model_response": "抽取模型输出",
        "candidate_extract_result": "候选抽取结果",
        "direct_extract_result": "直接抽取结果",
        "rewrite_input": "改写输入",
        "rewrite_result": "改写结果",
        "genericize_input": "通用化输入",
        "genericize_item_result": "单条通用化结果",
        "genericize_result": "通用化最终结果",
        "dialog_final_result": "单条对话最终 QA",
        "cluster_input": "聚类输入",
        "cluster_eps": "聚类参数",
        "cluster_dbscan_labels": "聚类标签",
        "cluster_result": "聚类结果",
        "selector_input": "代表性筛选输入",
        "local_selector_request": "代表性筛选模型 Prompt",
        "local_selector_response": "代表性筛选模型输出",
        "selector_result": "代表性筛选结果",
        "representative_review_input": "代表性 QA 复审输入",
        "representative_review_request": "代表性 QA 复审 Prompt",
        "representative_review_response": "代表性 QA 复审模型输出",
        "representative_review_result": "代表性 QA 复审结果",
        "representative_review_summary": "代表性 QA 复审汇总",
        "final_representative_filter": "最终硬过滤",
    }
    lines = [
        "=" * 100,
        f"时间: {data.get('created_at', '')}    PID: {data.get('pid', '')}",
        f"阶段: {stage} - {title_map.get(stage, '处理过程')}",
        "-" * 100,
    ]

    if stage == "dialog_input":
        lines.extend([
            f"模式: {payload.get('mode', '')}    字符数: {payload.get('length', '')}",
            "",
            "【原始对话】",
            _format_debug_value(payload.get("transcript")),
        ])
    elif stage in {"local_model_request", "local_selector_request"}:
        lines.extend([
            f"模型: {payload.get('model', '')}    尝试次数: {payload.get('attempt', '')}    max_tokens: {payload.get('max_tokens', '')}",
        ])
        if payload.get("scene"):
            lines.append(f"场景: {payload.get('scene')}")
        lines.extend([
            "",
            "【System Prompt】",
            _format_debug_value(payload.get("system_prompt") or "(未记录；如需隐藏 prompt 可设置 QA_DEBUG_TRACE_PROMPT=0)"),
            "",
            "【User Prompt】",
            _format_debug_value(payload.get("user_prompt") or "(未记录；如需隐藏 prompt 可设置 QA_DEBUG_TRACE_PROMPT=0)"),
        ])
    elif stage in {"local_model_response", "local_selector_response"}:
        if payload.get("scene"):
            lines.append(f"场景: {payload.get('scene')}")
        lines.extend([
            f"尝试次数: {payload.get('attempt', '')}",
            "",
            "【模型原始输出】",
            _format_debug_value(payload.get("raw_output")),
            "",
            "【去除 thinking 后输出】",
            _format_debug_value(payload.get("cleaned_output")),
        ])
    elif stage in {"candidate_extract_result", "direct_extract_result"}:
        lines.extend([
            _format_count_line(payload, "status", "parsed_count", "filtered_count"),
            "",
            "【模型解析出的 QA】",
            _format_debug_value(payload.get("parsed_pairs")),
            "",
            "【规则过滤后保留 QA】",
            _format_debug_value(payload.get("filtered_pairs") or payload.get("pairs")),
        ])
    elif stage == "rewrite_input":
        lines.extend([
            _format_count_line(payload, "input_count"),
            "",
            "【改写前 QA】",
            _format_debug_value(payload.get("input_pairs")),
        ])
    elif stage == "rewrite_result":
        lines.extend([
            _format_count_line(payload, "status", "parsed_count", "output_count"),
            "",
            "【改写模型解析结果】",
            _format_debug_value(payload.get("parsed_pairs")),
            "",
            "【改写后 QA】",
            _format_debug_value(payload.get("output_pairs")),
        ])
    elif stage == "genericize_input":
        lines.extend([
            _format_count_line(payload, "input_count"),
            "",
            "【通用化前 QA】",
            _format_debug_value(payload.get("input_pairs")),
        ])
    elif stage == "genericize_item_result":
        lines.extend([
            _format_count_line(payload, "status"),
            "",
            "【输入 QA】",
            _format_debug_value(payload.get("input_qa")),
            "",
            "【模型解析结果】",
            _format_debug_value(payload.get("parsed_pairs")),
            "",
            "【本步输出 QA】",
            _format_debug_value(payload.get("output_pairs")),
        ])
    elif stage == "genericize_result":
        lines.extend([
            _format_count_line(payload, "status", "merged_count", "filtered_count"),
            "",
            "【合并后 QA】",
            _format_debug_value(payload.get("merged_pairs")),
            "",
            "【通用化最终保留 QA】",
            _format_debug_value(payload.get("output_pairs")),
        ])
    elif stage == "dialog_final_result":
        lines.extend([
            _format_count_line(payload, "strict_count", "filtered_count", "rewritten_count", "final_count"),
            "",
            "【最终 QA】",
            _format_debug_value(payload.get("final_pairs")),
        ])
    elif stage == "cluster_input":
        lines.extend([
            f"QA 数量: {payload.get('qa_count', '')}",
            "",
            "【参与聚类的问题】",
            _format_debug_value(payload.get("questions")),
        ])
    elif stage == "cluster_eps":
        lines.extend([
            "【聚类参数】",
            _format_debug_value(payload),
        ])
    elif stage == "cluster_dbscan_labels":
        lines.extend([
            "【问题与聚类标签】",
            _format_debug_value(payload.get("question_labels")),
        ])
    elif stage == "cluster_result":
        lines.extend([
            f"簇数量: {payload.get('cluster_count', '')}",
            "",
            "【每个簇大小】",
            _format_debug_value(payload.get("cluster_sizes")),
        ])
    elif stage == "selector_input":
        lines.extend([
            f"簇大小: {payload.get('cluster_size', '')}    输入 QA 数: {payload.get('qa_used_count', '')}    是否截断: {payload.get('input_truncated', '')}",
            "",
            "【送入代表性筛选模型的簇内容】",
            _format_debug_value(payload.get("formatted_input")),
        ])
    elif stage == "selector_result":
        lines.extend([
            _format_count_line(payload, "status", "representative_count"),
            "",
            "【代表性 QA】",
            _format_debug_value(payload.get("representative_qa")),
        ])
    elif stage == "representative_review_input":
        lines.extend([
            f"复审序号: {payload.get('index', '')}",
            "",
            "【待复审 QA】",
            _format_debug_value(payload.get("qa")),
        ])
    elif stage == "representative_review_request":
        lines.extend([
            f"模型: {payload.get('model', '')}    尝试次数: {payload.get('attempt', '')}    max_tokens: {payload.get('max_tokens', '')}",
            "",
            "【System Prompt】",
            _format_debug_value(payload.get("system_prompt")),
            "",
            "【User Prompt】",
            _format_debug_value(payload.get("user_prompt")),
        ])
    elif stage == "representative_review_response":
        lines.extend([
            f"尝试次数: {payload.get('attempt', '')}",
            "",
            "【模型原始输出】",
            _format_debug_value(payload.get("raw_output")),
            "",
            "【去除 thinking 后输出】",
            _format_debug_value(payload.get("cleaned_output")),
        ])
    elif stage == "representative_review_result":
        lines.extend([
            f"是否保留: {payload.get('keep')}    分类: {payload.get('category', '')}",
            f"原因: {payload.get('reason', '')}",
            "",
            "【复审 QA】",
            _format_debug_value(payload.get("qa")),
        ])
    elif stage == "representative_review_summary":
        lines.extend([
            f"输入数量: {payload.get('input_count', '')}    保留数量: {payload.get('kept_count', '')}    删除数量: {payload.get('dropped_count', '')}",
            "",
            "【删除项】",
            _format_debug_value(payload.get("dropped_items")),
        ])
    elif stage == "final_representative_filter":
        lines.extend([
            f"是否跳过: {payload.get('skip')}    原因: {payload.get('reason') or '保留'}",
            "",
            "【检查的代表性 QA】",
            _format_debug_value(payload.get("qa")),
        ])
    else:
        lines.extend([
            "【处理过程】",
            _format_debug_value(payload),
        ])

    lines.extend(["", ""])
    return "\n".join(lines)


def _format_terminal_debug_trace(data):
    terminal_data = dict(data)
    payload = dict(terminal_data.get("payload") or {})
    for key in ("system_prompt", "user_prompt", "prompt", "transcript", "raw_output", "cleaned_output"):
        if isinstance(payload.get(key), str):
            payload[key] = f"<written_to_file chars={len(payload[key])}>"
    terminal_data["payload"] = payload
    try:
        return json.dumps(terminal_data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(terminal_data)


def _debug_trace(stage, payload=None):
    """Controlled terminal/file trace for prompt/result debugging."""
    if not _env_flag_enabled("QA_DEBUG_TRACE"):
        return
    data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "stage": stage,
        "payload": payload or {},
    }
    try:
        terminal_text = _format_terminal_debug_trace(data)
    except Exception:
        terminal_text = str(data)
        file_text = str(data)
    else:
        file_text = _format_readable_debug_trace(data)
    _safe_print(f"[qa_debug_trace] {terminal_text}", flush=True)
    _write_debug_trace_file(file_text)


def _include_prompt_in_trace():
    return os.getenv("QA_DEBUG_TRACE_PROMPT", "").strip().lower() not in {"0", "false", "no", "off"}


def _split_model_thinking_and_result(output_text):
    text = output_text or ""
    thinking_parts = []
    for pattern in (
        r"\<think\>([\s\S]*?)\</think\>",
        r"\<\|thinking\|\>([\s\S]*?)\</\|thinking\|\>",
    ):
        thinking_parts.extend(match.group(1).strip() for match in re.finditer(pattern, text) if match.group(1).strip())
    result = re.sub(
        r"(\<think\>[\s\S]*?\</think\>)|(\<\|thinking\|\>[\s\S]*?\</\|thinking\|\>)",
        "",
        text,
    ).strip()
    return "\n\n".join(thinking_parts).strip(), result


class BGEEmbeddingGenerator:
    def __init__(self, config, warning_reporter=None):
        self.local_model_path = config.bge_model_path
        self.config = config
        self.warning_reporter = warning_reporter or (lambda _message: None)

        if not os.path.exists(self.local_model_path):
            self.warning_reporter(f"⚠️ BGE 模型路径不存在: {self.local_model_path}，将使用在线下载")

        self.model = SentenceTransformer(self.local_model_path, device=config.embedding_device)

    def get_embedding(self, question):
        return self.model.encode(question, normalize_embeddings=True, show_progress_bar=False)


class QAPairExtractor:
    def __init__(
        self,
        client,
        config,
        normalize_compare_text,
        print_model_raw_output,
        report_api_exception,
        is_retryable_api_exception,
    ):
        self.client = client
        self.config = config
        self.normalize_compare_text = normalize_compare_text
        self.print_model_raw_output = print_model_raw_output
        self.report_api_exception = report_api_exception
        self.is_retryable_api_exception = is_retryable_api_exception
        self.dropped_qa_pairs = []
        self._current_source_dialog_text = ""
        self._current_flow_step_no = 0

        self.extract_prompt = """你需要从以下客服对话记录中提取**信息咨询类问答对**，并严格遵守以下规则：
1.  仅提取客户提出的**信息寻求类问题**，忽略寒暄、闲聊、个人隐私内容（如手机号、姓名）。
2.  答案必须来自客服的回复，可适当改写口语化表述，但不能添加额外信息。
3.  排除时效性内容（如仅限某日有效）和特定客户的专属内容。
4.  若问题涉及产品，必须在问答对中明确提及产品名称。
5.  输出格式为JSON数组，每个元素包含"question"和"answer"两个字段，不要添加任何其他内容。

对话记录：
{transcript}

提取的问答对："""

        shared_extract_rules = (
            "1. 只保留脱离当前会话也能成立、且最终可以写入知识库的知识型 QA：业务规则、办理条件、失败原因、查询结论、费用口径、功能点处理依据、是否可办理/续约/顺延。\n"
            "2. 答案必须包含明确业务知识，例如原因、规则、条件、结论、依据；不能只有处理动作。\n"
            "3. 如果答案只是当前会话中的客服反馈、临时状态、查询过程或处理动作，一律删除。例如：我这边查不到、没有短信、请稍等、已转接、需后台处理、需派单核查。\n"
            "4. 答案可综合整段对话中的有效回复，不要求紧跟在问题后；但只能基于原对话，不得臆造。\n"
            "5. 删除补资料、提单、关注工单、已处理请查看、等待转接、内部协同、纯确认、纯重复客户原话等低价值内容。\n"
            "6. 删除手机号、姓名、工号、具体客户、具体当前订单等个案标识；产品名、业务名、规则条件、报错文案、功能点可以保留。\n"
            "7. 如果答案没有新增知识，或者这条问答不能帮助其他用户解决同类问题，则不保留；没有合格结果返回 []。\n"
            "8. 只能输出最终 JSON 数组，禁止输出思考过程、解释、分析、<think> 标签或 Markdown。"
            "9. 不保留个案账单/话单/费用查询结论：如果问题包含具体月份、具体金额、具体号码、具体账期、具体话单，且答案只是解释该用户本次费用/账单/话单原因，则删除。\n"
            "10. 删除等待系统同步类答案：如果答案只是“等待系统同步/稍后查看/系统会更新/已协助处理/刷新后查看”，但没有说明稳定规则、触发条件、同步时限、用户或客服可执行步骤，则不保留。\n"
            "11. 严禁照搬示例中的产品名、套餐名、金额、期限或答案；最终 QA 的具体业务对象和数字必须来自当前对话原文。\n"
        )

        self.local_direct_system_prompt = (
            "你是客服知识库 QA 高精度抽取助手。\n"
            "你的任务是从单条客服对话中直接提取最终可以写入知识库的知识型问答。\n"
            "你必须优先保证精度，宁可少抽，也不要保留不具备知识库价值的问答。\n"
            "直接输出最终结果，不要输出思考过程；日志会记录每一步结果用于调试。\n\n"
            "规则：\n"
            + shared_extract_rules
        )

        self.local_direct_user_prompt = (
            "请直接输出最终可入库的知识库问答。\n"
            "只保留能够脱离当前会话、单独写入知识库、并帮助其他用户解决同类问题的知识型 QA。\n"
            "优先保留：失败原因、办理条件、业务规则、查询结论、费用口径、功能点处理依据、是否可办理/续约/顺延。\n"
            "答案必须包含明确业务知识，例如原因、规则、条件、结论、处理依据。\n"
            "问题提炼必须尽量贴近客户原始咨询场景，不能把有业务限定词的问题改写成范围过大的泛问题。\n"
            "以下内容一律删除：\n"
            "1. 当前客服查询反馈，如“我这边查不到”“没有短信”“系统没显示”；\n"
            "2. 当前会话临时状态，如“请稍等”“已转接”“待核实”；\n"
            "3. 只有处理动作、没有知识解释的话，如“需后台处理”“需派单核查”“联系某部门处理”。特别注意：即使答案提到后台/中台/政企「会执行某个操作」，只要核心路径是转派他人处理就不能保留——因为这不能帮助其他客服直接解决问题；\n"
            "4. 仅重复客户原话、没有新增知识的内容。\n"
            "5. 具体用户的账单、话费、流量费、短信费、发票、话单查询结论，包含具体月份、具体金额、具体号码、具体账期、具体订单流水号的问题，且答案只是本次查询结果，没有通用规则。\n"
            "6. 个案查询结果类：问题本质是在查询某个具体用户、具体号码、具体订单、具体时间段的数据；答案主要是该用户的查询结果、核查结果或系统回显，而不是通用业务规则。”；\n"
            "7. 数据回显类：答案包含大量具体数字、日期、月份、费用、流量、业务类型、订单号、工单号、账号等，仅用于说明某一次查询结果。”；\n"
            "8. 个性化核查类：答案仅表明“已核实正常”“当前状态为xx”“已处理”“当前系统显示xx”，不能为未来相似问题提供稳定处理依据。”；\n"
            "9. 如果客户问题中出现了具体产品名、套餐名、资费金额、档位、折扣、活动名称、功能包名称、账本名称、功能点名称、报错文案，提炼问题时必须保留这些关键信息。\n"
            "10. 如果去掉这些关键信息会导致问题范围明显变大、答案变得片面，则不能泛化，必须保留原始业务限定词。\n"
            "11. 优先生成“带业务限定词”的问题，例如“沃派套餐48元优化版为什么无法取消”“0元10G定向流量包为什么无法恢复”“提示活动已经进行过转兑，不允许返销如何处理”，不要改成“为什么无法取消”“为什么无法恢复”“报错如何处理”。\n"
            "12. 带具体订单号，具体号码的问题和答案，一律删除”。\n"
            "13. 严禁照搬示例：如果当前对话没有出现某个套餐名、产品名、金额、期限、地区或报错文案，不得把它写入问题或答案。\n"
            "14. 对话为电话录音转写，包含大量口语、语气词（嗯、噢、啊）和 ASR 错误（“那个”→“哪个”、“账号”→“张浩”），提取时需根据上下文推断真实含义。\n"
            "15. “用户：”和“客服：”标签可能不准确，需根据对话语义判断发言人角色。\n"
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
            "示例4：\n"
            "输入：\n"
            "CUSTOMER: 座机七月为什么产生0.22元费用？\n"
            "SERVICE: 呼转产生的，话单真实存在，计费正常，话单是云网交付中心下发的。\n"
            "输出：\n"
            "[]\n\n"
            "示例5：\n"
            "输入：\n"
            "CUSTOMER: 报错提示活动已经进行过转兑，不允许返销，取消WiFi赠费包怎么办？\n"
            "SERVICE: 这个报错需要派单核查处理。\n"
            "输出：\n"
            "[]\n\n"
            "现在处理以下对话：\n{transcript}\n\n只输出 JSON 数组："
        )

        self.local_extract_system_prompt = self.local_direct_system_prompt
        self.local_candidate_user_prompt = self.local_direct_user_prompt

        self.local_rewrite_system_prompt = """你是客服知识库 QA 整理助手。
你只负责把已经合格的知识库问答整理成表达更清晰、更规范的最终问答。

硬性规则：
1. 只能基于已有问答内容做整理，不能补充常识、不能臆造。
2. 不要改变原有业务结论。
3. 不要合并多个不同问题。
4. 只能输出严格 JSON 数组，禁止解释、思考过程和 Markdown 代码块。"""

        self.local_rewrite_user_prompt = """请将下面问答整理成更清晰的最终知识库问答。

要求：
- 问题整理成完整、自然、可检索的句子。
- 如果问题本身已经清楚，尽量少改。
- 必须保留必要的具体业务场景，如套餐名、产品名、资费金额、资费档位、折扣、活动名称、功能包名称、账本名称、报错文案、功能点名称。
- 如果原问题中已经有这些业务限定词，整理时不能删掉、不能弱化成“该产品”“这个业务”“这个报错”之类的泛称。
- 如果删除这些业务限定词会让问题适用范围明显变大或使答案显得片面，则保持原问题，不要为了通顺而泛化。
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

示例3
输入：
[{{"question": "为什么无法取消", "answer": "沃派套餐48元优化版（北京）预存400元用一年折扣合约存在合约计划，合约期内无法取消。"}}]
输出：
[{{"question": "沃派套餐48元优化版（北京）预存400元用一年折扣合约为什么无法取消？", "answer": "沃派套餐48元优化版（北京）预存400元用一年折扣合约存在合约计划，合约期内无法取消。"}}]

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
5. 只能输出严格 JSON 数组，禁止解释、思考过程和 Markdown 代码块。"""

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

示例3
输入：
[{{"question": "座机七月产生0.22元费用的原因是什么？", "answer": "座机七月产生0.22元费用是由于呼转移导致的，话单真实存在且计费正常，话单由云网交付中心下发。"}}]
输出：
[]

现在处理以下问答：
{qa_text}

只输出 JSON 数组："""

    def _parse_qa_output(self, output_text):
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
                    valid_pairs.append({
                        "question": qa["question"].strip(),
                        "answer": qa["answer"].strip(),
                    })

            if valid_pairs or qa_list == []:
                return valid_pairs

        return []

    def _normalize_local_qa_item(self, item):
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

    def _is_low_value_local_question(self, question, strict=True):
        text = self.normalize_compare_text(question)
        min_len = max(2, self.config.min_question_length) if not strict else max(4, self.config.min_question_length)
        if len(text) < min_len:
            return True

        low_value_patterns = (
            r"^(你好|您好|在吗|谢谢|好的|请稍等|稍等)$",
        )
        if any(re.search(pattern, text) for pattern in low_value_patterns):
            return True

        intent_signals = ("如何", "怎么", "为什么", "为何", "是否", "能否", "可以", "怎么办", "查询", "办理", "开具", "打印", "失败", "异常", "套餐", "流量", "账号", "发票", "宽带", "合约", "抵扣", "自动", "员工", "工号", "部门", "提醒", "通知", "续约", "上架", "到期", "费用")
        if strict and len(text) < 6 and not any(signal in text for signal in intent_signals):
            return True
        return False

    def _is_low_value_local_answer(self, answer, strict=True):
        text = self.normalize_compare_text(answer)
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

    def _qa_has_low_information_gain(self, question, answer):
        question_text = re.sub(r"[？?。！，,：:；;\s]", "", self.normalize_compare_text(question))
        answer_text = re.sub(r"[？?。！，,：:；;\s]", "", self.normalize_compare_text(answer))
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

    def _is_explicit_empty_json_array(self, output_text):
        cleaned = (output_text or "").strip()
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
        return cleaned == "[]" or (bool(re.search(r"\[\s*\]", cleaned)) and "{" not in cleaned)

    def _is_case_specific_billing_qa(self, question, answer):
        """过滤具体用户的账单/话单/费用个案结论，避免沉淀成知识库规则。"""
        text = self.normalize_compare_text(f"{question} {answer}")
        billing_signals = (
            "账单", "话单", "话费", "费用", "流量费", "短信费", "语音通话费", "本地通话费",
            "调账", "到账", "余额", "扣费", "欠费", "缴费", "发票",
        )
        case_query_signals = (
            "产生", "为什么", "原因", "未到账", "没到账", "没有到账", "怎么没有",
            "查询", "核实", "看一下", "显示", "分摊", "调减", "合计",
        )
        if not any(signal in text for signal in billing_signals):
            return False
        if not any(signal in text for signal in case_query_signals):
            return False

        concrete_patterns = (
            r"\d+(?:\.\d+)?\s*元",
            r"\d{6}\s*账期",
            r"\b20\d{4}\b",
            r"\d{11}",
            r"\d{1,2}\s*月份?",
            r"[一二三四五六七八九十]月份?",
        )
        concrete_count = sum(1 for pattern in concrete_patterns if re.search(pattern, text))
        return concrete_count >= 1

    def _get_explicit_identifier_drop_reason(self, question, answer):
        """明确个案标识直接丢弃，不再进入改写/通用化/模型复审。"""
        text = self.normalize_compare_text(f"{question} {answer}")
        if re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text):
            return "DROP_IDENTIFIER_PHONE"
        if re.search(r"(?<!\d)\d{12,}(?!\d)", text):
            return "DROP_IDENTIFIER_ORDER"
        if re.search(r"(?<!\d)\d{2,}\s*(?:号码|订单|工单|流水号|申请单|账号)", text):
            return "DROP_IDENTIFIER_CASE_ID"
        if re.search(r"(?:号码|订单|工单|流水号|申请单|账号)\s*(?:为|是|[:：#-])?\s*\d{2,}(?!\d)", text):
            return "DROP_IDENTIFIER_CASE_ID"
        identifier_signals = (
            "业务流水号", "算费流水号", "订单号", "订单：", "订单:", "申请单",
            "tradeId", "subscribeId", "serialNumber", "PRODUCT_ID", "START_DATE",
            "d_order", "TF_B_TRADE_PRODUCT", "订单表",
        )
        if any(signal in text for signal in identifier_signals):
            return "DROP_IDENTIFIER_INTERNAL_ID"
        return ""

    def _compact_source_support_text(self, value):
        text = self.normalize_compare_text(value)
        text = text.replace("（", "(").replace("）", ")")
        return re.sub(r"[\s，。！？；、,.!?;:：\"'“”‘’【】\[\]{}()（）\-~～_/\\]+", "", text)

    def _get_source_support_drop_reason(self, question, answer):
        """防止模型照搬示例或臆造源对话中不存在的具体业务信息。"""
        source_text = self._current_source_dialog_text or ""
        if not source_text:
            return ""

        qa_text = self.normalize_compare_text(f"{question} {answer}")
        compact_source = self._compact_source_support_text(source_text)
        compact_qa = self._compact_source_support_text(qa_text)

        unsupported_numbers = []
        for number in re.findall(r"\d+(?:\.\d+)?", qa_text):
            if len(number.replace(".", "")) < 2:
                continue
            if number not in source_text:
                unsupported_numbers.append(number)
        if unsupported_numbers:
            return "DROP_UNSUPPORTED_SOURCE_NUMBER:" + ",".join(sorted(set(unsupported_numbers))[:8])

        business_keywords = (
            "套餐", "合约", "产品", "活动", "功能包", "流量包", "赠费包", "账本",
            "资费", "预存", "折扣", "报错", "返销", "副卡", "主卡",
        )
        unsupported_phrases = []
        for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9（）()《》\-]{4,}", qa_text):
            if not any(keyword in phrase for keyword in business_keywords):
                continue
            compact_phrase = self._compact_source_support_text(phrase)
            if len(compact_phrase) < 6:
                continue
            if compact_phrase and compact_phrase not in compact_source:
                # Generic phrases such as "合约期内无法取消" may be a valid summary
                # even when not verbatim in ASR text; only hard-drop highly specific
                # phrases that carry numbers or named product-like prefixes.
                if re.search(r"\d", compact_phrase) or any(mark in compact_phrase for mark in ("沃派", "冰激凌", "FTTR", "WiFi", "CBSS")):
                    unsupported_phrases.append(phrase)
        if unsupported_phrases:
            return "DROP_UNSUPPORTED_SOURCE_PHRASE:" + "；".join(unsupported_phrases[:3])

        return ""

    def _record_dropped_qa_pair(self, item, reason, source):
        review_category = "DROP_SOURCE_UNSUPPORTED" if str(reason).startswith("DROP_UNSUPPORTED_SOURCE") else "DROP_IDENTIFIER"
        dropped_item = {
            "representative_question": item.get("question", ""),
            "representative_answer": item.get("answer", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "source_dialog_text": self._current_source_dialog_text,
            "source_dialog_preview": (
                self._current_source_dialog_text[:200] + "..."
                if len(self._current_source_dialog_text) > 200
                else self._current_source_dialog_text
            ),
            "source_match_method": source,
            "review_category": review_category,
            "review_reason": f"抽取阶段硬过滤命中：{reason}",
            "review_source": "rule_extract_filter",
        }
        key = (
            self.normalize_compare_text(dropped_item["representative_question"]),
            self.normalize_compare_text(dropped_item["representative_answer"]),
            dropped_item["review_reason"],
        )
        existing_keys = {
            (
                self.normalize_compare_text(x.get("representative_question", "")),
                self.normalize_compare_text(x.get("representative_answer", "")),
                x.get("review_reason", ""),
            )
            for x in self.dropped_qa_pairs
        }
        if key not in existing_keys:
            self.dropped_qa_pairs.append(dropped_item)
        _debug_trace(
            "extract_qa_hard_drop",
            {
                "reason": reason,
                "source": source,
                "qa": dropped_item,
            },
        )

    def _filter_local_qa_pairs(self, qa_pairs, strict=True):
        filtered_pairs = []
        seen = set()
        for qa in qa_pairs:
            item = self._normalize_local_qa_item(qa)
            if item is None:
                continue
            identifier_reason = self._get_explicit_identifier_drop_reason(item["question"], item["answer"])
            if identifier_reason:
                self._record_dropped_qa_pair(item, identifier_reason, "local_qa_filter")
                continue
            source_support_reason = self._get_source_support_drop_reason(item["question"], item["answer"])
            if source_support_reason:
                self._record_dropped_qa_pair(item, source_support_reason, "source_support_filter")
                continue
            if self._is_low_value_local_question(item["question"], strict=strict) or self._is_low_value_local_answer(item["answer"], strict=strict):
                continue
            if self._qa_has_low_information_gain(item["question"], item["answer"]):
                continue
            if self._is_case_specific_billing_qa(item["question"], item["answer"]):
                continue
            key = (self.normalize_compare_text(item["question"]), self.normalize_compare_text(item["answer"]))
            if key in seen:
                continue
            seen.add(key)
            filtered_pairs.append(item)
        return filtered_pairs

    def _merge_local_qa_pairs(self, *qa_pair_groups):
        merged = {}
        for qa_pairs in qa_pair_groups:
            for qa in qa_pairs or []:
                item = self._normalize_local_qa_item(qa)
                if item is None:
                    continue
                key = self.normalize_compare_text(item["question"])
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

    def _repair_local_json_candidate(self, text):
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

    def _extract_local_json_candidates(self, text):
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

    def _loads_local_json_candidate(self, candidate):
        try:
            return json.loads(candidate)
        except Exception:
            pass

        try:
            import ast
            return ast.literal_eval(candidate)
        except Exception:
            return None

    def _parse_local_qa_output(self, output_text):
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

    def _call_local_model(self, system_prompt, user_prompt, max_tokens=None, scene="local_qa_extract"):
        for attempt in range(1, self.config.local_max_attempts + 1):
            try:
                trace_payload = {
                    "scene": scene,
                    "attempt": attempt,
                    "model": self.config.qa_extract_model_name,
                    "max_tokens": max_tokens or self.config.local_max_tokens,
                }
                if _include_prompt_in_trace():
                    trace_payload["system_prompt"] = system_prompt
                    trace_payload["user_prompt"] = user_prompt
                _debug_trace("local_model_request", trace_payload)
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
                self.print_model_raw_output(scene, output)
                self._log_local_extract_flow_step(scene, output)
                cleaned_output = re.sub(r'(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)', '', output)
                _debug_trace(
                    "local_model_response",
                    {
                        "scene": scene,
                        "attempt": attempt,
                        "raw_output": output,
                        "cleaned_output": cleaned_output,
                    },
                )
                return cleaned_output
            except Exception as exc:
                if self.is_retryable_api_exception(exc) and attempt < self.config.local_max_attempts:
                    _debug_trace(
                        "local_model_retry",
                        {
                            "scene": scene,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    time.sleep(0.8 * attempt)
                    continue
                _debug_trace(
                    "local_model_error",
                    {
                        "scene": scene,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                self.report_api_exception(
                    "本地 QA 抽取 API 调用",
                    exc,
                    request_url=f"{self.config.qa_extract_base_url.strip()}/chat/completions",
                )
                return None

    def _log_local_extract_flow_start(self, transcript):
        self._current_flow_step_no = 0
        block = (
            "【单条对话 QA 抽取全过程】\n"
            "【原始对话】\n"
            f"{transcript.strip() if isinstance(transcript, str) else transcript}"
        )
        self.print_model_raw_output("local_qa_extract_flow", block)

    def _log_local_extract_flow_step(self, scene, raw_output):
        scene_titles = {
            "local_qa_direct_extract": "直接抽取候选 QA",
            "local_qa_rewrite": "整理改写 QA",
            "local_qa_genericize": "通用化 QA",
            "local_qa_candidate_extract": "候选抽取 QA",
        }
        if scene not in scene_titles:
            return
        self._current_flow_step_no += 1
        thinking, result = _split_model_thinking_and_result(raw_output)
        raw_output = (raw_output or "").strip()
        block = (
            "【单条对话 QA 抽取全过程】\n"
            f"【第 {self._current_flow_step_no} 步：{scene_titles.get(scene, scene)}】\n"
            "【模型原始输出】\n"
            f"{raw_output if raw_output else '[]'}\n\n"
            "【去除 think 后结果】\n"
            f"{result if result else '[]'}"
        )
        if thinking:
            block += "\n\n【说明】模型输出了 think，程序已忽略 think，仅使用“去除 think 后结果”。"
        self.print_model_raw_output("local_qa_extract_flow", block)

    def _log_local_extract_flow_result(self, title, payload):
        self.print_model_raw_output(
            "local_qa_extract_flow",
            "【单条对话 QA 抽取全过程】\n"
            f"【{title}】\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        )

    def _extract_candidates_local(self, transcript):
        prompt = self.local_candidate_user_prompt.format(transcript=transcript.strip())
        output = self._call_local_model(self.local_extract_system_prompt, prompt, scene="local_qa_candidate_extract")
        if output is None:
            _debug_trace("candidate_extract_result", {"status": "api_failed", "pairs": []})
            return []
        parsed_pairs = self._parse_local_qa_output(output)
        filtered_pairs = self._filter_local_qa_pairs(parsed_pairs, strict=True)
        _debug_trace(
            "candidate_extract_result",
            {
                "parsed_count": len(parsed_pairs),
                "filtered_count": len(filtered_pairs),
                "parsed_pairs": parsed_pairs,
                "filtered_pairs": filtered_pairs,
            },
        )
        return filtered_pairs

    def _extract_high_value_pairs_local(self, transcript):
        prompt = self.local_direct_user_prompt.format(transcript=transcript.strip())
        output = self._call_local_model(self.local_direct_system_prompt, prompt, scene="local_qa_direct_extract")
        if output is None:
            _debug_trace("direct_extract_result", {"status": "api_failed", "pairs": []})
            return []
        parsed_pairs = self._parse_local_qa_output(output)
        filtered_pairs = self._filter_local_qa_pairs(parsed_pairs, strict=True)
        self._log_local_extract_flow_result(
            "第 1 步规则处理后结果",
            {
                "parsed_count": len(parsed_pairs),
                "filtered_count": len(filtered_pairs),
                "parsed_pairs": parsed_pairs,
                "filtered_pairs": filtered_pairs,
            },
        )
        _safe_print(
            f"[local_qa_direct_extract] parsed_count={len(parsed_pairs)} filtered_count={len(filtered_pairs)}",
            flush=True,
        )
        _debug_trace(
            "direct_extract_result",
            {
                "parsed_count": len(parsed_pairs),
                "filtered_count": len(filtered_pairs),
                "parsed_pairs": parsed_pairs,
                "filtered_pairs": filtered_pairs,
            },
        )
        return filtered_pairs

    def _rewrite_pairs_local(self, qa_pairs):
        if not qa_pairs:
            _debug_trace("rewrite_result", {"status": "empty_input", "input_pairs": [], "output_pairs": []})
            return []
        _debug_trace("rewrite_input", {"input_count": len(qa_pairs), "input_pairs": qa_pairs})
        candidate_qa_text = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
        prompt = self.local_rewrite_user_prompt.format(candidate_qa_text=candidate_qa_text)
        output = self._call_local_model(self.local_rewrite_system_prompt, prompt, scene="local_qa_rewrite")
        if output is None:
            _debug_trace("rewrite_result", {"status": "api_failed_keep_input", "output_pairs": qa_pairs})
            self._log_local_extract_flow_result("第 2 步整理改写后结果", {"status": "api_failed_keep_input", "output_pairs": qa_pairs})
            return qa_pairs
        rewritten_pairs = []
        seen = set()
        parsed_output = self._parse_local_qa_output(output)
        for qa in parsed_output:
            item = self._normalize_local_qa_item(qa)
            if item is None:
                continue
            key = (self.normalize_compare_text(item["question"]), self.normalize_compare_text(item["answer"]))
            if key in seen:
                continue
            seen.add(key)
            rewritten_pairs.append(item)
        if rewritten_pairs:
            self._log_local_extract_flow_result(
                "第 2 步整理改写后结果",
                {
                    "status": "rewritten",
                    "parsed_count": len(parsed_output),
                    "output_count": len(rewritten_pairs),
                    "parsed_pairs": parsed_output,
                    "output_pairs": rewritten_pairs,
                },
            )
            _debug_trace(
                "rewrite_result",
                {
                    "status": "rewritten",
                    "parsed_count": len(parsed_output),
                    "output_count": len(rewritten_pairs),
                    "parsed_pairs": parsed_output,
                    "output_pairs": rewritten_pairs,
                },
            )
            return rewritten_pairs
        if self._is_explicit_empty_json_array(output):
            self._log_local_extract_flow_result(
                "第 2 步整理改写后结果",
                {
                    "status": "explicit_empty",
                    "parsed_count": len(parsed_output),
                    "parsed_pairs": parsed_output,
                    "output_pairs": [],
                },
            )
            _debug_trace(
                "rewrite_result",
                {
                    "status": "explicit_empty",
                    "parsed_count": len(parsed_output),
                    "parsed_pairs": parsed_output,
                    "output_pairs": [],
                },
            )
            return []
        _debug_trace(
            "rewrite_result",
            {
                "status": "unparseable_keep_input",
                "parsed_count": len(parsed_output),
                "parsed_pairs": parsed_output,
                "output_pairs": qa_pairs,
            },
        )
        self._log_local_extract_flow_result(
            "第 2 步整理改写后结果",
            {
                "status": "unparseable_keep_input",
                "parsed_count": len(parsed_output),
                "parsed_pairs": parsed_output,
                "output_pairs": qa_pairs,
            },
        )
        return qa_pairs

    def _genericize_pairs_local(self, qa_pairs):
        if not qa_pairs:
            _debug_trace("genericize_result", {"status": "empty_input", "input_pairs": [], "output_pairs": []})
            return []
        _debug_trace("genericize_input", {"input_count": len(qa_pairs), "input_pairs": qa_pairs})
        genericized_pairs = []
        for qa in qa_pairs:
            qa_text = json.dumps([qa], ensure_ascii=False, indent=2)
            prompt = self.local_generic_user_prompt.format(qa_text=qa_text)
            output = self._call_local_model(self.local_generic_system_prompt, prompt, max_tokens=768, scene="local_qa_genericize")
            if output is None:
                _debug_trace(
                    "genericize_item_result",
                    {"status": "api_failed_keep_input", "input_qa": qa, "output_pairs": [qa]},
                )
                genericized_pairs.append(qa)
                continue

            parsed_pairs = []
            seen = set()
            parsed_output = self._parse_local_qa_output(output)
            for item in parsed_output:
                normalized = self._normalize_local_qa_item(item)
                if normalized is None:
                    continue
                key = (self.normalize_compare_text(normalized["question"]), self.normalize_compare_text(normalized["answer"]))
                if key in seen:
                    continue
                seen.add(key)
                parsed_pairs.append(normalized)
            if parsed_pairs:
                _debug_trace(
                    "genericize_item_result",
                    {
                        "status": "genericized",
                        "input_qa": qa,
                        "parsed_pairs": parsed_output,
                        "output_pairs": parsed_pairs,
                    },
                )
                genericized_pairs.extend(parsed_pairs)
            elif self._is_explicit_empty_json_array(output):
                _debug_trace(
                    "genericize_item_result",
                    {
                        "status": "explicit_empty_drop",
                        "input_qa": qa,
                        "parsed_pairs": parsed_output,
                        "output_pairs": [],
                    },
                )
                continue
            else:
                _debug_trace(
                    "genericize_item_result",
                    {
                        "status": "unparseable_keep_input",
                        "input_qa": qa,
                        "parsed_pairs": parsed_output,
                        "output_pairs": [qa],
                    },
                )
                genericized_pairs.append(qa)

        merged_pairs = self._merge_local_qa_pairs(genericized_pairs)
        filtered_pairs = self._filter_local_qa_pairs(merged_pairs, strict=True)
        self._log_local_extract_flow_result(
            "第 3 步通用化过滤后结果",
            {
                "merged_count": len(merged_pairs),
                "filtered_count": len(filtered_pairs),
                "merged_pairs": merged_pairs,
                "output_pairs": filtered_pairs,
            },
        )
        _debug_trace(
            "genericize_result",
            {
                "merged_count": len(merged_pairs),
                "filtered_count": len(filtered_pairs),
                "merged_pairs": merged_pairs,
                "output_pairs": filtered_pairs,
            },
        )
        return filtered_pairs

    def extract_qa_from_transcript(self, transcript):
        if not transcript or not isinstance(transcript, str) or len(transcript.strip()) < self.config.min_dialog_length:
            _debug_trace(
                "dialog_skipped",
                {
                    "reason": "empty_or_too_short",
                    "min_dialog_length": self.config.min_dialog_length,
                    "transcript": transcript if isinstance(transcript, str) else repr(transcript),
                },
            )
            return []

        self._current_source_dialog_text = transcript
        _debug_trace(
            "dialog_input",
            {
                "mode": "local" if self.config.use_local_model_branch else "online",
                "length": len(transcript.strip()),
                "transcript": transcript,
            },
        )

        if self.config.use_local_model_branch:
            self._log_local_extract_flow_start(transcript)
            strict_pairs = self._extract_high_value_pairs_local(transcript)
            filtered_pairs = self._filter_local_qa_pairs(strict_pairs, strict=True)
            rewritten_pairs = self._rewrite_pairs_local(filtered_pairs)
            final_pairs = self._genericize_pairs_local(rewritten_pairs)
            self.print_model_raw_output(
                "local_qa_extract_flow",
                "【单条对话 QA 抽取全过程】\n"
                "【最终结果】\n"
                + json.dumps(final_pairs, ensure_ascii=False, indent=2),
            )
            _debug_trace(
                "dialog_final_result",
                {
                    "strict_count": len(strict_pairs),
                    "filtered_count": len(filtered_pairs),
                    "rewritten_count": len(rewritten_pairs),
                    "final_count": len(final_pairs),
                    "strict_pairs": strict_pairs,
                    "filtered_pairs": filtered_pairs,
                    "rewritten_pairs": rewritten_pairs,
                    "final_pairs": final_pairs,
                },
            )
            return final_pairs

        formatted_prompt = self.extract_prompt.format(transcript=transcript.strip())

        try:
            if _include_prompt_in_trace():
                _debug_trace("online_extract_prompt", {"prompt": formatted_prompt})
            response = self.client.chat.completions.create(
                model=self.config.qa_extract_model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )

            output = response.choices[0].message.content.strip()
            self.print_model_raw_output("online_qa_extract", output)
            output = re.sub(r'(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)', '', output)
            parsed_pairs = self._filter_local_qa_pairs(self._parse_qa_output(output), strict=True)
            _debug_trace(
                "dialog_final_result",
                {
                    "raw_output": output,
                    "final_count": len(parsed_pairs),
                    "final_pairs": parsed_pairs,
                },
            )
            return parsed_pairs
        except Exception as exc:
            _debug_trace(
                "online_extract_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self.report_api_exception(
                "QA 抽取 API 调用",
                exc,
                request_url=f"{self.config.qa_extract_base_url.strip()}/chat/completions",
            )
            return []


class QAClusterAndSelector:
    def __init__(
        self,
        client,
        config,
        embedding_generator,
        normalize_compare_text,
        print_model_raw_output,
        report_api_exception,
        is_retryable_api_exception,
        progress_bar=None,
        status_placeholder=None,
        info_reporter=None,
    ):
        self.client = client
        self.config = config
        self.embedding_generator = embedding_generator
        self.normalize_compare_text = normalize_compare_text
        self.print_model_raw_output = print_model_raw_output
        self.report_api_exception = report_api_exception
        self.is_retryable_api_exception = is_retryable_api_exception
        self.progress_bar = progress_bar
        self.status_placeholder = status_placeholder
        self.info_reporter = info_reporter or (lambda _message: None)

        self.select_prompt = """你需要从以下**语义相似的QA对聚类**中，筛选出1-2个**代表性问答对**，并严格遵守以下规则：
1.  代表性QA对需覆盖聚类的核心信息，表述简洁通用，适用于未来客户。
2.  合并重复内容，优先选择答案完整、适用范围广的QA对。
3.  排除时效性、个性化、含隐私的内容；涉及产品时必须明确产品名。
4.  输出格式为JSON数组，每个元素包含"representative_question"和"representative_answer"字段，不要添加其他内容。

QA对聚类：
{cluster_qa_text}

代表性问答对："""

        self.local_select_system_prompt = """你是客服知识库代表问答筛选助手。
你的任务是从同一语义簇里的多个问答中，选出 1 到 2 个代表问答。

硬性规则：
1. 优先保留表述清楚、答案完整的问答。
2. 可以保留通用问答，也可以保留具体业务场景问答。
3. 删除碎片问句、无明确答案、纯等待话术、纯转接话术。
4. 若该簇没有合格问答，返回 []。
5. 如果该簇的问答都是关于某一用户的套餐信息、使用信息、账单明细、话单明细、费用明细、业务类型分布，不是通用化知识，返回 []。
6. 不保留个案查询结果类问答：如果问题本质是在查询某个具体用户、具体号码、具体月份、具体时间段的流量使用情况、话单明细、账单明细、费用明细、业务类型分布、套餐使用情况，而答案主要是本次查询结果、数据回显或核查结论，则一律删除。
7. 以下类型必须删除：“查询X月流量使用情况”“用户共计使用流量XXMB”“主要使用业务类型为XX”“产生费用XX元”“核实话单真实存在、计费正常”。这类内容不是知识，是个案数据回显。
8. 只能输出严格 JSON 数组，禁止解释和 Markdown 代码块。"""

        self.local_select_user_prompt = """请从下面同一簇问答中，挑选 1 到 2 个代表问答。

筛选标准：
- 问题必须是完整、可读的问句。
- 如果问题是具体产品、具体部门或具体业务场景，也可以保留；但不能保留具体号码、具体月份、具体账期、具体使用量、具体费用等个案查询结果。
- 答案必须完整，且不能只是“请稍等”“已转接”“待核实”。
- 如果多个问答表达同一意思，只保留更完整、更清楚的那个。
- 如果问答只是某个用户的流量、话单、账单、费用、套餐使用情况查询结果，返回 []。
- 如果该簇都不合格，返回 []。

示例1
输入簇：
Q: 需要邀请集团
A: 请稍等

Q: 通过 APP 办理副卡报错怎么办？
A: 如果通过 APP 办理副卡报错，建议咨询公客销进一步核实。

正确输出：
[{{"representative_question": "通过 APP 办理副卡报错怎么办？", "representative_answer": "如果通过 APP 办理副卡报错，建议咨询公客销进一步核实。"}}]

示例2
输入簇：
Q: 欠费前有没有提醒
A: 已转接后台核查

Q: 需要邀请集团
A: 请稍等

正确输出：
[]

示例3
输入簇：
Q: 如何查询用户流量使用情况？
A: 核实用户话单真实存在，计费正常。用户在【20250501-20250531】共计使用流量【20618.83MB】，主要使用业务类型：【腾讯应用】【今日头条】，并产生费用【152.17元】。

正确输出：
[]

现在处理以下簇：
{cluster_qa_text}

只输出 JSON 数组："""

        self.representative_review_system_prompt = """你是客服知识库入库质检员。
你的任务是结合“原始对话”和“抽取后的代表性 QA”，判断该 QA 是否可以进入知识库。
判断重点是“抽取后的代表性 QA 是否已经成功通用化、能否独立复用”。原始对话只作为核验依据，用来识别 QA 是否仍然带有个案结果、实时查询结果、具体标识或不可复用结论。
如果原始对话是个案，但代表性 QA 已经抽象成稳定业务规则、失败原因、办理条件或可执行处理方法，且 QA 本身不依赖具体客户/号码/订单/账务结果，可以保留。"""

        self.representative_review_user_prompt = """请结合原始对话和待判断 QA，判断该 QA 是否可入库。

核心标准：
1. 可入库 QA 必须能脱离当前用户、号码、订单、账期、系统当前状态独立复用。
2. 可入库 QA 必须有明确业务对象，例如套餐、合约、账本、功能点、报错文案、业务状态或产品名称。
3. 答案必须是稳定规则、办理条件、失败原因、功能点依据或可执行处理路径。
4. 原始对话出现手机号、订单号、后台专家不着急删除；关键看最终 QA 是否仍依赖该个案结果。

必须删除：
1. QA 问题或答案中出现具体手机号、身份证号、客户姓名、订单号、流水号、申请单号、账号、具体用户等个案标识。
2. QA 是某个用户/号码/订单/账期/发票/话单/余额/当前状态/系统显示的查询结果或处理结果。
3. QA 答案只是“需派单、转后台、待核实、发协作、联系某部门、建议咨询”等处理动作，没有稳定规则或明确步骤。特别注意：即使答案描述了转派后对方「可能」执行的操作，只要核心处理路径是转派/升级，就不可用。
4. QA 答案是“系统问题、同步延迟、可能存在异常、建议重新订购、需后台修复”等低确定性描述，且没有稳定规则依据。
5. QA 问题太宽泛，答案没有具体操作方法、触发条件、业务限制或规则依据。
6. QA 的结论只对原始对话中的单一客户/号码/订单成立，换一个同类客户时不能直接复用。

可以保留：
1. QA 已经把原始个案抽象成稳定业务规则、办理条件、失败原因、功能点依据或明确处理方法。
2. QA 包含产品名、套餐名、活动名、资费金额、折扣、功能点名、报错文案或业务状态码，可以保留。
3. QA 虽然来自具体客户对话，但最终问题和答案不含具体客户/号码/订单，且未来同类问题可以直接复用。

示例-保留：
Q: 沃派套餐48元优化版（北京）预存400元用一年折扣合约下月生效后为什么无法取消？
A: 该合约存在合约计划，合约期内无法取消。
原因：有明确套餐和合约规则，可复用。

示例-保留：
Q: 预存款未转分月导致发票账务打印设置失败如何处理？
A: 需通过【CBSS-发票账务打印设置】功能点将订单转分月，增加用户可打金额。
原因：没有具体用户标识，形成稳定处理路径。

示例-删除：
Q: 订单1125072050673324为什么无法撤单？
A: 订单已经进入历史表中，无需处理。
原因：包含具体订单号，是当前订单查询结果。

示例-删除：
Q: 取消WiFi赠费包时提示活动已经进行过转兑，不允许返销如何处理？
A: 需派单核查处理。
原因：答案只有派单动作，没有可复用规则或处理路径。

示例-删除：
Q: 宽带合约待处理状态注销如何办理？
A: 宽带合约待处理状态注销需联系中台办理，由中台人员执行合约计划取消操作。
原因：核心处理路径是联系中台转派，非一线可直接执行或复用的知识。

示例-删除：
Q: 13031114679月结发票可打金额不足如何处理？
A: 该用户存在非直充方式的一卡通缴费，同时存在多笔预存款未转分月订单。
原因：包含具体号码，且答案是该用户发票/缴费个案核查结果。

	只输出 JSON 对象，不要解释，不要 Markdown，不要输出 <think>：
	{{"keep": true/false, "category": "KEEP|DROP_IDENTIFIER|DROP_CASE_LOOKUP|DROP_BACKEND_ONLY|DROP_TOO_BROAD|DROP_NO_KNOWLEDGE", "reason": "一句话说明原因"}}

原始对话：
{source_dialog_text}

待判断 QA：
Q: {question}
A: {answer}

只输出 JSON 对象："""

    def _normalize_local_representative_item(self, item):
        if not isinstance(item, dict):
            return None

        question_keys = ("representative_question", "question", "query", "q", "问题")
        answer_keys = ("representative_answer", "answer", "reply", "a", "答案", "response")
        question = next((item.get(key) for key in question_keys if isinstance(item.get(key), str)), None)
        answer = next((item.get(key) for key in answer_keys if isinstance(item.get(key), str)), None)
        if question is None or answer is None:
            return None

        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return None
        return {
            "representative_question": question,
            "representative_answer": answer,
        }

    def _is_low_value_representative(self, item):
        question = self.normalize_compare_text(item["representative_question"])
        answer = self.normalize_compare_text(item["representative_answer"])
        if len(question) < 3 or len(answer) < 2:
            return True

        low_value_patterns = (
            r"^(请稍等|稍等|请您稍等)[。！!？?]*$",
            r"^.*已为您转接.*$",
            r"^.*转接.*请稍等.*$",
        )
        if any(re.search(pattern, answer) for pattern in low_value_patterns):
            return True
        if re.search(r"^(请稍等|稍等)$", question):
            return True
        if self._is_case_specific_usage_representative(question, answer):
            return True
        return False

    def _is_case_specific_usage_representative(self, question, answer):
        text = self.normalize_compare_text(f"{question} {answer}")
        usage_signals = (
            "流量使用", "使用流量", "话单", "账单", "费用明细", "业务类型", "套餐使用",
            "共计使用", "产生费用", "计费正常", "话单真实存在", "批价轨迹",
            "查询结果", "核查结果", "系统回显", "系统显示", "数据明细",
        )
        case_query_signals = (
            "查询", "核实", "查看", "看一下", "使用情况", "明细", "本次",
            "用户", "号码", "具体", "月份", "时间段",
        )
        concrete_patterns = (
            r"\d+(?:\.\d+)?\s*MB",
            r"\d+(?:\.\d+)?\s*GB",
            r"\d+(?:\.\d+)?\s*元",
            r"\b20\d{6}\b",
            r"\b20\d{2}-\d{2}-\d{2}\b",
            r"\d{1,2}\s*月份?",
            r"[一二三四五六七八九十]月份?",
            r"\d{11}",
        )
        return any(signal in text for signal in usage_signals) and any(signal in text for signal in case_query_signals) and any(
            re.search(pattern, text, flags=re.IGNORECASE) for pattern in concrete_patterns
        )

    def _has_concrete_identifier_representative(self, question, answer):
        text = self.normalize_compare_text(f"{question} {answer}")
        identifier_patterns = (
            r"(?<!\d)1[3-9]\d{9}(?!\d)",  # 手机号，兼容“130xxxx月结”这类中文相邻场景
            r"(?<!\d)\d{13,}(?!\d)",      # 订单号/流水号，避免误伤 10086、48元、10G 等业务数字
            r"(?<!\d)\d{2,}\s*(?:号码|订单|工单|流水号|申请单|账号)",  # 例如“123号码”“123订单”
            r"(?:号码|订单|工单|流水号|申请单|账号)\s*(?:为|是|[:：#-])?\s*\d{2,}(?!\d)",  # 例如“号码123”“订单:123”
        )
        identifier_signals = (
            "业务流水号", "算费流水号", "订单号", "订单：", "订单:", "申请单",
            "tradeId", "subscribeId", "serialNumber", "PRODUCT_ID", "START_DATE",
            "d_order", "TF_B_TRADE_PRODUCT", "订单表",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in identifier_patterns) or any(
            signal in text for signal in identifier_signals
        )

    def _is_action_only_representative(self, question, answer):
        answer_text = self.normalize_compare_text(answer)
        question_text = self.normalize_compare_text(question)

        # 强信号：答案核心就是转派/升级，没有可复用知识
        action_only_signals = (
            "转后台专家", "后台专家协助", "派单处理", "需派单", "提交工单",
            "关注工单", "待核实", "联系后台处理", "联系相关部门处理",
            "发协作", "中台协助", "信息化部协助", "省份核实",
            # v1 新增：常见转派变体
            "联系中台", "联系政企", "反馈后台", "反馈至后台",
            "邀请专家", "在线邀请专家", "需后台处理", "需要通过后台",
            "需后台操作", "后台进行处理", "后台加急处理",
            "需由后台", "需派单核查", "派单核查处理",
            "需提供完整报告给后台",
            # v2 新增：补齐漏网模式
            "联系营业厅", "走协作", "报后台",
            "需等待后台", "加急反馈",
        )
        if any(signal in answer_text for signal in action_only_signals):
            return True

        # 转派关联词 + 答案缺乏具体知识信号 → 拦截
        escalate_keywords = (
            "中台", "政企", "后台处理", "转后台", "派单", "发协作",
            "联系.*部门", "联系.*处理", "邀请.*专家",
            # v2 新增：补齐漏网
            "走协作", "加急反馈",
            "需.{0,4}后台",  # 匹配 "需后台处理"/"需等待后台处理"/"需由后台处理"
        )
        # 强知识信号：出现即视为有可复用知识
        strong_knowledge = (
            "规则", "条件", "功能点", "操作步骤", "系统会", "自动",
            "步骤", "路径",
        )
        # 弱知识信号：单独出现不能算有知识，需至少出现2个
        weak_knowledge = (
            "原因", "由于", "因为", "需注意", "注意", "确认", "检查",
            "流程", "可尝试",
        )
        has_escalate = any(re.search(kw, answer_text) for kw in escalate_keywords)
        has_strong = any(signal in answer_text for signal in strong_knowledge)
        weak_count = sum(1 for signal in weak_knowledge if signal in answer_text)
        has_knowledge = has_strong or weak_count >= 2
        if has_escalate and not has_knowledge:
            return True

        vague_action_patterns = (
            r"^(需|需要)?核实处理[。！!？?]*$",
            r"^联系.{0,12}处理[。！!？?]*$",
            r"^按流程处理[。！!？?]*$",
            r"^建议咨询.{0,12}处理[。！!？?]*$",
            r"^建议联系.{0,20}(处理|咨询|核实)[。！!？?]*$",
            r"^可.{0,6}(反馈|联系|咨询).{0,12}(处理|协助)[。！!？?]*$",
        )
        return any(re.search(pattern, answer_text) for pattern in vague_action_patterns)

    def get_final_representative_skip_reason(self, qa):
        """Return skip reason for final representative QA; empty string means keep."""
        item = self._normalize_local_representative_item(qa)
        if item is None:
            return "invalid_representative_format"

        question = self.normalize_compare_text(item["representative_question"])
        answer = self.normalize_compare_text(item["representative_answer"])
        if self._is_low_value_representative(item):
            return "low_value_representative"
        if self._has_concrete_identifier_representative(question, answer):
            return "concrete_identifier_or_case_id"
        if self._is_action_only_representative(question, answer):
            return "action_only_or_backend_ticket"
        return ""

    def should_skip_final_representative(self, qa):
        """最终写入代表 QA 前的硬过滤，只拦截确定不可复用的个案/动作类结果。"""
        reason = self.get_final_representative_skip_reason(qa)
        _debug_trace(
            "final_representative_filter",
            {
                "skip": bool(reason),
                "reason": reason,
                "qa": qa,
            },
        )
        return bool(reason)

    def should_skip_backup_first(self, qa):
        """backup_first 兜底前复用代表 QA 过滤，避免把个案数据回显重新塞回结果。"""
        return self.should_skip_final_representative(qa)

    def _parse_representative_review_output(self, output_text):
        text = (output_text or "").strip()
        if not text:
            return None

        had_thinking_marker = bool(
            re.search(r"</?think>|<\|/?thinking\|>", text, flags=re.IGNORECASE)
        )
        text = re.sub(
            r'(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)',
            '',
            text,
        ).strip()
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

        candidates = [text]
        obj_start, obj_end = text.find("{"), text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append(text[obj_start:obj_end + 1])
        array_start, array_end = text.find("["), text.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            candidates.append(text[array_start:array_end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                continue

            keep = parsed.get("keep", parsed.get("是否保留", parsed.get("可入库")))
            if keep is None:
                category_hint = str(parsed.get("category") or parsed.get("类别") or "").strip().upper()
                if category_hint == "KEEP":
                    keep = True
                elif category_hint.startswith("DROP_"):
                    keep = False
                else:
                    continue
            elif isinstance(keep, str):
                keep = keep.strip().lower() in {"true", "1", "yes", "keep", "保留", "可入库"}
            else:
                keep = bool(keep)

            category = str(parsed.get("category") or parsed.get("类别") or ("KEEP" if keep else "DROP_NO_KNOWLEDGE")).strip()
            reason = str(parsed.get("reason") or parsed.get("原因") or "").strip()
            if keep:
                category = "KEEP"
            elif not category.startswith("DROP_"):
                category = "DROP_NO_KNOWLEDGE"

            return {
                "keep": keep,
                "category": category,
                "reason": reason or ("可入库" if keep else "模型判定不可入库"),
            }

        # If a thinking block was truncated or left in the response, do not infer
        # the final decision from words like "删除" that may appear in analysis.
        if had_thinking_marker or re.search(r"<think>|<\|thinking\|>", text, flags=re.IGNORECASE):
            return None

        normalized_text = self.normalize_compare_text(text)
        if re.search(r"^(不可入库|不建议入库|删除|丢弃|DROP_)", normalized_text, flags=re.IGNORECASE):
            return {
                "keep": False,
                "category": "DROP_NO_KNOWLEDGE",
                "reason": normalized_text[:120] or "模型判定不可入库",
            }
        if re.search(r"(?<!不)可入库|^保留$|\bKEEP\b", normalized_text, flags=re.IGNORECASE):
            return {
                "keep": True,
                "category": "KEEP",
                "reason": normalized_text[:120] or "模型判定可入库",
            }
        return None

    def _call_representative_review_model(self, qa):
        item = self._normalize_local_representative_item(qa)
        if item is None:
            return None

        prompt = self.representative_review_user_prompt.format(
            question=item["representative_question"],
            answer=item["representative_answer"],
            source_dialog_text=str(qa.get("source_dialog_text", "") or "")[:4000],
        )
        use_local = bool(getattr(self.config, "use_local_model_branch", False))
        model_name = (
            getattr(self.config, "selector_model_name", "")
            if use_local else
            getattr(self.config, "model_name", "")
        )
        max_tokens = min(1024, int(getattr(self.config, "local_selector_max_tokens", 1024) or 1024))
        timeout = getattr(self.config, "local_timeout", None) if use_local else None
        attempts = int(getattr(self.config, "local_max_attempts", 1) or 1)
        request_url_base = (
            getattr(self.config, "selector_base_url", "")
            if use_local else
            getattr(self.config, "base_url", "")
        )

        for attempt in range(1, attempts + 1):
            try:
                _debug_trace(
                    "representative_review_request",
                    {
                        "attempt": attempt,
                        "model": model_name,
                        "max_tokens": max_tokens,
                        "system_prompt": self.representative_review_system_prompt,
                        "user_prompt": prompt,
                    },
                )
                kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": self.representative_review_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "top_p": 1,
                }
                if timeout is not None:
                    kwargs["timeout"] = timeout
                response = self.client.chat.completions.create(**kwargs)
                output = response.choices[0].message.content.strip()
                self.print_model_raw_output("representative_qa_review", output)
                cleaned_output = re.sub(
                    r'(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)',
                    '',
                    output,
                )
                _debug_trace(
                    "representative_review_response",
                    {
                        "attempt": attempt,
                        "raw_output": output,
                        "cleaned_output": cleaned_output,
                    },
                )
                return self._parse_representative_review_output(cleaned_output)
            except Exception as exc:
                if self.is_retryable_api_exception(exc) and attempt < attempts:
                    time.sleep(0.8 * attempt)
                    continue
                self.report_api_exception(
                    "代表性 QA 入库复审",
                    exc,
                    request_url=f"{str(request_url_base).strip()}/chat/completions" if request_url_base else None,
                )
                return None
        return None

    def review_representative_qa_list(self, representative_qa_list):
        """Use a final LLM review pass before writing representative QA to output."""
        kept_items = []
        dropped_items = []

        for index, qa in enumerate(representative_qa_list, 1):
            _debug_trace("representative_review_input", {"index": index, "qa": qa})
            hard_reason = self.get_final_representative_skip_reason(qa)
            if hard_reason:
                review = {
                    "keep": False,
                    "category": "DROP_IDENTIFIER" if hard_reason == "concrete_identifier_or_case_id" else "DROP_NO_KNOWLEDGE",
                    "reason": f"硬过滤命中：{hard_reason}",
                    "source": "rule",
                }
            else:
                review = self._call_representative_review_model(qa)
                if review is None:
                    review = {
                        "keep": True,
                        "category": "KEEP",
                        "reason": "复审模型调用失败或输出不可解析，保守保留",
                        "source": "review_failed_keep",
                    }
                else:
                    review["source"] = "llm_review"

            _debug_trace(
                "representative_review_result",
                {
                    "qa": qa,
                    "keep": review.get("keep"),
                    "category": review.get("category"),
                    "reason": review.get("reason"),
                    "source": review.get("source"),
                },
            )

            if review.get("keep"):
                kept_item = dict(qa)
                kept_item["review_category"] = review.get("category", "KEEP")
                kept_item["review_reason"] = review.get("reason", "复审通过")
                kept_item["review_source"] = review.get("source")
                kept_items.append(kept_item)
            else:
                dropped_item = dict(qa)
                dropped_item["review_category"] = review.get("category")
                dropped_item["review_reason"] = review.get("reason")
                dropped_item["review_source"] = review.get("source")
                dropped_items.append(dropped_item)

        _debug_trace(
            "representative_review_summary",
            {
                "input_count": len(representative_qa_list),
                "kept_count": len(kept_items),
                "dropped_count": len(dropped_items),
                "dropped_items": dropped_items,
            },
        )
        self.info_reporter(
            f"🧪 代表性 QA 入库复审完成：输入 {len(representative_qa_list)}，保留 {len(kept_items)}，删除 {len(dropped_items)}"
        )
        return kept_items, dropped_items

    def _parse_local_representative_output(self, output_text):
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
            obj_text = text[obj_start:obj_end + 1]
            candidate_texts.append(obj_text)
            candidate_texts.append(f"[{obj_text}]")

        parsed_items = []
        for candidate in candidate_texts:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue

            rep_list = parsed
            if isinstance(parsed, dict):
                for key in ("representative_qa", "items", "data", "result"):
                    if isinstance(parsed.get(key), list):
                        rep_list = parsed.get(key)
                        break

            if isinstance(rep_list, dict):
                rep_list = [rep_list]
            if not isinstance(rep_list, list):
                continue

            for item in rep_list:
                normalized_item = self._normalize_local_representative_item(item)
                if normalized_item is not None and not self._is_low_value_representative(normalized_item):
                    parsed_items.append(normalized_item)

            if parsed_items or rep_list == []:
                break

        unique_items = []
        seen = set()
        for item in parsed_items:
            key = (
                self.normalize_compare_text(item["representative_question"]),
                self.normalize_compare_text(item["representative_answer"]),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)
        return unique_items

    def _call_local_selector_model(self, prompt):
        for attempt in range(1, self.config.local_max_attempts + 1):
            try:
                trace_payload = {
                    "attempt": attempt,
                    "model": self.config.selector_model_name,
                    "max_tokens": self.config.local_selector_max_tokens,
                }
                if _include_prompt_in_trace():
                    trace_payload["system_prompt"] = self.local_select_system_prompt
                    trace_payload["user_prompt"] = prompt
                _debug_trace("local_selector_request", trace_payload)
                response = self.client.chat.completions.create(
                    model=self.config.selector_model_name,
                    messages=[
                        {"role": "system", "content": self.local_select_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.local_temperature,
                    max_tokens=self.config.local_selector_max_tokens,
                    top_p=self.config.local_top_p,
                    presence_penalty=self.config.local_presence_penalty,
                    frequency_penalty=self.config.local_frequency_penalty,
                    timeout=self.config.local_timeout,
                )
                output = response.choices[0].message.content.strip()
                self.print_model_raw_output("local_qa_select", output)
                cleaned_output = re.sub(r'(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)', '', output)
                _debug_trace(
                    "local_selector_response",
                    {
                        "attempt": attempt,
                        "raw_output": output,
                        "cleaned_output": cleaned_output,
                    },
                )
                return cleaned_output
            except Exception as exc:
                if self.is_retryable_api_exception(exc) and attempt < self.config.local_max_attempts:
                    _debug_trace(
                        "local_selector_retry",
                        {
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    time.sleep(0.8 * attempt)
                    continue
                _debug_trace(
                    "local_selector_error",
                    {
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                self.report_api_exception(
                    "本地代表性 QA 筛选",
                    exc,
                    request_url=f"{self.config.selector_base_url.strip()}/chat/completions",
                )
                return None

    def _normalize_cluster_label(self, label):
        if isinstance(label, (np.integer, int)):
            return int(label)
        if isinstance(label, str):
            text = label.strip()
            if text.lstrip("-").isdigit():
                return int(text)
        return label

    def _resolve_cluster_zero_label(self, clusters):
        if 0 in clusters:
            return 0
        if "0" in clusters:
            return "0"
        for label in clusters.keys():
            normalized = self._normalize_cluster_label(label)
            if normalized == 0:
                return label
        return None

    def _resolve_split_target_label(self, clusters):
        min_cluster_size = 6
        zero_label = self._resolve_cluster_zero_label(clusters)
        if zero_label is not None:
            zero_items = clusters.get(zero_label, [])
            if isinstance(zero_items, list) and len(zero_items) >= min_cluster_size:
                return zero_label

        candidate_label = None
        candidate_size = 0
        for label, qa_list in clusters.items():
            if not isinstance(qa_list, list):
                continue
            cluster_size = len(qa_list)
            if cluster_size < min_cluster_size:
                continue
            if cluster_size > candidate_size:
                candidate_label = label
                candidate_size = cluster_size
        return candidate_label

    def _extract_intent_type(self, question):
        text = self.normalize_compare_text(question)
        if not text:
            return "其他类"
        if re.search(r"(如何处理|怎么办|怎么处理|如何解决|怎么解决)", text):
            return "处理类"
        if re.search(r"(是否正常|能否|可以吗|能不能|是否可以|是否可)", text):
            return "判断类"
        if re.search(r"(如何查询|怎么查询|如何核实|怎么核实|查询)", text):
            return "查询类"
        if "为什么" in text:
            return "原因类"
        return "其他类"

    def _extract_primary_biz_keyword(self, question):
        text = self.normalize_compare_text(question)
        if not text:
            return "通用"
        keywords = [
            "停机保号", "未竣工", "短信费", "流量", "固网", "合约", "融合", "撤机",
            "套餐", "返销", "解约", "无权限", "次月", "未生效", "已下架", "渠道不同",
        ]
        for kw in keywords:
            if kw in text:
                return kw
        return "通用"

    def _bucket_key_for_cluster_zero(self, question):
        return (self._extract_intent_type(question), self._extract_primary_biz_keyword(question))

    def _get_main_eps_distance_percentile(self):
        configured = getattr(self.config, "main_eps_distance_percentile", None)
        if configured is None:
            configured = getattr(self.config, "eps_distance_percentile", 30)
        try:
            configured = float(configured)
        except Exception:
            configured = 30.0
        return max(5.0, min(50.0, configured))

    def _compute_dynamic_eps(self, upper_triangle):
        distance_median = float(np.median(upper_triangle))
        distance_percentile = self._get_main_eps_distance_percentile()
        distance_anchor = float(np.percentile(upper_triangle, distance_percentile))
        dynamic_eps = max(
            self.config.min_eps_value,
            min(self.config.max_eps_value, distance_anchor * self.config.dynamic_eps_coefficient),
        )
        return dynamic_eps, distance_anchor, distance_percentile, distance_median

    def _split_cluster_zero(self, clusters):
        target_label = self._resolve_split_target_label(clusters)
        if target_label is None:
            return clusters

        target_items = clusters.get(target_label, [])
        if not isinstance(target_items, list) or len(target_items) < 6:
            return clusters

        bucket_map = {}
        for qa in target_items:
            question = qa.get("question", "")
            bucket_key = self._bucket_key_for_cluster_zero(question)
            bucket_map.setdefault(bucket_key, []).append(qa)

        sub_clusters = []
        for bucket_items in bucket_map.values():
            if len(bucket_items) <= 3:
                sub_clusters.append(bucket_items)
                continue

            questions = [item.get("question", "") for item in bucket_items]
            embeddings = [self.embedding_generator.get_embedding(q) for q in questions]
            embeddings = np.array(embeddings)
            if len(embeddings) <= 1:
                sub_clusters.append(bucket_items)
                continue

            distance_matrix = cosine_distances(embeddings)
            upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            if upper_triangle.size == 0:
                sub_clusters.append(bucket_items)
                continue

            bucket_median = float(np.median(upper_triangle))
            second_coeff = max(0.45, min(0.85, float(self.config.dynamic_eps_coefficient) * 0.85))
            second_min_eps = max(0.08, float(self.config.min_eps_value))
            second_max_eps = min(0.65, max(float(self.config.max_eps_value), 0.5))
            second_eps = max(second_min_eps, min(second_max_eps, bucket_median * second_coeff))
            second_min_samples = max(2, int(self.config.dbscan_min_samples))

            dbscan = DBSCAN(
                eps=second_eps,
                min_samples=second_min_samples,
                metric=self.config.clustering_metric,
            )
            labels = dbscan.fit_predict(embeddings)

            grouped = {}
            noise_items = []
            for idx, label in enumerate(labels):
                normalized = self._normalize_cluster_label(label)
                if normalized == -1:
                    noise_items.append(bucket_items[idx])
                    continue
                grouped.setdefault(normalized, []).append(bucket_items[idx])

            for group in grouped.values():
                sub_clusters.append(group)
            if noise_items:
                sub_clusters.append(noise_items)

        if len(sub_clusters) <= 1:
            return clusters

        refined_clusters = {}
        for label, qa_list in clusters.items():
            if label == target_label:
                continue
            refined_clusters[self._normalize_cluster_label(label)] = qa_list

        numeric_labels = [
            int(label) for label in refined_clusters.keys()
            if isinstance(label, (int, np.integer)) and int(label) >= 0
        ]
        next_label = (max(numeric_labels) + 1) if numeric_labels else 1
        for sub_cluster in sub_clusters:
            refined_clusters[next_label] = sub_cluster
            next_label += 1

        normalized_target_label = self._normalize_cluster_label(target_label)
        self.info_reporter(
            f"🧩 簇{normalized_target_label}二次切分完成：由 1 个簇拆分为 {len(sub_clusters)} 个子簇（温和模式）"
        )
        return refined_clusters

    def cluster_qa_pairs(self, all_qa_pairs):
        if len(all_qa_pairs) < 2:
            _debug_trace(
                "cluster_result",
                {
                    "reason": "less_than_two_pairs",
                    "cluster_count": 1,
                    "clusters": {0: all_qa_pairs},
                },
            )
            return {0: all_qa_pairs}

        questions = [qa["question"] for qa in all_qa_pairs]
        _debug_trace(
            "cluster_input",
            {
                "qa_count": len(all_qa_pairs),
                "questions": questions,
            },
        )
        embeddings = []

        for i, question in enumerate(questions):
            progress = (i + 1) / (len(questions) * 2)
            if self.progress_bar is not None:
                self.progress_bar.progress(progress)
            if self.status_placeholder is not None:
                self.status_placeholder.info(f"生成问题嵌入: {i+1}/{len(questions)}")
            emb = self.embedding_generator.get_embedding(question)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        if len(embeddings) > 1:
            distance_matrix = cosine_distances(embeddings)
            upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            dynamic_eps, distance_anchor, distance_percentile, distance_median = self._compute_dynamic_eps(
                upper_triangle
            )
            self.info_reporter(
                f"✅ 动态计算 eps: {dynamic_eps:.3f}（距离P{int(distance_percentile)}分位数："
                f"{distance_anchor:.3f}；距离中位数：{distance_median:.3f}）"
            )
            _debug_trace(
                "cluster_eps",
                {
                    "dynamic_eps": float(dynamic_eps),
                    "distance_anchor": float(distance_anchor),
                    "distance_percentile": float(distance_percentile),
                    "distance_median": float(distance_median),
                    "dbscan_min_samples": self.config.dbscan_min_samples,
                    "metric": self.config.clustering_metric,
                },
            )
        else:
            dynamic_eps = 0.2
            _debug_trace(
                "cluster_eps",
                {
                    "dynamic_eps": float(dynamic_eps),
                    "reason": "single_embedding",
                    "dbscan_min_samples": self.config.dbscan_min_samples,
                    "metric": self.config.clustering_metric,
                },
            )

        dbscan = DBSCAN(
            eps=dynamic_eps,
            min_samples=self.config.dbscan_min_samples,
            metric=self.config.clustering_metric,
        )
        labels = dbscan.fit_predict(embeddings)
        _debug_trace(
            "cluster_dbscan_labels",
            {
                "labels": [int(label) if isinstance(label, (np.integer, int)) else label for label in labels],
                "question_labels": [
                    {
                        "question": questions[idx],
                        "label": int(label) if isinstance(label, (np.integer, int)) else label,
                    }
                    for idx, label in enumerate(labels)
                ],
            },
        )

        clusters = {}
        for idx, label in enumerate(labels):
            normalized_label = self._normalize_cluster_label(label)
            if normalized_label not in clusters:
                clusters[normalized_label] = []
            clusters[normalized_label].append(all_qa_pairs[idx])

        if self.config.enable_cluster_zero_split:
            clusters = self._split_cluster_zero(clusters)

        _debug_trace(
            "cluster_result",
            {
                "cluster_count": len(clusters),
                "cluster_sizes": {str(label): len(qa_list) for label, qa_list in clusters.items()},
                "clusters": clusters,
            },
        )
        if self.status_placeholder is not None:
            self.status_placeholder.success(f"✅ 聚类完成，共生成 {len(clusters)} 个簇（标签 -1 为噪声簇）")
        return clusters

    def select_representative_qa(self, cluster_qa_list):
        if not cluster_qa_list:
            _debug_trace("selector_skipped", {"reason": "empty_cluster"})
            return []

        qa_to_use = cluster_qa_list if self.config.max_cluster_qa_count is None else cluster_qa_list[:self.config.max_cluster_qa_count]
        cluster_qa_text = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_to_use])
        formatted_input = cluster_qa_text[:self.config.max_input_length]
        _debug_trace(
            "selector_input",
            {
                "cluster_size": len(cluster_qa_list),
                "qa_used_count": len(qa_to_use),
                "max_input_length": self.config.max_input_length,
                "input_truncated": len(cluster_qa_text) > len(formatted_input),
                "cluster_qa_list": cluster_qa_list,
                "formatted_input": formatted_input,
            },
        )

        if self.config.use_local_model_branch:
            prompt = self.local_select_user_prompt.format(cluster_qa_text=formatted_input)
            output = self._call_local_selector_model(prompt)
            if output is None:
                _debug_trace("selector_result", {"status": "api_failed", "representative_qa": []})
                return []
            representative_qa = self._parse_local_representative_output(output)
            _debug_trace(
                "selector_result",
                {
                    "status": "parsed",
                    "representative_count": len(representative_qa),
                    "representative_qa": representative_qa,
                },
            )
            return representative_qa

        formatted_prompt = self.select_prompt.format(cluster_qa_text=formatted_input)

        try:
            if _include_prompt_in_trace():
                _debug_trace("online_selector_prompt", {"prompt": formatted_prompt})
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )

            output = response.choices[0].message.content.strip()
            self.print_model_raw_output("online_qa_select", output)
            output = re.sub(r'(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)', '', output)

            try:
                result = json.loads(output)
                if isinstance(result, list):
                    representative_qa = [{
                        "representative_question": item["representative_question"].strip(),
                        "representative_answer": item["representative_answer"].strip(),
                    } for item in result if
                        isinstance(item, dict) and
                        "representative_question" in item and
                        "representative_answer" in item]
                    _debug_trace(
                        "selector_result",
                        {
                            "status": "parsed",
                            "representative_count": len(representative_qa),
                            "representative_qa": representative_qa,
                        },
                    )
                    return representative_qa
                _debug_trace("selector_result", {"status": "non_list_json", "representative_qa": []})
                return []
            except json.JSONDecodeError:
                _debug_trace(
                    "selector_result",
                    {
                        "status": "json_decode_error",
                        "raw_output": output,
                        "representative_qa": [],
                    },
                )
                return []
        except Exception as exc:
            _debug_trace(
                "online_selector_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self.report_api_exception(
                "代表性 QA 筛选",
                exc,
                request_url=f"{self.config.base_url.strip()}/chat/completions",
            )
            return []
