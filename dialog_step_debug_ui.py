import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from openai import OpenAI
from qa_extractor_shared import QAPairExtractor as SharedQAPairExtractor


LOCAL_QA_DEFAULT_BASE_URL = "http://127.0.0.1:43722/v1"
LOCAL_QA_DEFAULT_API_KEY = "sk-local"
LOCAL_QA_DEFAULT_MODEL_NAME = "qa-extractor-qwen3"
ONLINE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ONLINE_DEFAULT_API_KEY = "sk-64987e1a5a82403b8f878a6c3939ad8a"
ONLINE_MODEL_OPTIONS = ["qwen3-max", "qwen3-plus", "qwen3-turbo"]
DEFAULT_EXCEL_PATH = "/home/majie/work/LtV1/data_test/2506_在线咨询-替换后.xlsx"


def normalize_compare_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def classify_api_exception(exc: Exception) -> dict[str, Any]:
    error_type = type(exc).__name__
    message = normalize_compare_text(str(exc) or repr(exc))
    status_code = getattr(exc, "status_code", None)
    response_obj = getattr(exc, "response", None)
    if status_code is None and response_obj is not None:
        status_code = getattr(response_obj, "status_code", None)

    category = "UNKNOWN"
    label = "未知错误"

    if status_code == 401:
        category = "HTTP_401"
        label = "401 认证失败"
    elif status_code == 429:
        category = "HTTP_429"
        label = "429 频率/配额限制"
    elif status_code is not None:
        category = f"HTTP_{status_code}"
        label = f"{status_code} 接口错误"
    elif "timeout" in message.lower():
        category = "TIMEOUT"
        label = "连接超时"
    elif "proxy" in message.lower():
        category = "PROXY"
        label = "代理异常"
    elif any(
        signal in message.lower()
        for signal in ("connection error", "connection refused", "apiconnectionerror")
    ):
        category = "CONNECTION"
        label = "连接失败"

    return {
        "category": category,
        "label": label,
        "status_code": status_code,
        "error_type": error_type,
        "message": message,
    }


def is_retryable_api_exception(exc: Exception) -> bool:
    info = classify_api_exception(exc)
    if info["category"] in {"HTTP_401", "HTTP_429"}:
        return False
    return info["category"] in {"TIMEOUT", "PROXY", "CONNECTION"} or "connection reset" in info["message"].lower()


def strip_model_thinking(text: str) -> str:
    return re.sub(
        r"(\<think\>[\s\S]*?\<\/think\>)|(\<\|thinking\|\>[\s\S]*?\<\/\|thinking\|\>)",
        "",
        text or "",
    ).strip()


def sk(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def build_debug_config_signature(config: "DebugConfig") -> str:
    return json.dumps(
        {
            "qa_extract_model_source": config.qa_extract_model_source,
            "qa_extract_base_url": (config.qa_extract_base_url or "").strip(),
            "qa_extract_api_key_present": bool((config.qa_extract_api_key or "").strip()),
            "qa_extract_model_name": (config.qa_extract_model_name or "").strip(),
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "local_temperature": config.local_temperature,
            "local_top_p": config.local_top_p,
            "local_max_tokens": config.local_max_tokens,
            "local_presence_penalty": config.local_presence_penalty,
            "local_frequency_penalty": config.local_frequency_penalty,
            "local_timeout": config.local_timeout,
            "local_max_attempts": config.local_max_attempts,
            "min_dialog_length": config.min_dialog_length,
            "min_question_length": config.min_question_length,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


@dataclass
class DebugConfig:
    qa_extract_model_source: str
    qa_extract_base_url: str
    qa_extract_api_key: str
    qa_extract_model_name: str
    temperature: float
    top_p: float
    max_tokens: int | None
    local_temperature: float
    local_top_p: float
    local_max_tokens: int
    local_presence_penalty: float
    local_frequency_penalty: float
    local_timeout: int
    local_max_attempts: int
    min_dialog_length: int
    min_question_length: int

    @property
    def use_local_model_branch(self) -> bool:
        return self.qa_extract_model_source == "本地模型"


class DebugQAPairExtractor(SharedQAPairExtractor):
    def run_debug(self, transcript: str) -> dict[str, Any]:
        transcript = transcript or ""
        result = {
            "success": True,
            "mode": "local" if self.config.use_local_model_branch else "online",
            "skip_reason": None,
            "raw_output": "",
            "parsed_output": [],
            "filtered_output": [],
            "final_output": [],
            "error": None,
        }
        if not transcript.strip():
            result["skip_reason"] = "空对白"
            return result
        if len(transcript.strip()) < self.config.min_dialog_length:
            result["skip_reason"] = f"对话长度低于最小阈值 {self.config.min_dialog_length}"
            return result
        try:
            if self.config.use_local_model_branch:
                prompt = self.local_direct_user_prompt.format(transcript=transcript.strip())
                raw_output = self._call_local_model(self.local_direct_system_prompt, prompt)
                parsed_output = self._parse_local_qa_output(raw_output or "")
                filtered_output = self._finalize_local_qa_pairs(transcript, parsed_output, raw_output or "")
                rewritten_output = self._rewrite_pairs_local(filtered_output)
                final_output = self._genericize_pairs_local(rewritten_output)
                result["raw_output"] = raw_output
                result["parsed_output"] = parsed_output
                result["filtered_output"] = filtered_output
                result["final_output"] = final_output
                return result

            formatted_prompt = self.extract_prompt.format(transcript=transcript.strip())
            response = self.client.chat.completions.create(
                model=self.config.qa_extract_model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            output = strip_model_thinking(response.choices[0].message.content or "")
            parsed_output = self._parse_qa_output(output)
            result["raw_output"] = output
            result["parsed_output"] = parsed_output
            result["filtered_output"] = parsed_output
            result["final_output"] = parsed_output
            return result
        except Exception as exc:
            result["success"] = False
            result["error"] = classify_api_exception(exc)
            return result


def load_dialogs_from_excel(excel_path: str, row_limit: int | None) -> list[str]:
    df = pd.read_excel(excel_path, usecols=[3])
    df.columns = ["dialog"]
    dialogs = df["dialog"].dropna().astype(str)
    if row_limit is not None:
        dialogs = dialogs.head(row_limit)
    return dialogs.tolist()


def render_result_card(result: dict[str, Any]) -> None:
    if result.get("skip_reason"):
        st.warning(f"未执行抽取：{result['skip_reason']}")
        return
    if not result.get("success", False):
        error = result.get("error") or {}
        st.error(f"调用失败：{error.get('label', '未知错误')}")
        detail = error.get("message", "")
        if error.get("status_code") is not None:
            detail = f"HTTP {error['status_code']} | {detail}"
        if detail:
            st.caption(detail)
        return
    final_output = result.get("final_output") or []
    st.metric("最终 QA 数", len(final_output))
    if not final_output:
        st.info("当前逻辑下，这条对话输出为空数组 `[]`。")
        return
    for idx, qa in enumerate(final_output, start=1):
        with st.expander(f"QA #{idx}", expanded=True):
            st.markdown(f"**问题**\n\n{qa['question']}")
            st.markdown(f"**答案**\n\n{qa['answer']}")


def render_dialog_step_debug_page(
    state_prefix: str = "dialog_step_debug",
    show_page_title: bool = True,
    embedded: bool = False,
) -> None:
    state_defaults = {
        sk(state_prefix, "dialogs"): [],
        sk(state_prefix, "current_dialog_index"): 0,
        sk(state_prefix, "loaded_excel_path"): "",
        sk(state_prefix, "loaded_row_limit"): None,
        sk(state_prefix, "debug_results"): {},
        sk(state_prefix, "config_signature"): "",
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        st.markdown("---")
        st.title("⚙️ 配置参数")
        if embedded:
            st.caption("当前位于 `qa_system.py` 内嵌测试界面。")
        else:
            st.caption("独立页面，只做逐条抽取观察，不写入现有结果目录。")

        st.markdown("---")
        st.subheader("🌐 API 配置")
        qa_extract_model_source = st.radio(
            "模型来源",
            ["线上模型", "本地模型"],
            index=0,
            key=sk(state_prefix, "qa_extract_model_source"),
            help="影响对话转 QA 的抽取步骤",
        )
        base_url = st.text_input(
            "API 地址",
            ONLINE_DEFAULT_BASE_URL,
            key=sk(state_prefix, "base_url"),
            help="API 地址",
        )
        api_key = st.text_input(
            "API Key",
            value=ONLINE_DEFAULT_API_KEY,
            type="password",
            key=sk(state_prefix, "api_key"),
            help="API 密钥",
        )
        model_name = st.selectbox(
            "模型名称",
            ONLINE_MODEL_OPTIONS,
            index=0,
            key=sk(state_prefix, "model_name"),
            help="模型名称",
        )

        st.markdown("---")
        st.subheader("📝 对话转 QA 模型")
        if qa_extract_model_source == "本地模型":
            qa_extract_base_url = st.text_input(
                "本地 QA API 地址",
                LOCAL_QA_DEFAULT_BASE_URL,
                key=sk(state_prefix, "local_base_url"),
                help="本地 vLLM OpenAI 兼容服务地址，需包含 /v1",
            )
            qa_extract_api_key = st.text_input(
                "本地 QA API Key",
                value=LOCAL_QA_DEFAULT_API_KEY,
                type="password",
                key=sk(state_prefix, "local_api_key"),
                help="本地 vLLM 服务 API Key",
            )
            qa_extract_model_name = st.text_input(
                "本地 QA 模型名",
                LOCAL_QA_DEFAULT_MODEL_NAME,
                key=sk(state_prefix, "local_model_name"),
                help="对应 vLLM 的 served-model-name",
            )
        else:
            qa_extract_base_url = base_url
            qa_extract_api_key = api_key
            qa_extract_model_name = model_name

        st.markdown("---")
        st.subheader("📊 数据源配置")
        excel_path = st.text_input(
            "Excel 文件路径",
            DEFAULT_EXCEL_PATH,
            key=sk(state_prefix, "excel_path"),
            help="包含对话记录的 Excel 文件路径",
        )
        excel_row_limit_raw = st.number_input(
            "读取行数限制",
            min_value=0,
            max_value=5000,
            value=30,
            key=sk(state_prefix, "excel_row_limit_raw"),
            help="读取前 N 行数据，0 表示全部",
        )
        st.markdown("---")
        st.subheader("🧹 数据处理参数")
        min_dialog_length = st.number_input(
            "最小对话长度",
            min_value=0,
            max_value=2000,
            value=0,
            key=sk(state_prefix, "min_dialog_length"),
            help="最小对话字符数，低于此值的对话将被忽略",
        )
        min_question_length = st.number_input(
            "最小问题长度",
            min_value=0,
            max_value=100,
            value=0,
            key=sk(state_prefix, "min_question_length"),
            help="最小问题字符数，低于此值的问题将被忽略",
        )

        st.markdown("---")
        st.subheader("🤖 模型生成参数")
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.1,
            0.1,
            key=sk(state_prefix, "temperature"),
            help="控制生成结果的随机性，值越低越确定",
        )
        top_p = st.slider(
            "Top-p",
            0.0,
            1.0,
            0.9,
            0.1,
            key=sk(state_prefix, "top_p"),
            help="采样累积概率阈值",
        )
        local_timeout = st.number_input(
            "本地超时秒数",
            min_value=5,
            max_value=300,
            value=45,
            key=sk(state_prefix, "local_timeout"),
            help="本地模型接口超时设置",
        )
        local_max_attempts = st.number_input(
            "本地重试次数",
            min_value=1,
            max_value=5,
            value=2,
            key=sk(state_prefix, "local_max_attempts"),
            help="仅对可重试错误生效",
        )

        st.markdown("---")
        st.subheader("🚀 操作")
        load_dialog_button = st.button("加载对话列表", type="primary", use_container_width=True, key=sk(state_prefix, "load_dialog_button"))
        process_current_button = st.button("处理当前条", use_container_width=True, key=sk(state_prefix, "process_current_button"))
        process_remaining_button = st.button("批量处理未处理", use_container_width=True, key=sk(state_prefix, "process_remaining_button"))
        clear_results_button = st.button("清空测试结果", use_container_width=True, key=sk(state_prefix, "clear_results_button"))

    if show_page_title:
        st.title("对话逐条观察台")
        st.markdown("复用当前抽取逻辑，逐条查看当前对话、模型原始输出和最终 QA 结果。")
        st.markdown("---")

    excel_row_limit = None if int(excel_row_limit_raw) == 0 else int(excel_row_limit_raw)

    if load_dialog_button:
        if not os.path.exists(excel_path):
            st.error(f"Excel 文件不存在: {excel_path}")
            st.stop()
        try:
            dialogs = load_dialogs_from_excel(excel_path, excel_row_limit)
        except Exception as exc:
            st.error(f"读取 Excel 失败: {exc}")
            st.stop()
        st.session_state[sk(state_prefix, "dialogs")] = dialogs
        st.session_state[sk(state_prefix, "current_dialog_index")] = 0
        st.session_state[sk(state_prefix, "loaded_excel_path")] = excel_path
        st.session_state[sk(state_prefix, "loaded_row_limit")] = excel_row_limit
        st.session_state[sk(state_prefix, "debug_results")] = {}
        st.success(f"已加载 {len(dialogs)} 条对话")
        st.rerun()

    dialogs = st.session_state[sk(state_prefix, "dialogs")]
    if not dialogs:
        st.info("先在左侧加载 Excel 对话列表。")
        return

    config = DebugConfig(
        qa_extract_model_source=qa_extract_model_source,
        qa_extract_base_url=qa_extract_base_url,
        qa_extract_api_key=qa_extract_api_key,
        qa_extract_model_name=qa_extract_model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=None,
        local_temperature=0.0,
        local_top_p=1.0,
        local_max_tokens=384,
        local_presence_penalty=0.0,
        local_frequency_penalty=0.0,
        local_timeout=int(local_timeout),
        local_max_attempts=int(local_max_attempts),
        min_dialog_length=int(min_dialog_length),
        min_question_length=int(min_question_length),
    )

    config_signature = build_debug_config_signature(config)
    signature_key = sk(state_prefix, "config_signature")
    results_key = sk(state_prefix, "debug_results")
    if st.session_state.get(signature_key) != config_signature:
        st.session_state[results_key] = {}
        st.session_state[signature_key] = config_signature
        st.info("检测到测试配置已变更，已自动清空旧测试结果。")

    if clear_results_button:
        st.session_state[results_key] = {}
        st.success("已清空当前测试结果。")
        st.rerun()

    if config.use_local_model_branch:
        if not (config.qa_extract_base_url or "").strip():
            st.error("本地模型模式下，`本地 QA API 地址` 不能为空。")
            return
        if not (config.qa_extract_api_key or "").strip():
            st.error("本地模型模式下，`本地 QA API Key` 不能为空。")
            return
        if not (config.qa_extract_model_name or "").strip():
            st.error("本地模型模式下，`本地 QA 模型名` 不能为空。")
            return
    elif not (config.qa_extract_api_key or "").strip():
        st.warning("当前是线上模型模式，处理前需要填写线上 API Key。")

    extractor = DebugQAPairExtractor(
        OpenAI(
            base_url=config.qa_extract_base_url.strip(),
            api_key=config.qa_extract_api_key,
            http_client=httpx.Client(trust_env=False),
        ),
        config,
        retryable_exception_checker=is_retryable_api_exception,
    )

    def process_dialog_at(index: int) -> None:
        transcript = dialogs[index]
        started_at = time.time()
        result = extractor.run_debug(transcript)
        result["dialog_index"] = index
        result["elapsed_seconds"] = round(time.time() - started_at, 2)
        result["dialog_preview"] = transcript[:120] + ("..." if len(transcript) > 120 else "")
        debug_results = dict(st.session_state[results_key])
        debug_results[index] = result
        st.session_state[results_key] = debug_results

    if process_current_button:
        process_dialog_at(st.session_state[sk(state_prefix, "current_dialog_index")])
        st.rerun()

    if process_remaining_button:
        progress_bar = st.progress(0.0)
        status = st.empty()
        current_results = st.session_state[results_key]
        remaining_indices = [i for i in range(len(dialogs)) if i not in current_results]
        total = len(remaining_indices)
        for done, index in enumerate(remaining_indices, start=1):
            status.info(f"处理中：第 {index + 1}/{len(dialogs)} 条")
            process_dialog_at(index)
            progress_bar.progress(done / total if total else 1.0)
        status.success(f"批量处理完成，共处理 {total} 条。")

    debug_results = st.session_state[results_key]
    processed_count = len(debug_results)
    non_empty_count = sum(
        1
        for item in debug_results.values()
        if item.get("success") and not item.get("skip_reason") and (item.get("final_output") or [])
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("总对话数", len(dialogs))
    summary_col2.metric("已处理", processed_count)
    summary_col3.metric("有 QA 输出", non_empty_count)
    st.markdown("---")

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 2])
    current_index = st.session_state[sk(state_prefix, "current_dialog_index")]

    with nav_col1:
        if st.button("上一条", use_container_width=True, disabled=current_index <= 0, key=sk(state_prefix, "prev_dialog")):
            st.session_state[sk(state_prefix, "current_dialog_index")] = current_index - 1
            st.rerun()
    with nav_col2:
        if st.button("下一条", use_container_width=True, disabled=current_index >= len(dialogs) - 1, key=sk(state_prefix, "next_dialog")):
            st.session_state[sk(state_prefix, "current_dialog_index")] = current_index + 1
            st.rerun()
    with nav_col3:
        selected_dialog_index = st.selectbox(
            "跳转到对话",
            options=list(range(len(dialogs))),
            index=current_index,
            format_func=lambda i: f"第 {i + 1} 条 | {normalize_compare_text(dialogs[i])[:36]}",
            key=sk(state_prefix, "jump_dialog"),
        )
        if selected_dialog_index != current_index:
            st.session_state[sk(state_prefix, "current_dialog_index")] = selected_dialog_index
            st.rerun()
    with nav_col4:
        current_result = debug_results.get(current_index)
        status_text = "未处理"
        if current_result:
            if current_result.get("skip_reason"):
                status_text = "已跳过"
            elif current_result.get("success") and (current_result.get("final_output") or []):
                status_text = "已产出 QA"
            elif current_result.get("success"):
                status_text = "已处理但为空"
            else:
                status_text = "处理失败"
        st.text_input("当前状态", value=status_text, disabled=True)

    current_index = st.session_state[sk(state_prefix, "current_dialog_index")]
    current_dialog = dialogs[current_index]
    current_result = debug_results.get(current_index)

    left_col, right_col = st.columns([1.15, 1])
    with left_col:
        st.subheader(f"当前对话：第 {current_index + 1} 条")
        st.text_area("对话全文", value=current_dialog, height=420, disabled=True)
    with right_col:
        st.subheader("处理结果")
        if current_result:
            info_col1, info_col2 = st.columns(2)
            info_col1.metric("模式", "本地" if current_result.get("mode") == "local" else "线上")
            info_col2.metric("耗时", f"{current_result.get('elapsed_seconds', 0)} 秒")
            render_result_card(current_result)
        else:
            st.info("这条还没处理。点左侧“处理当前条”即可。")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["模型原始输出", "结构化结果", "会话汇总"])

    with tab1:
        if current_result and current_result.get("raw_output") is not None:
            st.text_area("Raw Output", value=current_result.get("raw_output", ""), height=320, disabled=True)
        else:
            st.info("当前没有原始输出。")

    with tab2:
        if current_result:
            st.markdown("**Parsed Output**")
            st.json(current_result.get("parsed_output", []))
            st.markdown("**Filtered Output**")
            st.json(current_result.get("filtered_output", []))
            st.markdown("**Final Output**")
            st.json(current_result.get("final_output", []))
        else:
            st.info("当前还没有结构化结果。")

    with tab3:
        summary_rows = []
        for index, dialog in enumerate(dialogs):
            item = debug_results.get(index)
            if not item:
                state = "未处理"
                qa_count = 0
                elapsed = None
            elif item.get("skip_reason"):
                state = f"已跳过: {item['skip_reason']}"
                qa_count = 0
                elapsed = item.get("elapsed_seconds")
            elif item.get("success"):
                state = "成功" if (item.get("final_output") or []) else "成功但为空"
                qa_count = len(item.get("final_output") or [])
                elapsed = item.get("elapsed_seconds")
            else:
                state = "失败"
                qa_count = 0
                elapsed = item.get("elapsed_seconds")

            summary_rows.append(
                {
                    "序号": index + 1,
                    "状态": state,
                    "QA 数": qa_count,
                    "耗时(秒)": elapsed,
                    "预览": normalize_compare_text(dialog)[:80],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, height=420)
        st.download_button(
            "下载当前会话结果 JSON",
            data=json.dumps(
                {
                    "excel_path": st.session_state[sk(state_prefix, "loaded_excel_path")],
                    "row_limit": st.session_state[sk(state_prefix, "loaded_row_limit")],
                    "results": {str(index + 1): value for index, value in sorted(debug_results.items())},
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name="dialog_step_debug_results.json",
            mime="application/json",
            use_container_width=True,
            key=sk(state_prefix, "download_json"),
        )
