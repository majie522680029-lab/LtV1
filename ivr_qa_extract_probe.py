import argparse
import datetime
import json
import os
import re
import time
from types import SimpleNamespace


DEFAULT_EXCEL_PATH = "/home/majie/work/LtV1/data_raw/ivr/09月IVR咨询内容分析-替换后.xlsx"
DEFAULT_OUTPUT_ROOT = "/home/majie/work/LtV1/batch_extract_test_runs"


def normalize_compare_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def ensure_dirs(result_dir):
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "dialog_extractions"), exist_ok=True)


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_excel_column_name(value):
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def pick_dialog_column(df, requested_column=None):
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(f"指定对话列不存在: {requested_column}；当前列: {list(df.columns)}")
        return requested_column

    normalized_columns = {
        normalize_excel_column_name(column): column
        for column in df.columns
    }
    candidates = (
        "专家对话内容",
        "对话内容",
        "清洗后对话",
        "会话内容",
        "聊天内容",
        "聊天记录",
        "咨询内容",
        "ivr咨询内容",
        "ivr内容",
        "对话",
        "dialog",
    )
    for candidate in candidates:
        normalized = normalize_excel_column_name(candidate)
        if normalized in normalized_columns:
            return normalized_columns[normalized]

    text_columns = []
    for column in df.columns:
        series = df[column].dropna().astype(str)
        if series.empty:
            continue
        avg_len = series.str.len().mean()
        max_len = series.str.len().max()
        text_columns.append((avg_len, max_len, column))
    if not text_columns:
        raise ValueError("Excel 中没有可读取的文本列")

    text_columns.sort(reverse=True)
    return text_columns[0][2]


def load_dialog_rows(excel_path, row_limit=None, start_offset=0, dialog_column=None):
    import pandas as pd

    df = pd.read_excel(excel_path)
    selected_column = pick_dialog_column(df, dialog_column)
    rows = []
    for df_index, value in df[selected_column].items():
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            continue
        # Excel row number is 1-based and includes the header row.
        rows.append({
            "dialog_index": len(rows) + 1,
            "excel_row_number": int(df_index) + 2,
            "dialog_text": text,
        })

    if start_offset > 0:
        rows = rows[start_offset:]
    if row_limit is not None:
        rows = rows[:row_limit]
    return rows, selected_column, list(df.columns)


def build_config(args, result_dir):
    return SimpleNamespace(
        qa_extract_base_url=args.base_url,
        qa_extract_api_key=args.api_key,
        qa_extract_model_name=args.model,
        use_local_model_branch=True,
        temperature=0.1,
        max_tokens=None,
        top_p=0.9,
        local_temperature=args.temperature,
        local_top_p=args.top_p,
        local_max_tokens=args.max_tokens,
        local_presence_penalty=0.0,
        local_frequency_penalty=0.0,
        local_timeout=args.timeout,
        local_max_attempts=args.max_attempts,
        min_dialog_length=args.min_dialog_length,
        min_question_length=args.min_question_length,
        result_dir=result_dir,
    )


def classify_api_exception(exc):
    status_code = getattr(exc, "status_code", None)
    response_obj = getattr(exc, "response", None)
    if status_code is None and response_obj is not None:
        status_code = getattr(response_obj, "status_code", None)
    message = str(exc).strip() or repr(exc)
    return {
        "category": f"HTTP_{status_code}" if status_code is not None else type(exc).__name__,
        "status_code": status_code,
        "error_type": type(exc).__name__,
        "message": message,
    }


def is_retryable_api_exception(exc):
    info = classify_api_exception(exc)
    message = info["message"].lower()
    if info["status_code"] in {401, 429}:
        return False
    return (
        "timeout" in message
        or "connection" in message
        or "temporarily unavailable" in message
        or "connection reset" in message
    )


def report_api_exception(scene, exc, request_url=None):
    payload = classify_api_exception(exc)
    payload["scene"] = scene
    payload["request_url"] = request_url or ""
    print(f"[api_error] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def make_raw_output_logger(result_dir, print_raw=False):
    log_path = os.path.join(result_dir, "model_raw_outputs.log")

    def logger(scene, output_text):
        text = (output_text or "").strip()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{scene}] raw_output_begin\n")
            f.write(text)
            f.write(f"\n[{scene}] raw_output_end\n")
        if print_raw:
            print(f"[{scene}] {text}", flush=True)

    return logger


def normalize_qa_list(items):
    normalized = []
    for item in items or []:
        if (
            isinstance(item, dict)
            and isinstance(item.get("question"), str)
            and isinstance(item.get("answer"), str)
        ):
            question = item["question"].strip()
            answer = item["answer"].strip()
            if question and answer:
                normalized.append({"question": question, "answer": answer})
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description="独立 IVR Excel QA 抽取试跑脚本，不影响现有 Streamlit/批跑入口。")
    parser.add_argument("--excel", default=DEFAULT_EXCEL_PATH, help="待抽取的 Excel 文件路径")
    parser.add_argument("--dialog-column", default=None, help="对话内容列名；不填则自动识别")
    parser.add_argument("--row-limit", type=int, default=30, help="试跑行数；<=0 表示全部")
    parser.add_argument("--start-offset", type=int, default=0, help="跳过前 N 条非空对话后再开始")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="结果输出根目录")
    parser.add_argument("--job-name", default="ivr_qa_extract_probe", help="输出目录前缀")
    parser.add_argument("--base-url", default="http://127.0.0.1:43722/v1", help="本地 OpenAI 兼容 API 地址")
    parser.add_argument("--api-key", default="sk-local", help="API Key")
    parser.add_argument("--model", default="qa-extractor-qwen3", help="served-model-name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-dialog-length", type=int, default=0)
    parser.add_argument("--min-question-length", type=int, default=0)
    parser.add_argument("--print-raw", action="store_true", help="同时在终端打印模型原始输出")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.excel):
        raise FileNotFoundError(f"Excel 文件不存在: {args.excel}")

    try:
        from openai import OpenAI
        from qa_start_process_runtime import QAPairExtractor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"缺少运行依赖: {exc.name}。请在项目运行环境中执行本脚本，例如已安装 pandas/openai/sentence_transformers 的环境。"
        ) from exc

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output_root, f"{args.job_name}_{timestamp}")
    ensure_dirs(result_dir)

    row_limit = None if args.row_limit is not None and args.row_limit <= 0 else args.row_limit
    rows, dialog_column, excel_columns = load_dialog_rows(
        args.excel,
        row_limit=row_limit,
        start_offset=max(0, args.start_offset),
        dialog_column=args.dialog_column,
    )
    config = build_config(args, result_dir)
    client = OpenAI(base_url=config.qa_extract_base_url.strip(), api_key=config.qa_extract_api_key)
    extractor = QAPairExtractor(
        client,
        config,
        normalize_compare_text,
        make_raw_output_logger(result_dir, print_raw=args.print_raw),
        report_api_exception,
        is_retryable_api_exception,
    )

    run_config = {
        "excel": args.excel,
        "dialog_column": dialog_column,
        "excel_columns": excel_columns,
        "row_limit": row_limit,
        "start_offset": args.start_offset,
        "base_url": args.base_url,
        "model": args.model,
        "result_dir": result_dir,
        "started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(run_config, os.path.join(result_dir, "run_config.json"))

    print(f"读取列: {dialog_column}", flush=True)
    print(f"待处理对话数: {len(rows)}", flush=True)
    print(f"输出目录: {result_dir}", flush=True)

    start_time = time.time()
    all_qa_pairs = []
    success_dialog_count = 0
    for idx, row in enumerate(rows, 1):
        dialog_index = row["dialog_index"]
        excel_row_number = row["excel_row_number"]
        transcript = row["dialog_text"]
        print(f"[{idx}/{len(rows)}] 抽取 Excel 行 {excel_row_number}", flush=True)

        qa_pairs = normalize_qa_list(extractor.extract_qa_from_transcript(transcript))
        if qa_pairs:
            success_dialog_count += 1

        dialog_preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
        annotated_pairs = []
        for qa in qa_pairs:
            item = dict(qa)
            item["source_dialog_index"] = dialog_index
            item["source_excel_row_number"] = excel_row_number
            item["source_dialog_text"] = transcript
            item["source_dialog_preview"] = dialog_preview
            annotated_pairs.append(item)
        all_qa_pairs.extend(annotated_pairs)

        dialog_result = {
            "dialog_index": dialog_index,
            "excel_row_number": excel_row_number,
            "success": bool(qa_pairs),
            "extraction_count": len(qa_pairs),
            "dialog_preview": dialog_preview,
            "full_dialog_text": transcript,
            "extracted_qa_pairs": annotated_pairs,
        }
        save_json(dialog_result, os.path.join(result_dir, "dialog_extractions", f"dialog_{dialog_index}.json"))
        save_json(all_qa_pairs, os.path.join(result_dir, "raw_qa_pairs.json"))

    stats = {
        "processing_time_seconds": round(time.time() - start_time, 2),
        "total_dialogs_processed": len(rows),
        "dialogs_with_valid_qa": success_dialog_count,
        "total_raw_qa_pairs": len(all_qa_pairs),
        "qa_extraction_rate": f"{len(all_qa_pairs) / len(rows):.2f} QA/对话" if rows else "0.00 QA/对话",
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(stats, os.path.join(result_dir, "processing_stats.json"))
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"完成。结果目录: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
