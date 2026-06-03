import argparse
import datetime
import json
import os
import re
import signal
import sys
import time

import pandas as pd
from openai import OpenAI

from qa_start_process_runtime import (
    BGEEmbeddingGenerator,
    QAPairExtractor,
    QAClusterAndSelector,
)


CURRENT_WORKER_JOB_DIR = None


def convert_numpy_types(obj):
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(value) for value in obj]
    return obj


def save_json_file(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(data), f, ensure_ascii=False, indent=2)


def load_json_if_exists(filepath, default_value):
    if not os.path.exists(filepath):
        return default_value
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value


def normalize_compare_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_excel_column_name(value):
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def load_dialogs_from_excel(excel_path, row_limit=None):
    df = pd.read_excel(excel_path)
    normalized_columns = {
        normalize_excel_column_name(column): column
        for column in df.columns
    }

    dialog_column_candidates = (
        "专家对话内容",
        "对话内容",
        "清洗后对话",
        "会话内容",
        "聊天内容",
        "聊天记录",
        "对话",
        "dialog",
    )

    dialog_column = None
    for candidate in dialog_column_candidates:
        normalized_candidate = normalize_excel_column_name(candidate)
        if normalized_candidate in normalized_columns:
            dialog_column = normalized_columns[normalized_candidate]
            break

    if dialog_column is None:
        raise ValueError(
            "未找到对话内容列。请确认 Excel 中包含以下列名之一："
            + "、".join(dialog_column_candidates)
        )

    dialogs = df[dialog_column].dropna().astype(str)
    dialogs = dialogs[dialogs.str.strip() != ""]
    if row_limit is not None:
        dialogs = dialogs.head(row_limit)
    return dialogs.tolist(), dialog_column


def ensure_result_output_dirs(result_dir):
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "dialog_extractions"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "cluster_details"), exist_ok=True)


def clear_directory_files(directory):
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        file_path = os.path.join(directory, name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def init_openai_client(base_url, api_key):
    return OpenAI(base_url=base_url.strip(), api_key=api_key)


def safe_console_log(message):
    text = str(message)
    if CURRENT_WORKER_JOB_DIR:
        try:
            append_worker_log(CURRENT_WORKER_JOB_DIR, text)
        except Exception:
            pass
    try:
        print(text, flush=True)
    except OSError:
        pass
    except Exception:
        pass


def print_model_raw_output(scene, output_text):
    text = (output_text or "").strip()
    if not text:
        safe_console_log(f"[{scene}] raw_output_empty")
        return
    safe_console_log(f"[{scene}] raw_output_begin")
    safe_console_log(text)
    safe_console_log(f"[{scene}] raw_output_end")


def classify_api_exception(exc):
    error_type = type(exc).__name__

    def build_chain_message(err):
        parts = []
        seen = set()
        current = err
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            text = (str(current).strip() or repr(current))
            parts.append(f"{type(current).__name__}: {text}")
            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        return " | ".join(parts)

    message = build_chain_message(exc)
    lower_message = message.lower()
    status_code = getattr(exc, "status_code", None)
    response_obj = getattr(exc, "response", None)
    if status_code is None and response_obj is not None:
        status_code = getattr(response_obj, "status_code", None)

    category = "UNKNOWN"
    timeout_error_names = {"APITimeoutError", "ConnectTimeout", "ReadTimeout", "TimeoutException"}
    dns_signals = (
        "temporary failure in name resolution",
        "name or service not known",
        "nodename nor servname",
        "getaddrinfo",
        "dns",
    )
    proxy_signals = ("proxy", "tunnel connection failed", "proxyerror")
    connection_signals = ("connection error", "connection refused", "failed to establish a new connection")

    if status_code == 401:
        category = "HTTP_401"
    elif status_code == 429:
        category = "HTTP_429"
    elif status_code is not None:
        category = f"HTTP_{status_code}"
    elif error_type in timeout_error_names or "timeout" in lower_message or "timed out" in lower_message:
        category = "TIMEOUT"
    elif any(signal in lower_message for signal in dns_signals):
        category = "DNS"
    elif any(signal in lower_message for signal in proxy_signals):
        category = "PROXY"
    elif "apiconnectionerror" in error_type.lower() or any(signal in lower_message for signal in connection_signals):
        category = "CONNECTION"

    return {
        "category": category,
        "status_code": status_code,
        "error_type": error_type,
        "message": message,
    }


def report_api_exception(scene, exc, request_url=None):
    info = classify_api_exception(exc)
    payload = {
        "scene": scene,
        "request_url": request_url or "",
        "category": info["category"],
        "status_code": info["status_code"],
        "error_type": info["error_type"],
        "message": info["message"],
    }
    safe_console_log(f"[api_error] {json.dumps(payload, ensure_ascii=False)}")


def is_retryable_api_exception(exc):
    info = classify_api_exception(exc)
    if info["category"] in {"HTTP_401", "HTTP_429"}:
        return False
    return info["category"] in {"TIMEOUT", "DNS", "PROXY", "CONNECTION"} or (
        "connection reset" in info["message"].lower()
        or "temporarily unavailable" in info["message"].lower()
    )


def save_json_safe(data, filepath):
    save_json_file(data, filepath)
    return True


def normalize_runtime_qa_list(items):
    normalized_items = []
    for item in items or []:
        if (
            isinstance(item, dict)
            and isinstance(item.get("question"), str)
            and isinstance(item.get("answer"), str)
        ):
            normalized_item = {
                "question": item["question"].strip(),
                "answer": item["answer"].strip(),
            }
            for key in ("source_dialog_index", "source_dialog_text", "source_dialog_preview"):
                if key in item:
                    normalized_item[key] = item.get(key)
            normalized_items.append(normalized_item)
    return normalized_items


def attach_source_dialog_to_qa_list(qa_list, dialog_index, transcript):
    annotated_items = []
    dialog_preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
    for qa in qa_list or []:
        item = dict(qa)
        item["source_dialog_index"] = dialog_index
        item["source_dialog_text"] = transcript
        item["source_dialog_preview"] = dialog_preview
        annotated_items.append(item)
    return annotated_items


def enrich_representative_qa_with_sources(representative_qa, cluster_qa):
    enriched_items = []
    for rep in representative_qa or []:
        item = dict(rep)
        matched_source = None
        rep_question = normalize_compare_text(item.get("representative_question", ""))
        rep_answer = normalize_compare_text(item.get("representative_answer", ""))
        for qa in cluster_qa or []:
            if (
                normalize_compare_text(qa.get("question", "")) == rep_question
                and normalize_compare_text(qa.get("answer", "")) == rep_answer
            ):
                matched_source = qa
                item["source_match_method"] = "exact"
                break
        if matched_source is None and cluster_qa:
            matched_source = cluster_qa[0]
            item["source_match_method"] = "cluster_first"
        if matched_source is not None:
            for key in ("source_dialog_index", "source_dialog_text", "source_dialog_preview"):
                if key in matched_source:
                    item[key] = matched_source.get(key)
        enriched_items.append(item)
    return enriched_items


def build_fixed_batch_ranges(total_dialogs, batch_size=30):
    ranges = []
    if total_dialogs <= 0:
        return ranges
    batch_no = 1
    start_index = 1
    while start_index <= total_dialogs:
        end_index = min(start_index + batch_size - 1, total_dialogs)
        ranges.append({
            "batch_no": batch_no,
            "start_index": start_index,
            "end_index": end_index,
        })
        batch_no += 1
        start_index = end_index + 1
    return ranges


def get_batch_dir_name(batch_no, start_index, end_index):
    return f"batch_{batch_no:04d}_{start_index:05d}_{end_index:05d}"


def get_batch_result_dir(job_dir, batch_range):
    return os.path.join(
        job_dir,
        "batches",
        get_batch_dir_name(batch_range["batch_no"], batch_range["start_index"], batch_range["end_index"]),
    )


def is_complete_result_dir(result_dir):
    return (
        os.path.exists(os.path.join(result_dir, "cluster_results.json"))
        and os.path.exists(os.path.join(result_dir, "representative_qa_pairs.json"))
        and os.path.exists(os.path.join(result_dir, "processing_stats.json"))
    )


def aggregate_batch_job_results(job_dir):
    ensure_result_output_dirs(job_dir)
    root_dialog_dir = os.path.join(job_dir, "dialog_extractions")
    root_cluster_details_dir = os.path.join(job_dir, "cluster_details")
    clear_directory_files(root_dialog_dir)
    clear_directory_files(root_cluster_details_dir)

    batches_root = os.path.join(job_dir, "batches")
    raw_qa_pairs = []
    representative_qa = []
    representative_qa_before_review = []
    dropped_representative_qa = []
    aggregated_clusters = {}
    total_dialogs_processed = 0
    dialogs_with_valid_qa = 0
    total_processing_seconds = 0.0
    total_review_dropped_representative_qa = 0

    if os.path.isdir(batches_root):
        for batch_name in sorted(os.listdir(batches_root)):
            batch_dir = os.path.join(batches_root, batch_name)
            if not os.path.isdir(batch_dir) or not is_complete_result_dir(batch_dir):
                continue

            batch_raw_qa = load_json_if_exists(os.path.join(batch_dir, "raw_qa_pairs.json"), [])
            batch_rep_qa = load_json_if_exists(os.path.join(batch_dir, "representative_qa_pairs.json"), [])
            batch_rep_before_review = load_json_if_exists(os.path.join(batch_dir, "representative_qa_pairs_before_review.json"), [])
            batch_rep_dropped = load_json_if_exists(os.path.join(batch_dir, "representative_qa_pairs_dropped.json"), [])
            batch_stats = load_json_if_exists(os.path.join(batch_dir, "processing_stats.json"), {})
            batch_cluster_results = load_json_if_exists(os.path.join(batch_dir, "cluster_results.json"), {})

            raw_qa_pairs.extend(batch_raw_qa if isinstance(batch_raw_qa, list) else [])
            representative_qa.extend(batch_rep_qa if isinstance(batch_rep_qa, list) else [])
            representative_qa_before_review.extend(batch_rep_before_review if isinstance(batch_rep_before_review, list) else [])
            dropped_representative_qa.extend(batch_rep_dropped if isinstance(batch_rep_dropped, list) else [])
            total_dialogs_processed += int(batch_stats.get("total_dialogs_processed", 0) or 0)
            dialogs_with_valid_qa += int(batch_stats.get("dialogs_with_valid_qa", 0) or 0)
            total_processing_seconds += float(batch_stats.get("processing_time_seconds", 0) or 0)
            total_review_dropped_representative_qa += int(batch_stats.get("review_dropped_representative_qa_count", 0) or 0)

            raw_clusters = batch_cluster_results.get("clusters", {}) if isinstance(batch_cluster_results, dict) else {}
            for label_str, cluster_data in raw_clusters.items():
                qa_pairs = cluster_data.get("qa_pairs", []) if isinstance(cluster_data, dict) else cluster_data
                if str(label_str).strip() == "-1":
                    aggregated_clusters.setdefault(-1, []).extend(qa_pairs)
                else:
                    aggregated_clusters[f"{batch_name}::{label_str}"] = qa_pairs

            batch_dialog_dir = os.path.join(batch_dir, "dialog_extractions")
            if os.path.isdir(batch_dialog_dir):
                for dialog_file in sorted(os.listdir(batch_dialog_dir)):
                    if not dialog_file.endswith(".json"):
                        continue
                    dialog_result = load_json_if_exists(os.path.join(batch_dialog_dir, dialog_file), None)
                    if dialog_result is not None:
                        save_json_safe(dialog_result, os.path.join(root_dialog_dir, dialog_file))

    root_cluster_results = {
        "cluster_count": len(aggregated_clusters),
        "noise_cluster_count": 1 if isinstance(aggregated_clusters.get(-1), list) and aggregated_clusters.get(-1) else 0,
        "total_qa_pairs": len(raw_qa_pairs),
        "clusters": {
            str(label): {"size": len(qa_list), "qa_pairs": qa_list}
            for label, qa_list in aggregated_clusters.items()
        },
    }

    for label, qa_list in aggregated_clusters.items():
        if isinstance(qa_list, list) and qa_list:
            save_json_safe(
                {
                    "cluster_id": label,
                    "size": len(qa_list),
                    "qa_pairs": qa_list,
                },
                os.path.join(root_cluster_details_dir, f"cluster_{str(label).replace('/', '_')}.json"),
            )

    stats = {
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_seconds": round(total_processing_seconds, 2),
        "total_dialogs_processed": total_dialogs_processed,
        "dialogs_with_valid_qa": dialogs_with_valid_qa,
        "total_raw_qa_pairs": len(raw_qa_pairs),
        "total_clusters": len(aggregated_clusters),
        "noise_cluster_size": len(aggregated_clusters.get(-1, [])) if isinstance(aggregated_clusters.get(-1, []), list) else 0,
        "final_representative_qa_count": len(representative_qa),
        "review_dropped_representative_qa_count": total_review_dropped_representative_qa or len(dropped_representative_qa),
        "success_rate": f"{dialogs_with_valid_qa/total_dialogs_processed*100:.1f}%" if total_dialogs_processed else "",
        "qa_extraction_rate": f"{len(raw_qa_pairs)/total_dialogs_processed:.2f} QA/对话" if total_dialogs_processed else "",
        "representative_qa_rate": f"{len(representative_qa)/len(raw_qa_pairs)*100:.1f}%" if raw_qa_pairs else "",
    }
    save_json_safe(raw_qa_pairs, os.path.join(job_dir, "raw_qa_pairs.json"))
    save_json_safe(root_cluster_results, os.path.join(job_dir, "cluster_results.json"))
    save_json_safe(representative_qa, os.path.join(job_dir, "representative_qa_pairs.json"))
    save_json_safe(representative_qa_before_review, os.path.join(job_dir, "representative_qa_pairs_before_review.json"))
    save_json_safe(dropped_representative_qa, os.path.join(job_dir, "representative_qa_pairs_dropped.json"))
    save_json_safe(stats, os.path.join(job_dir, "processing_stats.json"))


def update_job_manifest(job_dir, **kwargs):
    manifest_path = os.path.join(job_dir, "job_manifest.json")
    manifest = load_json_if_exists(manifest_path, {})
    manifest.update(kwargs)
    manifest["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_json_safe(manifest, manifest_path)


def append_worker_log(job_dir, message):
    log_path = os.path.join(job_dir, "batch_worker.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def run_exact_processing_pipeline(
    config,
    dialog_texts,
    dialog_indices,
    result_dir,
    dialog_column="",
    batch_manifest_path=None,
    batch_manifest_base=None,
    job_progress_callback=None,
    embedding_generator=None,
):
    ensure_result_output_dirs(result_dir)
    dialog_extractions_dir = os.path.join(result_dir, "dialog_extractions")
    cluster_details_dir = os.path.join(result_dir, "cluster_details")
    raw_qa_file = os.path.join(result_dir, "raw_qa_pairs.json")
    cluster_file = os.path.join(result_dir, "cluster_results.json")
    rep_qa_file = os.path.join(result_dir, "representative_qa_pairs.json")
    rep_qa_before_review_file = os.path.join(result_dir, "representative_qa_pairs_before_review.json")
    rep_qa_dropped_file = os.path.join(result_dir, "representative_qa_pairs_dropped.json")
    stats_file = os.path.join(result_dir, "processing_stats.json")

    processed_dialog_indices = set()
    batch_manifest_seed = dict(batch_manifest_base or {})
    if batch_manifest_path and os.path.exists(batch_manifest_path):
        existing_manifest = load_json_if_exists(batch_manifest_path, {})
        processed_dialog_indices = set(existing_manifest.get("processed_dialog_indices", []))
        batch_manifest_seed.update(existing_manifest)

    start_time = time.time()
    qa_extract_client = init_openai_client(config.qa_extract_base_url, config.qa_extract_api_key)
    selector_client = init_openai_client(config.selector_base_url, config.selector_api_key)
    safe_console_log("✅ OpenAI 客户端初始化成功")

    if config.embedding_method == "bge" and embedding_generator is None:
        safe_console_log("🚀 初始化 BGE 嵌入模型...")
        embedding_generator = BGEEmbeddingGenerator(config, warning_reporter=safe_console_log)
        safe_console_log("✅ BGE 嵌入模型加载成功")

    qa_extractor = QAPairExtractor(
        qa_extract_client,
        config,
        normalize_compare_text,
        print_model_raw_output,
        report_api_exception,
        is_retryable_api_exception,
    )
    qa_cluster_selector = QAClusterAndSelector(
        selector_client,
        config,
        embedding_generator,
        normalize_compare_text,
        print_model_raw_output,
        report_api_exception,
        is_retryable_api_exception,
        progress_bar=None,
        status_placeholder=None,
        info_reporter=safe_console_log,
    )

    def persist_batch_manifest(stage, current_dialog_index=None, completed=False):
        if not batch_manifest_path:
            return
        manifest = dict(batch_manifest_seed)
        processed_count = len(processed_dialog_indices)
        total_count = len(dialog_indices)
        manifest.update({
            "dialog_column": dialog_column,
            "processed_dialog_indices": sorted(processed_dialog_indices),
            "processed_dialog_count": processed_count,
            "current_dialog_index": current_dialog_index,
            "current_stage": stage,
            "current_batch_progress": round(processed_count / total_count, 4) if total_count else 1.0,
            "completed": completed,
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_json_safe(manifest, batch_manifest_path)
        if job_progress_callback is not None:
            job_progress_callback(stage=stage, current_dialog_index=current_dialog_index, batch_progress=manifest["current_batch_progress"], batch_processed_dialogs=processed_count, batch_total_dialogs=total_count)

    safe_console_log(f"🔍 开始提取问答对 (共 {len(dialog_texts)} 条对话)...")
    all_qa_pairs = []
    extract_dropped_qa = []
    valid_count = 0

    for idx, (dialog_index, transcript) in enumerate(zip(dialog_indices, dialog_texts), 1):
        safe_console_log(f"处理对话 #{idx}/{len(dialog_texts)}")
        dialog_file = os.path.join(dialog_extractions_dir, f"dialog_{dialog_index}.json")

        if dialog_index in processed_dialog_indices and os.path.exists(dialog_file):
            dialog_result = load_json_if_exists(dialog_file, {})
            qa_list = normalize_runtime_qa_list(dialog_result.get("extracted_qa_pairs", []))
            qa_list = attach_source_dialog_to_qa_list(qa_list, dialog_index, transcript)
            dialog_dropped = dialog_result.get("dropped_qa_pairs", [])
        else:
            before_drop_count = len(getattr(qa_extractor, "dropped_qa_pairs", []))
            qa_list = normalize_runtime_qa_list(qa_extractor.extract_qa_from_transcript(transcript))
            qa_list = attach_source_dialog_to_qa_list(qa_list, dialog_index, transcript)
            dialog_dropped = getattr(qa_extractor, "dropped_qa_pairs", [])[before_drop_count:]
            dialog_dropped = attach_source_dialog_to_qa_list(dialog_dropped, dialog_index, transcript)
            dialog_result = {
                "dialog_index": dialog_index,
                "full_dialog_text": transcript,
                "dialog_preview": transcript[:200] + "..." if len(transcript) > 200 else transcript,
                "extracted_qa_pairs": qa_list,
                "dropped_qa_pairs": dialog_dropped,
                "extraction_count": len(qa_list),
                "success": True,
            }
            save_json_safe(dialog_result, dialog_file)
            processed_dialog_indices.add(dialog_index)

        if qa_list:
            valid_count += 1
        all_qa_pairs.extend(qa_list)
        if isinstance(dialog_dropped, list):
            extract_dropped_qa.extend(dialog_dropped)
        save_json_safe(all_qa_pairs, raw_qa_file)
        persist_batch_manifest("extracting", current_dialog_index=dialog_index, completed=False)

    safe_console_log(f"✅ 共提取 {len(all_qa_pairs)} 个 QA 对 (来自 {valid_count}/{len(dialog_texts)} 个对话)")

    clusters = {}
    final_representative_qa = []
    dropped_representative_qa = list(extract_dropped_qa)
    clear_directory_files(cluster_details_dir)

    if all_qa_pairs:
        safe_console_log("🧩 开始聚类分析...")
        persist_batch_manifest("clustering", completed=False)
        clusters = qa_cluster_selector.cluster_qa_pairs(all_qa_pairs)

        cluster_results = {
            "cluster_count": len(clusters),
            "noise_cluster_count": len(clusters.get(-1, [])) if isinstance(clusters.get(-1, []), list) else 0,
            "total_qa_pairs": len(all_qa_pairs),
            "clusters": {str(label): {"size": len(qa_list), "qa_pairs": qa_list} for label, qa_list in clusters.items()},
        }
        save_json_safe(cluster_results, cluster_file)

        for label, cluster_qa in clusters.items():
            if isinstance(cluster_qa, list) and cluster_qa:
                save_json_safe(
                    {"cluster_id": label, "size": len(cluster_qa), "qa_pairs": cluster_qa},
                    os.path.join(cluster_details_dir, f"cluster_{label}.json"),
                )

        safe_console_log("🌟 筛选代表性问答对...")
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        total_valid_clusters = len([1 for label, cluster_qa in sorted_clusters if label != -1 and len(cluster_qa) > 0])

        completed_cluster_count = 0
        for i, (label, cluster_qa) in enumerate(sorted_clusters):
            if label == -1 or len(cluster_qa) == 0:
                continue
            completed_cluster_count += 1
            safe_console_log(f"处理簇 #{label} ({completed_cluster_count}/{total_valid_clusters})")
            representative_qa = qa_cluster_selector.select_representative_qa(cluster_qa)
            if representative_qa:
                enriched_items = enrich_representative_qa_with_sources(representative_qa, cluster_qa)
                final_representative_qa.extend(enriched_items)
            elif config.use_backup_strategy and cluster_qa:
                backup_item = {
                    "representative_question": cluster_qa[0]["question"],
                    "representative_answer": cluster_qa[0]["answer"],
                    "source_dialog_index": cluster_qa[0].get("source_dialog_index"),
                    "source_dialog_text": cluster_qa[0].get("source_dialog_text"),
                    "source_dialog_preview": cluster_qa[0].get("source_dialog_preview"),
                    "source_match_method": "backup_first",
                }
                final_representative_qa.append(backup_item)
            persist_batch_manifest("selecting_representative", completed=False)
        save_json_safe(final_representative_qa, rep_qa_before_review_file)
        safe_console_log("🧪 开始代表性 QA 入库复审...")
        final_representative_qa, dropped_representative_qa = qa_cluster_selector.review_representative_qa_list(
            final_representative_qa
        )
        dropped_representative_qa = list(extract_dropped_qa) + dropped_representative_qa
        save_json_safe(final_representative_qa, rep_qa_file)
        save_json_safe(dropped_representative_qa, rep_qa_dropped_file)
    else:
        save_json_safe({"cluster_count": 0, "noise_cluster_count": 0, "total_qa_pairs": 0, "clusters": {}}, cluster_file)
        save_json_safe([], rep_qa_file)
        save_json_safe([], rep_qa_before_review_file)
        save_json_safe(dropped_representative_qa, rep_qa_dropped_file)

    processing_time = time.time() - start_time
    stats = {
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_seconds": round(processing_time, 2),
        "total_dialogs_processed": len(dialog_texts),
        "dialogs_with_valid_qa": valid_count,
        "total_raw_qa_pairs": len(all_qa_pairs),
        "total_clusters": len(clusters),
        "noise_cluster_size": len(clusters.get(-1, [])) if isinstance(clusters.get(-1, []), list) else 0,
        "final_representative_qa_count": len(final_representative_qa),
        "review_dropped_representative_qa_count": len(dropped_representative_qa) if all_qa_pairs else 0,
        "success_rate": f"{valid_count/len(dialog_texts)*100:.1f}%" if dialog_texts else "",
        "qa_extraction_rate": f"{len(all_qa_pairs)/len(dialog_texts):.2f} QA/对话" if dialog_texts else "",
        "representative_qa_rate": f"{len(final_representative_qa)/len(all_qa_pairs)*100:.1f}%" if all_qa_pairs else "",
    }
    save_json_safe(stats, stats_file)
    persist_batch_manifest("completed", completed=True)
    safe_console_log(f"🎉 批次处理完成！总耗时: {processing_time:.2f} 秒")


class WorkerConfig:
    def __init__(self, payload):
        for key, value in payload.items():
            setattr(self, key, value)


def main():
    global CURRENT_WORKER_JOB_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()

    job_dir = os.path.abspath(args.job_dir)
    CURRENT_WORKER_JOB_DIR = job_dir
    os.environ.setdefault("QA_DEBUG_TRACE", "1")
    os.environ.setdefault("QA_DEBUG_TRACE_PROMPT", "0")
    os.environ.setdefault("QA_DEBUG_TRACE_FILE_AUTO", "1")
    if (
        os.getenv("QA_DEBUG_TRACE", "").strip()
        and (
            not os.getenv("QA_DEBUG_TRACE_FILE", "").strip()
            or os.getenv("QA_DEBUG_TRACE_FILE_AUTO", "").strip() == "1"
        )
    ):
        os.environ["QA_DEBUG_TRACE_FILE"] = os.path.join(job_dir, "qa_debug_trace.log")
        os.environ["QA_DEBUG_TRACE_FILE_AUTO"] = "1"
    try:
        append_worker_log(job_dir, f"worker started pid={os.getpid()}")
        config_path = os.path.join(job_dir, "job_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"未找到 job 配置文件: {config_path}")

        payload = load_json_if_exists(config_path, {})
        config = WorkerConfig(payload["runtime_config"])
        excel_path = payload["excel_path"]
        batch_size = int(payload.get("batch_size", 30))
        selected_batch_no = payload.get("selected_batch_no") if payload.get("only_selected_batch", False) else None
        selected_batch_start_no = payload.get("selected_batch_start_no")
        selected_batch_end_no = payload.get("selected_batch_end_no")
        shared_embedding_generator = None

        dialogs, dialog_column = load_dialogs_from_excel(excel_path, row_limit=None)
        all_batch_ranges = build_fixed_batch_ranges(len(dialogs), batch_size=batch_size)
        if selected_batch_start_no is not None and selected_batch_end_no is not None:
            start_no = int(selected_batch_start_no)
            end_no = int(selected_batch_end_no)
            if end_no < start_no:
                raise ValueError(f"批次范围非法：起始批次 {start_no} 大于结束批次 {end_no}")
            batch_ranges = [
                batch_range for batch_range in all_batch_ranges
                if start_no <= batch_range["batch_no"] <= end_no
            ]
            if not batch_ranges:
                raise ValueError(f"指定批次范围不存在：{start_no}-{end_no}，当前总批次数为 {len(all_batch_ranges)}")
        elif selected_batch_no is not None:
            batch_ranges = [
                batch_range for batch_range in all_batch_ranges
                if batch_range["batch_no"] == int(selected_batch_no)
            ]
            if not batch_ranges:
                raise ValueError(f"指定批次不存在：{selected_batch_no}，当前总批次数为 {len(all_batch_ranges)}")
        else:
            batch_ranges = all_batch_ranges

        def on_term(_signum, _frame):
            update_job_manifest(
                job_dir,
                status="stopped",
                pid=os.getpid(),
                current_stage="stopped_by_signal",
            )
            append_worker_log(job_dir, "worker stopped by signal")
            sys.exit(0)

        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)

        update_job_manifest(
            job_dir,
            excel_path=excel_path,
            dialog_column=dialog_column,
            total_dialogs=len(dialogs),
            batch_size=batch_size,
            total_batches=len(batch_ranges),
            all_batches_in_excel=len(all_batch_ranges),
            only_selected_batch=selected_batch_no is not None,
            selected_batch_no=int(selected_batch_no) if selected_batch_no is not None else None,
            selected_batch_start_no=int(selected_batch_start_no) if selected_batch_start_no is not None else None,
            selected_batch_end_no=int(selected_batch_end_no) if selected_batch_end_no is not None else None,
            pid=os.getpid(),
            status="running",
            current_stage="starting",
        )
        append_worker_log(
            job_dir,
            f"loaded excel rows={len(dialogs)} selected_batch={selected_batch_no} "
            f"selected_range={selected_batch_start_no}-{selected_batch_end_no}",
        )

        if config.embedding_method == "bge":
            safe_console_log("🚀 初始化共享 BGE 嵌入模型...")
            shared_embedding_generator = BGEEmbeddingGenerator(config, warning_reporter=safe_console_log)
            safe_console_log("✅ 共享 BGE 嵌入模型加载成功")

        completed_batches = 0
        for batch_range in batch_ranges:
            batch_dir = get_batch_result_dir(job_dir, batch_range)
            if is_complete_result_dir(batch_dir):
                completed_batches += 1
                append_worker_log(job_dir, f"skip completed batch {batch_range['batch_no']}")
                continue

            update_job_manifest(
                job_dir,
                status="running",
                current_batch_no=batch_range["batch_no"],
                current_batch_start_index=batch_range["start_index"],
                current_batch_end_index=batch_range["end_index"],
                completed_batches=completed_batches,
                overall_progress=round(completed_batches / len(batch_ranges), 4) if batch_ranges else 1.0,
                current_stage="preparing_batch",
            )
            append_worker_log(job_dir, f"start batch {batch_range['batch_no']} range={batch_range['start_index']}-{batch_range['end_index']}")

            batch_dialogs = dialogs[batch_range["start_index"] - 1: batch_range["end_index"]]
            batch_indices = list(range(batch_range["start_index"], batch_range["end_index"] + 1))
            batch_manifest_path = os.path.join(batch_dir, "batch_manifest.json")

            def job_progress_callback(stage, current_dialog_index=None, batch_progress=0.0, batch_processed_dialogs=0, batch_total_dialogs=0):
                update_job_manifest(
                    job_dir,
                    status="running",
                    current_batch_no=batch_range["batch_no"],
                    current_batch_start_index=batch_range["start_index"],
                    current_batch_end_index=batch_range["end_index"],
                    completed_batches=completed_batches,
                    current_stage=stage,
                    current_dialog_index=current_dialog_index,
                    current_batch_progress=batch_progress,
                    current_batch_processed_dialogs=batch_processed_dialogs,
                    current_batch_total_dialogs=batch_total_dialogs,
                    overall_progress=round((completed_batches + batch_progress) / len(batch_ranges), 4) if batch_ranges else 1.0,
                )

            run_exact_processing_pipeline(
                config=config,
                dialog_texts=batch_dialogs,
                dialog_indices=batch_indices,
                result_dir=batch_dir,
                dialog_column=dialog_column,
                batch_manifest_path=batch_manifest_path,
                batch_manifest_base={
                    "excel_path": excel_path,
                    "dialog_column": dialog_column,
                    "batch_no": batch_range["batch_no"],
                    "batch_start_index": batch_range["start_index"],
                    "batch_end_index": batch_range["end_index"],
                    "batch_size": batch_size,
                    "total_dialogs": len(dialogs),
                },
                job_progress_callback=job_progress_callback,
                embedding_generator=shared_embedding_generator,
            )

            completed_batches += 1
            aggregate_batch_job_results(job_dir)
            update_job_manifest(
                job_dir,
                status="running",
                completed_batches=completed_batches,
                current_batch_no=batch_range["batch_no"],
                current_batch_start_index=batch_range["start_index"],
                current_batch_end_index=batch_range["end_index"],
                current_stage="batch_completed",
                current_batch_progress=1.0,
                overall_progress=round(completed_batches / len(batch_ranges), 4) if batch_ranges else 1.0,
            )
            append_worker_log(job_dir, f"completed batch {batch_range['batch_no']}")

        aggregate_batch_job_results(job_dir)
        update_job_manifest(
            job_dir,
            status="completed",
            completed_batches=completed_batches,
            overall_progress=1.0 if batch_ranges else 0.0,
            current_stage="completed",
        )
        append_worker_log(job_dir, "worker completed")
    except Exception as exc:
        import traceback
        error_text = traceback.format_exc()
        append_worker_log(job_dir, error_text)
        update_job_manifest(
            job_dir,
            status="failed",
            current_stage="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
