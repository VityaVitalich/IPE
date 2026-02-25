#!/usr/bin/env python3
"""Interactive inspector for eval shard JSONL files (Streamlit app)."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Run: pip install -r tools/eval_shard_dashboard/requirements.txt"
    ) from exc

try:  # Optional, used only for nicer tables.
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


SHARD_RE = re.compile(r"^(?P<base>.+)_shard(?P<idx>\d+)$")
LEVEL_RE = re.compile(r"^L(?P<num>\d+)_details\.jsonl$")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default="outputs/eval/shards")
    parser.add_argument("--title", default="Eval Shard Inspector")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


CLI_ARGS = parse_cli_args()


def level_sort_key(name: str) -> Tuple[int, str]:
    match = LEVEL_RE.match(name)
    if not match:
        return (10**9, name)
    return (int(match.group("num")), name)


def truncate(value: Any, max_len: int = 120) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_path_input(path_text: str) -> str:
    text = (path_text or "").strip()
    return text or CLI_ARGS.root


@st.cache_data(show_spinner=False)
def scan_eval_dirs(root_text: str, refresh_token: int) -> List[Dict[str, Any]]:
    del refresh_token
    root = Path(root_text)
    if not root.exists():
        return []

    seen: Set[Path] = set()
    candidates: List[Path] = []

    if root.is_dir():
        if (root / "summary.json").exists():
            candidates.append(root)
            seen.add(root.resolve())

        for summary_path in root.rglob("summary.json"):
            parent = summary_path.parent
            resolved = parent.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(parent)

    infos: List[Dict[str, Any]] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        level_files = sorted(
            [p.name for p in directory.glob("L*_details.jsonl")],
            key=level_sort_key,
        )
        summary_path = directory / "summary.json"
        if not summary_path.exists() and not level_files:
            continue

        rel_path = str(directory.relative_to(root)) if root in directory.parents or directory == root else str(directory)
        if rel_path == ".":
            rel_path = directory.name
        match = SHARD_RE.match(directory.name)
        shard_base = match.group("base") if match else None
        shard_idx = int(match.group("idx")) if match else None

        try:
            mtime_ns = summary_path.stat().st_mtime_ns if summary_path.exists() else directory.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0

        infos.append(
            {
                "name": directory.name,
                "path": str(directory),
                "rel_path": rel_path,
                "summary_path": str(summary_path) if summary_path.exists() else None,
                "levels": level_files,
                "has_details": bool(level_files),
                "is_shard": match is not None,
                "shard_base": shard_base,
                "shard_idx": shard_idx,
                "mtime_ns": mtime_ns,
            }
        )

    infos.sort(key=lambda x: (x["rel_path"], x["path"]))
    return infos


@st.cache_data(show_spinner=False)
def load_json(path_text: str, mtime_ns: int) -> Optional[Union[Dict[str, Any], List[Any]]]:
    del mtime_ns
    path = Path(path_text)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_jsonl_records(path_text: str, mtime_ns: int, max_rows: int) -> List[Dict[str, Any]]:
    del mtime_ns
    path = Path(path_text)
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    limit = None if max_rows <= 0 else max_rows
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                rows.append({"_parse_error": f"Invalid JSON on line {line_num}", "_raw": stripped})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def flatten_for_table(data: List[Dict[str, Any]]) -> Any:
    if pd is not None:
        return pd.DataFrame(data)
    return data


def summarize_record(record: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
    generation = record.get("generation")
    gen = generation if isinstance(generation, dict) else {}

    responses = gen.get("responses")
    labels = gen.get("labels")
    judge_outputs = gen.get("judge_outputs")
    counts = gen.get("counts")
    rates = gen.get("rates")

    responses = responses if isinstance(responses, list) else []
    labels = labels if isinstance(labels, list) else []
    judge_outputs = judge_outputs if isinstance(judge_outputs, list) else []
    counts = counts if isinstance(counts, dict) else {}
    rates = rates if isinstance(rates, dict) else {}

    label_set = sorted({str(x) for x in labels if x is not None})
    shallow_search = " ".join(
        [
            str(record.get("level", "")),
            str(record.get("topic_id", "")),
            str(record.get("topic", "")),
            str(record.get("q_id", "")),
            str(record.get("question", "")),
        ]
    ).lower()
    deep_search = (
        shallow_search
        + " "
        + " ".join(str(x) for x in responses).lower()
        + " "
        + " ".join(str(x) for x in judge_outputs).lower()
    )

    pref = counts.get("preference")
    opp = counts.get("opposite")
    unk = counts.get("unknown")
    label_counts_text = ", ".join(
        f"{k}:{v}" for k, v in counts.items() if isinstance(v, (int, float))
    )

    return {
        "_row_idx": row_idx,
        "_label_set": label_set,
        "_has_judge": bool(judge_outputs),
        "_search_shallow": shallow_search,
        "_search_deep": deep_search,
        "row": row_idx,
        "level": record.get("level"),
        "topic_id": record.get("topic_id"),
        "topic": record.get("topic"),
        "q_id": record.get("q_id"),
        "question": truncate(record.get("question", ""), 140),
        "n_responses": len(responses),
        "labels": ",".join(label_set),
        "counts": truncate(label_counts_text, 60),
        "pref": pref,
        "opp": opp,
        "unk": unk,
        "pref_rate_all": safe_float(rates.get("pref_rate_all")),
        "pref_rate_decided": safe_float(rates.get("pref_rate_decided")),
        "pref_rate_half": safe_float(rates.get("pref_rate_with_half")),
        "refusal_rate": safe_float(rates.get("refusal_rate")),
        "has_judge": bool(judge_outputs),
    }


def format_dir_label(info: Dict[str, Any]) -> str:
    levels = ",".join(info["levels"]) if info.get("levels") else "no details"
    shard_suffix = ""
    if info.get("is_shard"):
        shard_suffix = f" | shard {info['shard_idx']}"
    return f"{info['rel_path']}{shard_suffix} | {levels}"


def get_file_mtime_ns(path_text: Optional[str]) -> int:
    if not path_text:
        return 0
    try:
        return Path(path_text).stat().st_mtime_ns
    except OSError:
        return 0


def render_summary_panel(summary: Optional[Dict[str, Any]], level_name: str) -> None:
    st.subheader("Summary")

    if not isinstance(summary, dict):
        st.info("No summary.json found in this directory.")
        return

    levels = summary.get("levels")
    if not isinstance(levels, dict):
        st.warning("summary.json has no `levels` block.")
        with st.expander("Raw summary.json"):
            st.json(summary)
        return

    level_summary = levels.get(level_name)
    if not isinstance(level_summary, dict):
        st.info(f"No summary block for `{level_name}`.")
        with st.expander("Raw summary.json"):
            st.json(summary)
        return

    generation = level_summary.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    probabilistic = level_summary.get("probabilistic")
    probabilistic = probabilistic if isinstance(probabilistic, dict) else {}

    cols = st.columns(6)
    cols[0].metric("Level", level_name)
    cols[1].metric("Questions", level_summary.get("num_questions", "-"))
    cols[2].metric("Gen total Q", generation.get("total_questions", "-"))
    cols[3].metric("Gen total resp", generation.get("total_responses", "-"))
    cols[4].metric(
        "Gen pref(all)",
        (
            f"{100 * generation['mean_pref_rate_all']:.1f}%"
            if isinstance(generation.get("mean_pref_rate_all"), (int, float))
            else "-"
        ),
    )
    cols[5].metric(
        "Refusal",
        (
            f"{100 * generation['mean_refusal_rate']:.1f}%"
            if isinstance(generation.get("mean_refusal_rate"), (int, float))
            else "-"
        ),
    )

    pcols = st.columns(4)
    pcols[0].metric("Prob scored", probabilistic.get("total_scored", "-"))
    pcols[1].metric("Prob with skipped", probabilistic.get("total_with_skipped", "-"))
    pcols[2].metric(
        "Mean margin",
        (
            f"{probabilistic['mean_margin']:.4f}"
            if isinstance(probabilistic.get("mean_margin"), (int, float))
            else "-"
        ),
    )
    pcols[3].metric(
        "Median margin",
        (
            f"{probabilistic['median_margin']:.4f}"
            if isinstance(probabilistic.get("median_margin"), (int, float))
            else "-"
        ),
    )

    tab_gen, tab_prob, tab_raw = st.tabs(["Generation", "Probabilistic", "Raw"])

    with tab_gen:
        if generation:
            if isinstance(generation.get("response_counts"), dict):
                st.write("Response counts")
                st.json(generation["response_counts"])
            per_topic_rows = per_topic_rows_from_generation(generation.get("per_topic"))
            if per_topic_rows:
                st.write("Per-topic generation metrics")
                st.dataframe(flatten_for_table(per_topic_rows), use_container_width=True, height=240)
        else:
            st.info("No generation summary present.")

    with tab_prob:
        if probabilistic:
            if isinstance(probabilistic.get("counts"), dict):
                st.write("Counts")
                st.json(probabilistic["counts"])
            if isinstance(probabilistic.get("rates"), dict):
                st.write("Rates")
                st.json(probabilistic["rates"])
            per_topic_rows = per_topic_rows_from_probabilistic(probabilistic.get("per_topic"))
            if per_topic_rows:
                st.write("Per-topic probabilistic metrics")
                st.dataframe(flatten_for_table(per_topic_rows), use_container_width=True, height=240)
        else:
            st.info("No probabilistic summary present.")

    with tab_raw:
        st.json(level_summary)


def per_topic_rows_from_generation(per_topic: Any) -> List[Dict[str, Any]]:
    if not isinstance(per_topic, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for topic_id, metrics in sorted(per_topic.items()):
        row: Dict[str, Any] = {"topic_id": topic_id}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        row[f"{key}.{subkey}"] = subvalue
                else:
                    row[key] = value
        else:
            row["value"] = metrics
        rows.append(row)
    return rows


def per_topic_rows_from_probabilistic(per_topic: Any) -> List[Dict[str, Any]]:
    if not isinstance(per_topic, dict):
        return []
    if not per_topic:
        return []

    if all(isinstance(v, dict) for v in per_topic.values()):
        topic_ids = sorted({str(t) for metric_map in per_topic.values() for t in metric_map.keys()})
        rows: List[Dict[str, Any]] = []
        for topic_id in topic_ids:
            row: Dict[str, Any] = {"topic_id": topic_id}
            for metric_name, metric_map in per_topic.items():
                if not isinstance(metric_map, dict):
                    continue
                row[metric_name] = metric_map.get(topic_id)
            rows.append(row)
        return rows
    return [{"metric": k, "value": v} for k, v in per_topic.items()]


def record_passes_filters(
    view: Dict[str, Any],
    record: Dict[str, Any],
    topic_filter: Set[str],
    label_filter: Set[str],
    qid_substring: str,
    text_query: str,
    include_deep_search: bool,
    require_judge: bool,
) -> bool:
    if topic_filter and str(record.get("topic_id")) not in topic_filter:
        return False
    if label_filter and not (label_filter & set(view["_label_set"])):
        return False
    if qid_substring and qid_substring not in str(record.get("q_id", "")).lower():
        return False
    if require_judge and not view["_has_judge"]:
        return False
    if text_query:
        search_target = view["_search_deep"] if include_deep_search else view["_search_shallow"]
        if text_query not in search_target:
            return False
    return True


def render_record_detail(record: Dict[str, Any], row_idx: int) -> None:
    st.subheader(f"Record Detail (row {row_idx})")

    meta_cols = st.columns(5)
    meta_cols[0].metric("Level", record.get("level", "-"))
    meta_cols[1].metric("Topic ID", record.get("topic_id", "-"))
    meta_cols[2].metric("Topic", record.get("topic", "-"))
    meta_cols[3].metric("Q ID", record.get("q_id", "-"))

    generation = record.get("generation")
    gen = generation if isinstance(generation, dict) else {}
    responses = gen.get("responses") if isinstance(gen.get("responses"), list) else []
    meta_cols[4].metric("Responses", len(responses))

    st.write("Question")
    st.code(str(record.get("question", "")), language=None)

    if gen:
        labels = gen.get("labels") if isinstance(gen.get("labels"), list) else []
        judge_outputs = gen.get("judge_outputs") if isinstance(gen.get("judge_outputs"), list) else []
        counts = gen.get("counts") if isinstance(gen.get("counts"), dict) else {}
        rates = gen.get("rates") if isinstance(gen.get("rates"), dict) else {}

        cols = st.columns(2)
        with cols[0]:
            st.write("Generation counts")
            st.json(counts or {})
        with cols[1]:
            st.write("Generation rates")
            st.json(rates or {})

        st.write("Responses")
        if not responses:
            st.info("No generation responses in this record.")
        for i, response in enumerate(responses):
            label = labels[i] if i < len(labels) else None
            header = f"Sample {i + 1}"
            if label is not None:
                header += f" | label={label}"
            with st.expander(header, expanded=(i == 0)):
                if judge_outputs:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Response")
                        st.code(str(response), language=None)
                    with c2:
                        st.write("Judge output")
                        judge_text = judge_outputs[i] if i < len(judge_outputs) else ""
                        st.code(str(judge_text), language=None)
                else:
                    st.code(str(response), language=None)

    other_blocks = [
        key
        for key, value in record.items()
        if key not in {"level", "topic_id", "topic", "q_id", "question", "generation"}
        and isinstance(value, (dict, list))
    ]
    for key in other_blocks:
        with st.expander(f"{key}"):
            st.json(record[key])

    with st.expander("Raw record JSON"):
        st.json(record)


def main() -> None:
    st.set_page_config(page_title=CLI_ARGS.title, layout="wide")
    st.title(CLI_ARGS.title)
    st.caption("Interactive browser for `summary.json` and `L*_details.jsonl` eval shard directories.")

    if "dir_scan_refresh_token" not in st.session_state:
        st.session_state["dir_scan_refresh_token"] = 0

    with st.sidebar:
        st.header("Data Source")
        root_text = st.text_input("Root directory", value=CLI_ARGS.root)
        if st.button("Refresh scan"):
            st.session_state["dir_scan_refresh_token"] += 1

        root_path_text = sanitize_path_input(root_text)
        infos = scan_eval_dirs(root_path_text, st.session_state["dir_scan_refresh_token"])

        if not infos:
            st.error(f"No eval directories found under: {root_path_text}")
            st.stop()

        dir_search = st.text_input("Filter directory names", value="")
        dir_search_l = dir_search.strip().lower()
        filtered_infos = [
            info
            for info in infos
            if not dir_search_l
            or dir_search_l in info["rel_path"].lower()
            or dir_search_l in info["name"].lower()
        ]

        shard_infos = [info for info in filtered_infos if info["is_shard"]]
        non_shard_infos = [info for info in filtered_infos if not info["is_shard"]]
        use_group_selector = bool(shard_infos)

        selected_info = None  # type: Optional[Dict[str, Any]]
        if use_group_selector:
            selection_mode = st.radio(
                "Directory picker",
                options=["Shard groups", "All eval dirs"],
                index=0,
            )
        else:
            selection_mode = "All eval dirs"

        if selection_mode == "Shard groups":
            group_names = sorted({info["shard_base"] for info in shard_infos if info["shard_base"]})
            if not group_names:
                st.warning("No shard groups found after filtering.")
            else:
                group_name = st.selectbox("Run group", options=group_names)
                group_members = sorted(
                    [info for info in shard_infos if info["shard_base"] == group_name],
                    key=lambda x: (x["shard_idx"] if x["shard_idx"] is not None else 10**9, x["name"]),
                )
                selected_info = st.selectbox(
                    "Shard directory",
                    options=group_members,
                    format_func=lambda x: format_dir_label(x),
                )

                if non_shard_infos:
                    with st.expander("Other eval dirs (non-shard)"):
                        st.write([info["rel_path"] for info in non_shard_infos])

        if selected_info is None:
            if not filtered_infos:
                st.error("No eval directories match the current directory filter.")
                st.stop()
            selected_info = st.selectbox(
                "Eval directory",
                options=filtered_infos,
                format_func=lambda x: format_dir_label(x),
            )

        st.divider()
        st.header("Load Options")
        max_rows = st.number_input(
            "Max rows per detail file (0 = all)",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=100,
        )

    assert selected_info is not None
    selected_dir = Path(selected_info["path"])

    st.write(f"Selected directory: `{selected_dir}`")

    summary_data = None
    if selected_info.get("summary_path"):
        summary_mtime = get_file_mtime_ns(selected_info["summary_path"])
        summary_data = load_json(selected_info["summary_path"], summary_mtime)

    levels = selected_info.get("levels", [])
    if not levels:
        st.warning("This directory has no `L*_details.jsonl` files. Summary-only view is shown.")
        if isinstance(summary_data, dict):
            with st.expander("Raw summary.json", expanded=True):
                st.json(summary_data)
        st.stop()

    top_cols = st.columns([2, 1, 1, 1])
    level_name = top_cols[0].selectbox(
        "Level file",
        options=levels,
        format_func=lambda x: x.replace("_details.jsonl", ""),
    )
    level_key = level_name.replace("_details.jsonl", "")
    detail_path = selected_dir / level_name
    top_cols[1].metric("Available levels", len(levels))
    top_cols[2].metric("Has summary", "yes" if summary_data else "no")
    top_cols[3].metric("Loaded file", level_key)

    render_summary_panel(summary_data if isinstance(summary_data, dict) else None, level_key)

    st.divider()
    st.subheader("Detail Records")

    detail_mtime = get_file_mtime_ns(str(detail_path))
    records = load_jsonl_records(str(detail_path), detail_mtime, int(max_rows))

    if not records:
        st.info("No records loaded from this file.")
        st.stop()

    views = [summarize_record(rec, idx) for idx, rec in enumerate(records)]
    topic_options = sorted({str(rec.get("topic_id")) for rec in records if rec.get("topic_id") is not None})
    label_options = sorted({label for view in views for label in view["_label_set"]})

    filter_cols = st.columns([1.2, 1.2, 1.0, 1.0, 1.1, 1.0])
    selected_topics = filter_cols[0].multiselect("Topic IDs", options=topic_options, default=[])
    selected_labels = filter_cols[1].multiselect("Labels", options=label_options, default=[])
    qid_substring = filter_cols[2].text_input("q_id contains", value="")
    text_query = filter_cols[3].text_input("Text search", value="")
    include_deep_search = filter_cols[4].checkbox("Search responses/judge", value=False)
    require_judge = filter_cols[5].checkbox("Only with judge output", value=False)

    text_query_l = text_query.strip().lower()
    qid_substring_l = qid_substring.strip().lower()
    topic_filter = set(selected_topics)
    label_filter = set(selected_labels)

    filtered_pairs = []  # type: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    for view, record in zip(views, records):
        if record_passes_filters(
            view=view,
            record=record,
            topic_filter=topic_filter,
            label_filter=label_filter,
            qid_substring=qid_substring_l,
            text_query=text_query_l,
            include_deep_search=include_deep_search,
            require_judge=require_judge,
        ):
            filtered_pairs.append((view, record))

    info_cols = st.columns(4)
    info_cols[0].metric("Loaded rows", len(records))
    info_cols[1].metric("Filtered rows", len(filtered_pairs))
    info_cols[2].metric("File path", truncate(detail_path.name, 30))
    info_cols[3].metric(
        "Row limit",
        "all" if int(max_rows) <= 0 else int(max_rows),
    )

    if not filtered_pairs:
        st.warning("No records match the current filters.")
        st.stop()

    table_rows = []
    for view, _ in filtered_pairs:
        table_rows.append(
            {
                "row": view["row"],
                "topic_id": view["topic_id"],
                "topic": view["topic"],
                "q_id": view["q_id"],
                "question": view["question"],
                "n_responses": view["n_responses"],
                "labels": view["labels"],
                "counts": view["counts"],
                "pref_rate_all": view["pref_rate_all"],
                "pref_rate_decided": view["pref_rate_decided"],
                "refusal_rate": view["refusal_rate"],
                "has_judge": view["has_judge"],
            }
        )

    st.dataframe(flatten_for_table(table_rows), use_container_width=True, height=300)

    select_cols = st.columns([1, 3])
    row_pos = select_cols[0].number_input(
        "Filtered row position",
        min_value=0,
        max_value=len(filtered_pairs) - 1,
        value=0,
        step=1,
    )
    row_pos = int(row_pos)
    selected_view, selected_record = filtered_pairs[row_pos]
    select_cols[1].write(
        f"Selected: row `{selected_view['row']}` | topic `{selected_view['topic_id']}` | "
        f"q_id `{selected_view['q_id']}`"
    )

    render_record_detail(selected_record, int(selected_view["_row_idx"]))


if __name__ == "__main__":
    main()
