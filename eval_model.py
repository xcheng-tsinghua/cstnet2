import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

_here = os.path.dirname(os.path.abspath(__file__))

_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


def _latest_timestamp_run_dir(out_root: str) -> Optional[str]:
    """Last EvalScope run folder like outputs/20260423_031347/."""
    if not os.path.isdir(out_root):
        return None
    dirs = [
        os.path.join(out_root, d)
        for d in os.listdir(out_root)
        if _RUN_ID_RE.match(d) and os.path.isdir(os.path.join(out_root, d))
    ]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def _csv_to_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def export_vlmeval_scores_json(run_dir: str, out_name: str = "unified_results.json") -> Optional[str]:
    """
    Merge VLMEvalKit per-dataset *_score.csv / *_acc.csv under run_dir into one JSON
    (EvalScope VLMEval backend does not return metrics dict nor feed Summarizer).
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None
    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": os.path.abspath(run_dir),
        "datasets": {},
    }
    for root, _dirs, files in os.walk(run_dir):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            if "score" not in fn and "acc" not in fn:
                continue
            rel = os.path.relpath(os.path.join(root, fn), run_dir)
            key = rel.replace(os.sep, "/")
            try:
                payload["datasets"][key] = _csv_to_rows(os.path.join(root, fn))
            except OSError:
                continue
    if not payload["datasets"]:
        return None
    out_path = os.path.join(run_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def run_eval():
    # # Option 1: Python dictionary
    # task_cfg = task_cfg_dict
    # Option 2: YAML configuration file (path relative to this script, not CWD)
    if load_dotenv is not None:
        load_dotenv(os.path.join(_here, '.env'), override=False)
    task_cfg = os.path.join(_here, 'eval_openai_api.yaml')

    # VLMEvalKit 后端：run_task 在 evalscope 里对非 Native 固定返回 {}，不汇总指标。
    run_task(task_cfg=task_cfg)

    out_root = os.path.join(_here, "outputs")
    latest_run = _latest_timestamp_run_dir(out_root)
    if latest_run:
        merged_path = export_vlmeval_scores_json(latest_run)
        if merged_path:
            print(f"\n>> Unified results JSON: {merged_path}")
        else:
            print("\n>> No *score*/*acc* CSV found under the latest run dir; skip unified JSON.")

    print(">> Start to get the report with summarizer ...")
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    # Summarizer 主要面向 evalscope native 的 reports/；VLMEvalKit 常为 []。
    print(f"\n>> The report list: {report_list}")


if __name__ == "__main__":
    run_eval()