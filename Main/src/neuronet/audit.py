"""Audit logging for all autonomous and escalated network decisions."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

def _json_default(value):
    try:
        return value.item()
    except Exception:
        return str(value)

def log_decision(path: Path, record: Dict[str, Any]) -> None:
    """Appends record to a JSONL audit file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, default=_json_default) + "\n")

def append_decision_log_json(json_path: Path, record: Dict[str, Any], max_records: int = 500) -> None:
    """Appends record to a JSON array file formatted for the Streamlit dashboard."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    records = loaded
        except Exception:
            records = []

    item = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        **record,
    }
    records.append(item)
    if len(records) > max_records:
        records = records[-max_records:]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=_json_default)
