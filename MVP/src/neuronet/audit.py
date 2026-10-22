import json
from datetime import datetime, timezone
from pathlib import Path

def log_decision(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, default=_json_default) + "\n")

def _json_default(value):
    try:
        return value.item()
    except Exception:
        return str(value)
