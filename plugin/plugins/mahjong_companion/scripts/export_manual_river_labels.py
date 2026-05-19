from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


SKIP_LABELS = {"", "unknown"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export manually labeled river crops as a training JSONL manifest.")
    parser.add_argument("labels_json", type=Path, help="manual_labels_from_user.json path.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Output summary JSON path.")
    args = parser.parse_args()

    rows = _load_rows(args.labels_json)
    out_path = args.out or args.labels_json.with_name("training_manifest.jsonl")
    summary_path = args.summary_out or out_path.with_suffix(".summary.json")
    records = [_record_from_row(row, labels_path=args.labels_json) for row in rows]
    records = [record for record in records if record is not None]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "source": str(args.labels_json),
        "manifest": str(out_path),
        "record_count": len(records),
        "labels": dict(sorted(Counter(record["label"] for record in records).items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest={out_path}")
    print(f"summary={summary_path}")
    print(f"record_count={len(records)}")
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"labels JSON must be an array: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _record_from_row(row: dict[str, Any], *, labels_path: Path) -> dict[str, Any] | None:
    label = str(row.get("label", "")).strip()
    if label in SKIP_LABELS:
        return None
    image_path = Path(str(row.get("file", "")))
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    if not image_path.exists():
        return None
    return {
        "image": str(image_path),
        "label": label,
        "source_labels": str(labels_path),
        "source_index": int(row.get("index", 0) or 0),
        "source_filename": str(row.get("filename", image_path.name)),
        "reason": str(row.get("reason", "")),
        "text_label": str(row.get("user_label_text", "")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
