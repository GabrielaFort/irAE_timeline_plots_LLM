import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from datetime import datetime


DAYS_PER_MONTH = 30.4375


def parse_date(value):
    if value is None:
        return None

    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "na", "n/a"}:
        return None

    if len(value) == 4:
        return date(int(value), 1, 1)

    if len(value) == 7 and "-" in value:
        year, month = map(int, value.split("-"))
        return date(year, month, 1)

    try:
        return date.fromisoformat(value)
    except ValueError:
        pass

    for fmt in ("%m/%d/%Y", "%m/%Y", "%Y/%m/%d", "%Y/%m"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def months_since(start, baseline):
    return round((start - baseline).days / DAYS_PER_MONTH, 2)

def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_records(records):
    by_patient = defaultdict(list)
    for record in records:
        patient_id = record.get("patient_id") or "unknown"
        by_patient[patient_id].append(record)

    normalized = []
    for patient_id, events in by_patient.items():
        parsed = []
        for event in events:
            if (
                event.get("condition_type") == "irae"
                and (
                    str(event.get("condition", "")).lower() == "unknown"
                    or str(event.get("irae_type", "")).lower() == "unknown"
                )
            ):
                continue

            start = parse_date(event.get("start_date"))
            end = parse_date(event.get("end_date")) or start
            if start is None:
                continue
            if end < start:
                end = start
            parsed.append((event, start, end))

        if not parsed:
            continue

        baseline = min(start for _, start, _ in parsed)
        for event, start, end in parsed:
            normalized.append(
                {
                    "patient_id": patient_id,
                    "source_file": event.get("source_file"),
                    "oncotree_tissue": event.get("oncotree_tissue"),
                    "oncotree_name": event.get("oncotree_name"),
                    "oncotree_code": event.get("oncotree_code"),
                    "condition_type": event.get("condition_type"),
                    "condition": event.get("condition"),
                    "raw_condition": event.get("raw_condition"),
                    "irae_type": event.get("irae_type"),
                    "time_start": months_since(start, baseline),
                    "time_stop": months_since(end, baseline),
                }
            )

    return normalized


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize patient event dates to months from first event.")
    parser.add_argument("--input", default="data/patient_events.jsonl", help="Input event JSONL.")
    parser.add_argument("--output", default="data/patient_events_normalized.jsonl", help="Output normalized JSONL.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = read_jsonl(input_path)
    normalized = normalize_records(records)
    write_jsonl(normalized, output_path)

    print(f"Wrote {len(normalized)} normalized events to {output_path}")
