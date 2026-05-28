import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    records = []
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            index = 0
            while index < len(line):
                record, index = decoder.raw_decode(line, index)
                records.append(record)
    return records


def is_unknown(value):
    return str(value or "").strip().lower() in {"", "unknown", "none", "null", "na", "n/a"}


def add(counter, category, value):
    value = str(value or "").strip()
    if value:
        counter[(category, value, "")] += 1


def add_with_raw(counter, category, value, raw_value):
    value = str(value or "").strip()
    raw_value = str(raw_value or "").strip()
    if value:
        counter[(category, value, raw_value)] += 1


def collect_terms(records):
    terms = Counter()

    for record in records:
        condition_type = record.get("condition_type")
        condition = record.get("condition")

        if condition_type == "irae":
            if is_unknown(condition):
                add(terms, "unmapped_irae", record.get("raw_condition"))
            else:
                add_with_raw(terms, "mapped_irae", condition, record.get("raw_condition"))
        elif condition_type == "immunotherapy":
            add(terms, "immunotherapy", condition)
        elif condition_type == "irae_treatment":
            add(terms, "irae_treatment", condition)

    return terms


def write_csv(terms, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "term", "raw_term", "count"])
        for (category, term, raw_term), count in sorted(terms.items()):
            writer.writerow([category, term, raw_term, count])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export unique terms for mapping file review.")
    parser.add_argument("--input", default="data/patient_events.jsonl")
    parser.add_argument("--output", default="data/mapping_terms_review.csv")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    terms = collect_terms(records)
    write_csv(terms, Path(args.output))
    print(f"Wrote {len(terms)} unique terms to {args.output}")
