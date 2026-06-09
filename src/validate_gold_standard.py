import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EVENT_TYPES = ("immunotherapy", "irae", "irae_treatment")
ONCOTREE_FIELDS = ("oncotree_tissue", "oncotree_name")
KEY_MODES = ("condition", "time", "condition_and_time")


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {e}") from e
    return records


def normalized_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalized_combo_text(value):
    text = normalized_text(value)
    if " + " not in text and "+" not in text:
        return text

    parts = [
        normalized_text(part)
        for part in text.split("+")
        if normalized_text(part)
    ]
    return " + ".join(sorted(parts))


def normalized_time(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return normalized_text(value)


def by_source_file(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("source_file") or "unknown"].append(record)
    return grouped


def event_key(record, key_mode):
    condition_type = normalized_text(record.get("condition_type"))
    if key_mode == "condition":
        return (
            condition_type,
            normalized_combo_text(record.get("condition")),
        )
    if key_mode == "time":
        return (
            condition_type,
            normalized_time(record.get("time_start")),
        )
    if key_mode == "condition_and_time":
        return (
            condition_type,
            normalized_combo_text(record.get("condition")),
            normalized_time(record.get("time_start")),
        )
    raise ValueError(f"Unknown key mode: {key_mode}")


def compare_multisets(gold_records, pred_records, key_mode):
    gold_counts = Counter(event_key(record, key_mode) for record in gold_records)
    pred_counts = Counter(event_key(record, key_mode) for record in pred_records)

    tp = sum((gold_counts & pred_counts).values())
    fp = sum((pred_counts - gold_counts).values())
    fn = sum((gold_counts - pred_counts).values())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": gold_counts - pred_counts,
        "extra": pred_counts - gold_counts,
    }


def oncotree_accuracy_rows(source_file, oncotree_rows):
    rows = []
    matches = sum(1 for row in oncotree_rows if row["match"])
    total = len(oncotree_rows)

    for field in ONCOTREE_FIELDS:
        field_rows = [row for row in oncotree_rows if row["field"] == field]
        field_matches = sum(1 for row in field_rows if row["match"])
        rows.append(accuracy_row(source_file, f"oncotree_{field}", field_matches, len(field_rows)))

    rows.append(accuracy_row(source_file, "oncotree_overall", matches, total))
    return rows


def patient_oncotree_result(source_file, gold_records, pred_records):
    rows = []
    for field in ONCOTREE_FIELDS:
        gold_value = next((record.get(field) for record in gold_records if record.get(field)), None)
        pred_value = next((record.get(field) for record in pred_records if record.get(field)), None)
        rows.append(
            {
                "source_file": source_file,
                "level": "oncotree",
                "field": field,
                "gold": gold_value,
                "predicted": pred_value,
                "match": normalized_text(gold_value) == normalized_text(pred_value),
            }
        )
    return rows


def filtered(records, condition_type):
    return [record for record in records if record.get("condition_type") == condition_type]


def issue_rows(source_file, result, level):
    rows = []
    for key, count in result["missing"].items():
        rows.append(
            {
                "source_file": source_file,
                "level": level,
                "issue": "missing",
                "count": count,
                "event_key": " | ".join(key),
            }
        )
    for key, count in result["extra"].items():
        rows.append(
            {
                "source_file": source_file,
                "level": level,
                "issue": "extra",
                "count": count,
                "event_key": " | ".join(key),
            }
        )
    return rows


def metric_row(source_file, level, result):
    return {
        "source_file": source_file,
        "level": level,
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "precision": round(result["precision"], 3),
        "recall": round(result["recall"], 3),
        "f1": round(result["f1"], 3),
    }


def accuracy_row(source_file, level, correct, total):
    return {
        "source_file": source_file,
        "level": level,
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 3) if total else 0,
    }


def write_csv(rows, path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate normalized pipeline output against normalized gold-standard JSONL.")
    parser.add_argument("--gold", default="data/gold_standard_results_normalized.jsonl")
    parser.add_argument("--pred", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs/gold_validation")
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    pred = read_jsonl(args.pred)
    gold_by_source = by_source_file(gold)
    pred_by_source = by_source_file(pred)

    metric_rows = []
    oncotree_metric_rows = []
    issue_output_rows = []
    oncotree_rows = []

    all_gold_for_scored_patients = []
    all_pred_for_scored_patients = []

    for source_file in sorted(gold_by_source):
        gold_patient = gold_by_source[source_file]
        pred_patient = pred_by_source.get(source_file, [])
        all_gold_for_scored_patients.extend(gold_patient)
        all_pred_for_scored_patients.extend(pred_patient)

        patient_oncotree_rows = patient_oncotree_result(source_file, gold_patient, pred_patient)
        oncotree_rows.extend(patient_oncotree_rows)
        oncotree_metric_rows.extend(oncotree_accuracy_rows(source_file, patient_oncotree_rows))

        for event_type in EVENT_TYPES:
            gold_events = filtered(gold_patient, event_type)
            pred_events = filtered(pred_patient, event_type)
            for key_mode in KEY_MODES:
                result = compare_multisets(gold_events, pred_events, key_mode)
                level = f"{event_type}_{key_mode}"
                metric_rows.append(metric_row(source_file, level, result))
                issue_output_rows.extend(issue_rows(source_file, result, level))

        for key_mode in KEY_MODES:
            result = compare_multisets(gold_patient, pred_patient, key_mode)
            level = f"overall_{key_mode}"
            metric_rows.append(metric_row(source_file, level, result))
            issue_output_rows.extend(issue_rows(source_file, result, level))

    for event_type in EVENT_TYPES:
        gold_events = filtered(all_gold_for_scored_patients, event_type)
        pred_events = filtered(all_pred_for_scored_patients, event_type)
        for key_mode in KEY_MODES:
            result = compare_multisets(gold_events, pred_events, key_mode)
            metric_rows.append(metric_row("ALL", f"{event_type}_{key_mode}", result))

    for key_mode in KEY_MODES:
        result = compare_multisets(all_gold_for_scored_patients, all_pred_for_scored_patients, key_mode)
        metric_rows.append(metric_row("ALL", f"overall_{key_mode}", result))

    oncotree_metric_rows.extend(oncotree_accuracy_rows("ALL", oncotree_rows))

    output_dir = Path(args.output_dir)
    write_csv(
        metric_rows,
        output_dir / "metrics_by_patient.csv",
        ["source_file", "level", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    write_csv(
        issue_output_rows,
        output_dir / "event_issues.csv",
        ["source_file", "level", "issue", "count", "event_key"],
    )
    write_csv(
        oncotree_rows,
        output_dir / "oncotree_issues.csv",
        ["source_file", "level", "field", "gold", "predicted", "match"],
    )
    write_csv(
        oncotree_metric_rows,
        output_dir / "oncotree_metrics.csv",
        ["source_file", "level", "correct", "total", "accuracy"],
    )

    overall = compare_multisets(all_gold_for_scored_patients, all_pred_for_scored_patients, "condition_and_time")
    print(
        "Overall normalized event validation: "
        f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
        f"precision={overall['precision']:.3f} recall={overall['recall']:.3f} f1={overall['f1']:.3f}"
    )
    print(f"Wrote validation reports to {output_dir}")


if __name__ == "__main__":
    main()
