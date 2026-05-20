import argparse
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


DAYS_PER_MONTH = 30.4375


def parse_date(value):
    if value is None:
        return None

    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "na", "n/a"}:
        return None

    if len(value) == 4 and value.isdigit():
        return date(int(value), 1, 1)

    if len(value) == 7 and "-" in value:
        try:
            year, month = map(int, value.split("-"))
            return date(year, month, 1)
        except ValueError:
            return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        pass

    for fmt in ("%m/%d/%Y", "%m/%Y", "%Y/%m/%d", "%Y/%m", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def months_since(start, baseline):
    return round((start - baseline).days / DAYS_PER_MONTH, 2)


def mapped_ici_class(condition, ici_class_map):
    if condition is None:
        return None

    return ici_class_map.get(str(condition).strip().lower(), condition)


def immunotherapy_date_labels(parsed_events, ici_map, irae_treatment_map, ici_class_map):
    drugs_by_date = defaultdict(set)
    classes_by_date = defaultdict(set)

    for event, start, _ in parsed_events:
        if event.get("condition_type") != "immunotherapy":
            continue

        condition = mapped_condition(event, ici_map, irae_treatment_map)
        if is_empty_mapped_value(condition):
            continue
        drugs_by_date[start].add(str(condition))
        classes_by_date[start].add(str(mapped_ici_class(condition, ici_class_map)))

    labels = {}
    for start in drugs_by_date:
        labels[start] = {
            "ici_combo": " + ".join(sorted(drugs_by_date[start])),
            "ici_class_combo": " + ".join(sorted(classes_by_date[start])),
        }

    return labels


def immunotherapy_episodes(date_labels):
    episodes = []
    last_label = None
    for start in sorted(date_labels):
        label = date_labels[start]["ici_combo"]
        if label != last_label:
            episodes.append(
                {
                    "start": start,
                    "label": label,
                    "class_label": date_labels[start]["ici_class_combo"],
                }
            )
            last_label = label

    return episodes


def latest_episode_before(episodes, value):
    previous_episodes = [episode for episode in episodes if episode["start"] <= value]
    if not previous_episodes:
        return None
    return previous_episodes[-1]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalized_map(raw_map):
    normalized = {}
    for key, value in raw_map.items():
        if isinstance(value, list):
            for item in value:
                normalized[str(item).strip().lower()] = key
        else:
            normalized[str(key).strip().lower()] = value
    return normalized


def mapped_condition(event, ici_map, irae_treatment_map):
    condition = event.get("condition")
    condition_type = event.get("condition_type")
    lookup = {
        "immunotherapy": ici_map,
        "irae_treatment": irae_treatment_map,
    }.get(condition_type)

    if lookup is None or condition is None:
        return condition

    return lookup.get(str(condition).strip().lower(), condition)


def mapped_irae_treatment_type(condition, irae_treatment_type_map):
    if condition is None:
        return None

    return irae_treatment_type_map.get(str(condition).strip().lower(), condition)


def is_empty_mapped_value(value):
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null", "na", "n/a"}


def normalize_records(
    records,
    ici_map=None,
    irae_treatment_map=None,
    irae_treatment_type_map=None,
    ici_class_map=None,
):
    ici_map = normalized_map(ici_map or {})
    irae_treatment_map = normalized_map(irae_treatment_map or {})
    irae_treatment_type_map = normalized_map(irae_treatment_type_map or {})
    ici_class_map = normalized_map(ici_class_map or {})
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
        ici_date_labels = immunotherapy_date_labels(parsed, ici_map, irae_treatment_map, ici_class_map)
        episodes = immunotherapy_episodes(ici_date_labels)
        for event, start, end in parsed:
            condition = mapped_condition(event, ici_map, irae_treatment_map)
            if (
                event.get("condition_type") in {"immunotherapy", "irae_treatment"}
                and is_empty_mapped_value(condition)
            ):
                continue
            ici_class = None
            ici_combo = None
            ici_class_combo = None
            if event.get("condition_type") == "immunotherapy":
                ici_class = mapped_ici_class(condition, ici_class_map)
                date_labels = ici_date_labels.get(start, {})
                ici_combo = date_labels.get("ici_combo")
                ici_class_combo = date_labels.get("ici_class_combo")

            irae_treatment_type = None
            if event.get("condition_type") == "irae_treatment":
                irae_treatment_type = mapped_irae_treatment_type(condition, irae_treatment_type_map)
            time_to_onset_months = None
            associated_ici = None
            associated_ici_class = None
            associated_ici_start_date = None
            if event.get("condition_type") == "irae":
                episode = latest_episode_before(episodes, start)
                if episode is not None:
                    associated_ici = episode["label"]
                    associated_ici_class = episode["class_label"]
                    associated_ici_start_date = episode["start"].isoformat()
                    time_to_onset_months = months_since(start, episode["start"])

            normalized.append(
                {
                    "patient_id": patient_id,
                    "source_file": event.get("source_file"),
                    "oncotree_tissue": event.get("oncotree_tissue"),
                    "oncotree_name": event.get("oncotree_name"),
                    "oncotree_code": event.get("oncotree_code"),
                    "condition_type": event.get("condition_type"),
                    "condition": condition,
                    "raw_condition": event.get("raw_condition") or (
                        event.get("condition") if condition != event.get("condition") else None
                    ),
                    "ici_class": ici_class,
                    "ici_combo": ici_combo,
                    "ici_class_combo": ici_class_combo,
                    "irae_type": event.get("irae_type"),
                    "irae_treatment_type": irae_treatment_type,
                    "associated_ici": associated_ici,
                    "associated_ici_class": associated_ici_class,
                    "associated_ici_start_date": associated_ici_start_date,
                    "time_to_onset_months": time_to_onset_months,
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
    parser.add_argument("--ici-map", default="data/treatment_terms/ici_map.json")
    parser.add_argument("--ici-class-map", default="data/treatment_terms/ici_class_map.json")
    parser.add_argument("--irae-treatment-map", default="data/treatment_terms/irae_treatment_map.json")
    parser.add_argument("--irae-treatment-type-map", default="data/treatment_terms/irae_treatment_type_map.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = read_jsonl(input_path)
    normalized = normalize_records(
        records,
        ici_map=read_json(Path(args.ici_map)),
        ici_class_map=read_json(Path(args.ici_class_map)),
        irae_treatment_map=read_json(Path(args.irae_treatment_map)),
        irae_treatment_type_map=read_json(Path(args.irae_treatment_type_map)),
    )
    write_jsonl(normalized, output_path)

    print(f"Wrote {len(normalized)} normalized events to {output_path}")
