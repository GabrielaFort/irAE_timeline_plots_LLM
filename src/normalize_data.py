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


def split_combo(value):
    normalized = str(value).replace("/", "+")
    return [part.strip() for part in normalized.split("+") if part.strip()]


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def mapped_ici_regimen(condition, ici_map):
    return mapped_regimen(condition, ici_map)


def mapped_regimen(condition, treatment_map, keep_unmapped=True):
    if condition is None:
        return None

    mapped_parts = []
    for part in split_combo(condition):
        mapped_part = treatment_map.get(part.lower(), part if keep_unmapped else None)
        if is_empty_mapped_value(mapped_part):
            continue
        mapped_parts.append(str(mapped_part))

    if not mapped_parts:
        return None

    return " + ".join(sorted(unique_in_order(mapped_parts)))


def mapped_all_cancer_therapy_regimen(condition, therapy_maps):
    if condition is None:
        return None

    mapped_parts = []
    for part in split_combo(condition):
        part_key = part.lower()
        mapped_part = None
        for therapy_map in therapy_maps:
            mapped_part = therapy_map.get(part_key)
            if not is_empty_mapped_value(mapped_part):
                break
        if is_empty_mapped_value(mapped_part):
            mapped_part = part
        mapped_parts.append(str(mapped_part))

    if not mapped_parts:
        return None

    return " + ".join(sorted(unique_in_order(mapped_parts)))


def mapped_ici_class(condition, ici_class_map):
    if condition is None:
        return None

    classes = [
        str(ici_class_map.get(part.lower(), part))
        for part in split_combo(condition)
    ]
    return " + ".join(sorted(unique_in_order(classes)))


def immunotherapy_episodes(parsed_events, ici_map, irae_treatment_map, ici_class_map):
    episodes = []
    seen = set()

    for event, start, _ in sorted(parsed_events, key=lambda item: item[1]):
        if event.get("condition_type") != "immunotherapy":
            continue

        label = mapped_ici_regimen(event.get("condition"), ici_map)
        if is_empty_mapped_value(label):
            continue

        key = (start, label)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(
            {
                "start": start,
                "label": label,
                "class_label": mapped_ici_class(label, ici_class_map),
            }
        )

    return episodes


def latest_episode_before(episodes, value):
    previous_episodes = [episode for episode in episodes if episode["start"] <= value]
    if not previous_episodes:
        return None
    return previous_episodes[-1]


def read_jsonl(path):
    records = []
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            index = 0
            while index < len(line):
                while index < len(line) and line[index].isspace():
                    index += 1
                if index >= len(line):
                    break
                try:
                    record, index = decoder.raw_decode(line, index)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e
                records.append(record)
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


def mapped_condition(event, ici_map, irae_treatment_map, chemotherapy_map=None, targeted_therapy_map=None):
    condition = event.get("condition")
    condition_type = event.get("condition_type")
    if condition_type == "immunotherapy":
        return mapped_all_cancer_therapy_regimen(
            condition,
            [
                ici_map,
                chemotherapy_map or {},
                targeted_therapy_map or {},
            ],
        )

    lookup = {
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


def has_valid_condition(events, condition_type):
    return any(
        event.get("condition_type") == condition_type
        and not is_empty_mapped_value(event.get("condition"))
        and str(event.get("condition", "")).strip().lower() != "unknown"
        for event in events
    )


def cohort_exclusion_reason(events):
    if not has_valid_condition(events, "irae"):
        return "no_irae_after_normalization"
    if not has_valid_condition(events, "immunotherapy"):
        return "no_valid_immunotherapy_after_normalization"
    return None


def append_skip_log(log_path, record):
    if log_path is None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def normalize_records(
    records,
    ici_map=None,
    chemotherapy_map=None,
    targeted_therapy_map=None,
    irae_treatment_map=None,
    irae_treatment_type_map=None,
    ici_class_map=None,
    skip_log_path=None,
):
    ici_map = normalized_map(ici_map or {})
    chemotherapy_map = normalized_map(chemotherapy_map or {})
    targeted_therapy_map = normalized_map(targeted_therapy_map or {})
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
            if start is None:
                continue
            parsed.append((event, start, start))

        if not parsed:
            append_skip_log(
                skip_log_path,
                {
                    "patient_id": patient_id,
                    "source_files": sorted(
                        {
                            str(event.get("source_file"))
                            for event in events
                            if event.get("source_file")
                        }
                    ),
                    "reason": "no_irae_after_normalization",
                    "stage": "normalization",
                    "event_count": 0,
                },
            )
            continue

        baseline = min(start for _, start, _ in parsed)
        episodes = immunotherapy_episodes(parsed, ici_map, irae_treatment_map, ici_class_map)
        patient_normalized = []
        for event, start, end in parsed:
            condition = mapped_condition(
                event,
                ici_map,
                irae_treatment_map,
                chemotherapy_map=chemotherapy_map,
                targeted_therapy_map=targeted_therapy_map,
            )
            if (
                event.get("condition_type") in {"immunotherapy", "irae_treatment"}
                and is_empty_mapped_value(condition)
            ):
                continue
            ici_class = None
            ici_combo = None
            ici_class_combo = None
            chemotherapy = None
            targeted_therapy = None
            if event.get("condition_type") == "immunotherapy":
                ici_combo = mapped_ici_regimen(event.get("condition"), ici_map)
                ici_class_combo = mapped_ici_class(ici_combo, ici_class_map)
                ici_class = ici_class_combo
                chemotherapy = mapped_regimen(event.get("condition"), chemotherapy_map, keep_unmapped=False)
                targeted_therapy = mapped_regimen(event.get("condition"), targeted_therapy_map, keep_unmapped=False)

            irae_treatment_type = None
            if event.get("condition_type") == "irae_treatment":
                irae_treatment_type = mapped_irae_treatment_type(condition, irae_treatment_type_map)
            time_to_onset_months = None
            associated_ici = None
            associated_ici_class = None
            if event.get("condition_type") == "irae":
                episode = latest_episode_before(episodes, start)
                if episode is not None:
                    associated_ici = episode["label"]
                    associated_ici_class = episode["class_label"]
                    time_to_onset_months = months_since(start, episode["start"])

            patient_normalized.append(
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
                    "chemotherapy": chemotherapy,
                    "targeted_therapy": targeted_therapy,
                    "irae_type": event.get("irae_type"),
                    "irae_treatment_type": irae_treatment_type,
                    "associated_ici": associated_ici,
                    "associated_ici_class": associated_ici_class,
                    "time_to_onset_months": time_to_onset_months,
                    "time_start": months_since(start, baseline),
                }
            )

        reason = cohort_exclusion_reason(patient_normalized)
        if reason:
            append_skip_log(
                skip_log_path,
                {
                    "patient_id": patient_id,
                    "source_files": sorted(
                        {
                            str(event.get("source_file"))
                            for event in events
                            if event.get("source_file")
                        }
                    ),
                    "reason": reason,
                    "stage": "normalization",
                    "event_count": len(patient_normalized),
                },
            )
            continue

        normalized.extend(patient_normalized)

    return deduplicate_records(normalized)


def deduplicate_records(records):
    deduplicated = []
    seen = set()

    for record in records:
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(record)

    return deduplicated


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
    parser.add_argument("--chemotherapy-map", default="data/treatment_terms/chemotherapy_map.json")
    parser.add_argument("--targeted-therapy-map", default="data/treatment_terms/targeted_therapy_map.json")
    parser.add_argument("--irae-treatment-map", default="data/treatment_terms/irae_treatment_map.json")
    parser.add_argument("--irae-treatment-type-map", default="data/treatment_terms/irae_treatment_type_map.json")
    parser.add_argument("--skip-log", default="data/patient_events_skipped.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = read_jsonl(input_path)
    normalized = normalize_records(
        records,
        ici_map=read_json(Path(args.ici_map)),
        ici_class_map=read_json(Path(args.ici_class_map)),
        chemotherapy_map=read_json(Path(args.chemotherapy_map)),
        targeted_therapy_map=read_json(Path(args.targeted_therapy_map)),
        irae_treatment_map=read_json(Path(args.irae_treatment_map)),
        irae_treatment_type_map=read_json(Path(args.irae_treatment_type_map)),
        skip_log_path=Path(args.skip_log) if args.skip_log else None,
    )
    write_jsonl(normalized, output_path)

    print(f"Wrote {len(normalized)} normalized events to {output_path}")
