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


THERAPY_MAP_SPECS = [
    ("ICI", "ici_map"),
    ("ADC", "adc_map"),
    ("chemotherapy", "chemotherapy_map"),
    ("targeted therapy", "targeted_therapy_map"),
]

THERAPY_TYPE_ORDER = {
    "ICI": 0,
    "targeted therapy": 1,
    "chemotherapy": 2,
    "ADC": 3,
    "unmapped": 4,
}


def normalize_therapy_regimen(condition, therapy_maps):
    if condition is None:
        return {
            "condition": None,
            "therapy_type": None,
            "ici_combo": None,
            "has_ici": False,
        }

    mapped_entries = []

    for part in split_combo(condition):
        part_key = part.lower()
        matched_null = False
        matched_value = None
        matched_type = None

        for therapy_type, map_name in THERAPY_MAP_SPECS:
            therapy_map = therapy_maps.get(map_name, {})
            if part_key not in therapy_map:
                continue

            value = therapy_map[part_key]
            if is_empty_mapped_value(value):
                matched_null = True
                continue

            matched_value = str(value)
            matched_type = therapy_type
            break

        if matched_value is None:
            if matched_null:
                continue
            matched_value = part_key
            matched_type = "unmapped"

        mapped_entries.append((matched_value, matched_type))

    mapped_entries = sorted(
        unique_in_order(mapped_entries),
        key=lambda item: (THERAPY_TYPE_ORDER.get(item[1], 99), item[0].lower()),
    )
    mapped_parts = [value for value, _ in mapped_entries]
    therapy_types = [therapy_type for _, therapy_type in mapped_entries]
    ici_parts = [value for value, therapy_type in mapped_entries if therapy_type == "ICI"]

    return {
        "condition": " + ".join(mapped_parts) if mapped_parts else None,
        "therapy_type": " + ".join(therapy_types) if therapy_types else None,
        "therapy_type_consolidated": consolidated_therapy_type(therapy_types),
        "ici_combo": " + ".join(ici_parts) if ici_parts else None,
        "has_ici": bool(ici_parts),
    }


def consolidated_therapy_type(therapy_types):
    if not therapy_types:
        return None

    ici_count = sum(1 for therapy_type in therapy_types if therapy_type == "ICI")
    non_ici_types = unique_in_order(
        therapy_type for therapy_type in therapy_types if therapy_type != "ICI"
    )

    if ici_count == 0:
        return " plus ".join(non_ici_types) if non_ici_types else None

    if not non_ici_types:
        return "combo ICI" if ici_count > 1 else "single ICI"

    ici_label = "combo ICI" if ici_count > 1 else "ICI"
    return " plus ".join([ici_label, *non_ici_types])


def mapped_ici_class(condition, ici_class_map):
    if condition is None:
        return None

    classes = [
        str(ici_class_map.get(part.lower(), part))
        for part in split_combo(condition)
    ]
    return " + ".join(sorted(unique_in_order(classes)))


def immunotherapy_episodes(parsed_events, therapy_maps, ici_class_map):
    episodes = []
    seen = set()

    for event, start, _ in sorted(parsed_events, key=lambda item: item[1]):
        if event.get("condition_type") != "immunotherapy":
            continue

        regimen = normalize_therapy_regimen(event.get("condition"), therapy_maps)
        label = regimen["ici_combo"]
        if is_empty_mapped_value(label) or not regimen["has_ici"]:
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
                "full_regimen": regimen["condition"],
                "therapy_type": regimen["therapy_type"],
                "therapy_type_consolidated": regimen["therapy_type_consolidated"],
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


def mapped_condition(event, ici_map, irae_treatment_map):
    condition = event.get("condition")
    condition_type = event.get("condition_type")

    lookup = {
        "irae_treatment": irae_treatment_map,
    }.get(condition_type)

    if lookup is None or condition is None:
        return condition

    condition_key = str(condition).strip().lower()
    return lookup.get(condition_key, condition_key)


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
    if not has_valid_ici_immunotherapy(events):
        return "no_ici_after_immunotherapy_normalization"
    return None


def has_valid_ici_immunotherapy(events):
    return any(
        event.get("condition_type") == "immunotherapy"
        and not is_empty_mapped_value(event.get("condition"))
        and not is_empty_mapped_value(event.get("ici_combo"))
        for event in events
    )


def append_skip_log(log_path, record):
    if log_path is None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def normalize_records(
    records,
    ici_map=None,
    adc_map=None,
    chemotherapy_map=None,
    targeted_therapy_map=None,
    irae_treatment_map=None,
    irae_treatment_type_map=None,
    ici_class_map=None,
    skip_log_path=None,
):
    ici_map = normalized_map(ici_map or {})
    adc_map = normalized_map(adc_map or {})
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

        therapy_maps = {
            "ici_map": ici_map,
            "adc_map": adc_map,
            "chemotherapy_map": chemotherapy_map,
            "targeted_therapy_map": targeted_therapy_map,
        }
        baseline = min(start for _, start, _ in parsed)
        episodes = immunotherapy_episodes(parsed, therapy_maps, ici_class_map)
        patient_normalized = []
        for event, start, end in parsed:
            therapy_type = None
            therapy_type_consolidated = None
            ici_combo = None
            if event.get("condition_type") == "immunotherapy":
                regimen = normalize_therapy_regimen(event.get("condition"), therapy_maps)
                condition = regimen["condition"]
                therapy_type = regimen["therapy_type"]
                therapy_type_consolidated = regimen["therapy_type_consolidated"]
                ici_combo = regimen["ici_combo"]
            else:
                condition = mapped_condition(event, ici_map, irae_treatment_map)

            if (
                event.get("condition_type") in {"immunotherapy", "irae_treatment"}
                and is_empty_mapped_value(condition)
            ):
                continue
            if event.get("condition_type") == "immunotherapy" and is_empty_mapped_value(ici_combo):
                continue
            ici_class = None
            ici_class_combo = None
            if event.get("condition_type") == "immunotherapy":
                ici_class_combo = mapped_ici_class(ici_combo, ici_class_map)
                ici_class = ici_class_combo

            irae_treatment_type = None
            if event.get("condition_type") == "irae_treatment":
                irae_treatment_type = mapped_irae_treatment_type(condition, irae_treatment_type_map)
            time_to_onset_months = None
            associated_ici = None
            associated_ici_class = None
            associated_treatment = None
            associated_therapy_type = None
            associated_therapy_type_consolidated = None
            if event.get("condition_type") == "irae":
                episode = latest_episode_before(episodes, start)
                if episode is not None:
                    associated_ici = episode["label"]
                    associated_ici_class = episode["class_label"]
                    associated_treatment = episode["full_regimen"]
                    associated_therapy_type = episode["therapy_type"]
                    associated_therapy_type_consolidated = episode["therapy_type_consolidated"]
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
                    "therapy_type": therapy_type,
                    "therapy_type_consolidated": therapy_type_consolidated,
                    "irae_type": event.get("irae_type"),
                    "irae_treatment_type": irae_treatment_type,
                    "associated_ici": associated_ici,
                    "associated_ici_class": associated_ici_class,
                    "associated_treatment": associated_treatment,
                    "associated_therapy_type": associated_therapy_type,
                    "associated_therapy_type_consolidated": associated_therapy_type_consolidated,
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
    parser.add_argument("--adc-map", default="data/treatment_terms/adc_map.json")
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
        adc_map=read_json(Path(args.adc_map)),
        ici_class_map=read_json(Path(args.ici_class_map)),
        chemotherapy_map=read_json(Path(args.chemotherapy_map)),
        targeted_therapy_map=read_json(Path(args.targeted_therapy_map)),
        irae_treatment_map=read_json(Path(args.irae_treatment_map)),
        irae_treatment_type_map=read_json(Path(args.irae_treatment_type_map)),
        skip_log_path=Path(args.skip_log) if args.skip_log else None,
    )
    write_jsonl(normalized, output_path)

    print(f"Wrote {len(normalized)} normalized events to {output_path}")
