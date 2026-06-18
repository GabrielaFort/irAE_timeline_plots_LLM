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
    return [part.strip() for part in str(value).split("+") if part.strip()]


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


THERAPY_TYPE_ORDER = {
    "ICI": 0,
    "targeted therapy": 1,
    "chemotherapy": 2,
    "ADC": 3,
    "unmapped": 4,
}


def normalize_therapy_regimen(condition, rxnorm_cache, ingredient_class_map):
    if condition is None:
        return {
            "condition": None,
            "therapy_type": None,
            "therapy_type_consolidated": None,
            "ici_combo": None,
            "ici_class_combo": None,
            "has_ici": False,
            "ingredients": [],
        }

    ingredients = rxnorm_ingredients_for_condition(condition, rxnorm_cache)
    mapped_entries = []
    for ingredient in ingredients:
        class_info = ingredient_class(ingredient, ingredient_class_map)
        therapy_type = class_info.get("therapy_type") or "unmapped"
        mapped_entries.append(
            (
                display_ingredient_name(ingredient, ingredient_class_map),
                therapy_type,
                ingredient,
            )
        )

    if any(therapy_type == "ICI" for _, therapy_type, _ in mapped_entries):
        mapped_entries = [
            (value, therapy_type, ingredient)
            for value, therapy_type, ingredient in mapped_entries
            if not (
                therapy_type == "unmapped"
                and ingredient_class(ingredient, ingredient_class_map).get("irae_treatment_type")
            )
        ]

    seen_entries = set()
    unique_entries = []
    for value, therapy_type, ingredient in mapped_entries:
        key = (ingredient.get("rxcui") or value.lower(), therapy_type)
        if key in seen_entries:
            continue
        seen_entries.add(key)
        unique_entries.append((value, therapy_type, ingredient))
    mapped_entries = sorted(
        unique_entries,
        key=lambda item: (THERAPY_TYPE_ORDER.get(item[1], 99), item[0].lower()),
    )
    mapped_parts = [value for value, _, _ in mapped_entries]
    therapy_types = [therapy_type for _, therapy_type, _ in mapped_entries]
    ici_ingredients = [
        ingredient for _, therapy_type, ingredient in mapped_entries if therapy_type == "ICI"
    ]
    ici_parts = ingredient_names(ici_ingredients, ingredient_class_map)

    return {
        "condition": " + ".join(mapped_parts) if mapped_parts else None,
        "therapy_type": " + ".join(therapy_types) if therapy_types else None,
        "therapy_type_consolidated": consolidated_therapy_type(therapy_types),
        "ici_combo": " + ".join(ici_parts) if ici_parts else None,
        "ici_class_combo": mapped_rxnorm_ici_class(ici_ingredients, ingredient_class_map),
        "has_ici": bool(ici_parts),
        "ingredients": ingredients,
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


def immunotherapy_episodes(
    parsed_events,
    rxnorm_cache,
    ingredient_class_map,
):
    episodes = []
    seen = set()

    for event, start, _ in sorted(parsed_events, key=lambda item: item[1]):
        if event.get("condition_type") != "immunotherapy":
            continue

        regimen = normalize_therapy_regimen(
            event.get("condition"),
            rxnorm_cache,
            ingredient_class_map,
        )
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
                "class_label": regimen.get("ici_class_combo"),
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


def read_json_or_empty(path):
    if path is None or not path.exists():
        return {}
    return read_json(path)


def accepted_rxnorm_entry(term, rxnorm_cache):
    entry = (rxnorm_cache or {}).get(str(term or "").strip().lower())
    if not entry or entry.get("status") != "accepted":
        return None
    return entry


def normalized_ingredient_name(ingredient):
    return str(ingredient.get("name") or "").strip().lower()


def ingredient_class(ingredient, ingredient_class_map):
    value = ingredient_class_map.get(normalized_ingredient_name(ingredient))
    if isinstance(value, str):
        return {"therapy_type": value}
    return value or {}


def display_ingredient_name(ingredient, ingredient_class_map):
    class_info = ingredient_class(ingredient, ingredient_class_map)
    if class_info.get("name"):
        return str(class_info["name"])
    return normalized_ingredient_name(ingredient)


def unique_ingredients(ingredients):
    seen = set()
    out = []
    for ingredient in ingredients:
        key = ingredient.get("rxcui") or str(ingredient.get("name", "")).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ingredient)
    return out


def rxnorm_ingredients_for_condition(condition, rxnorm_cache):
    ingredients = []
    for part in split_combo(condition):
        entry = accepted_rxnorm_entry(part, rxnorm_cache)
        if not entry:
            continue
        ingredients.extend(entry.get("ingredients") or [])
    return unique_ingredients(ingredients)


def ingredient_names(ingredients, ingredient_class_map):
    return [
        display_ingredient_name(ingredient, ingredient_class_map)
        for ingredient in ingredients
        if normalized_ingredient_name(ingredient)
    ]


def ingredient_log_values(ingredients):
    return [
        {
            "rxcui": ingredient.get("rxcui"),
            "name": ingredient.get("name"),
            "normalized_name": normalized_ingredient_name(ingredient),
        }
        for ingredient in ingredients
    ]


def mapped_rxnorm_irae_treatment_type(ingredients, ingredient_class_map):
    treatment_types = []
    for ingredient in ingredients:
        treatment_type = ingredient_class(ingredient, ingredient_class_map).get("irae_treatment_type")
        if treatment_type:
            treatment_types.append(str(treatment_type))
    return " + ".join(sorted(unique_in_order(treatment_types))) if treatment_types else None


def mapped_rxnorm_ici_class(ingredients, ingredient_class_map):
    classes = []
    for ingredient in ingredients:
        ici_class = ingredient_class(ingredient, ingredient_class_map).get("ici_class")
        if ici_class:
            classes.append(str(ici_class))
    return " + ".join(sorted(unique_in_order(classes))) if classes else None


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


def append_row_skip(row_skip_log_path, event, patient_id, reason, condition=None, ingredients=None):
    append_skip_log(
        row_skip_log_path,
        {
            "patient_id": patient_id,
            "source_file": event.get("source_file"),
            "condition_type": event.get("condition_type"),
            "condition": event.get("condition"),
            "start_date": event.get("start_date"),
            "reason": reason,
            "normalized_condition": condition,
            "ingredients": ingredient_log_values(ingredients or []),
        },
    )


def normalize_records(
    records,
    rxnorm_cache=None,
    ingredient_class_map=None,
    skip_log_path=None,
    row_skip_log_path=None,
):
    rxnorm_cache = rxnorm_cache or {}
    ingredient_class_map = ingredient_class_map or {}
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
        episodes = immunotherapy_episodes(
            parsed,
            rxnorm_cache,
            ingredient_class_map,
        )
        patient_normalized = []
        for event, start, end in parsed:
            therapy_type = None
            therapy_type_consolidated = None
            ici_combo = None
            rxnorm_ingredients = []
            if event.get("condition_type") == "immunotherapy":
                regimen = normalize_therapy_regimen(
                    event.get("condition"),
                    rxnorm_cache,
                    ingredient_class_map,
                )
                condition = regimen["condition"]
                therapy_type = regimen["therapy_type"]
                therapy_type_consolidated = regimen["therapy_type_consolidated"]
                ici_combo = regimen["ici_combo"]
                rxnorm_ingredients = regimen.get("ingredients") or []
            else:
                if event.get("condition_type") == "irae_treatment":
                    rxnorm_ingredients = rxnorm_ingredients_for_condition(
                        event.get("condition"),
                        rxnorm_cache,
                    )
                    names = ingredient_names(rxnorm_ingredients, ingredient_class_map)
                    condition = " + ".join(names) if names else None
                else:
                    condition = event.get("condition")

            if (
                event.get("condition_type") in {"immunotherapy", "irae_treatment"}
                and is_empty_mapped_value(condition)
            ):
                append_row_skip(
                    row_skip_log_path,
                    event,
                    patient_id,
                    "no_accepted_rxnorm_ingredients",
                    condition=condition,
                    ingredients=rxnorm_ingredients,
                )
                continue
            if event.get("condition_type") == "immunotherapy" and is_empty_mapped_value(ici_combo):
                append_row_skip(
                    row_skip_log_path,
                    event,
                    patient_id,
                    "no_ici_after_class_mapping",
                    condition=condition,
                    ingredients=rxnorm_ingredients,
                )
                continue
            ici_class = None
            ici_class_combo = None
            if event.get("condition_type") == "immunotherapy":
                ici_class_combo = mapped_rxnorm_ici_class(
                    [
                        ingredient
                        for ingredient in rxnorm_ingredients
                        if ingredient_class(ingredient, ingredient_class_map).get("therapy_type") == "ICI"
                    ],
                    ingredient_class_map,
                )
                ici_class = ici_class_combo

            irae_treatment_type = None
            if event.get("condition_type") == "irae_treatment":
                irae_treatment_type = mapped_rxnorm_irae_treatment_type(
                    rxnorm_ingredients,
                    ingredient_class_map,
                )
                if is_empty_mapped_value(irae_treatment_type):
                    append_row_skip(
                        row_skip_log_path,
                        event,
                        patient_id,
                        "no_irae_treatment_type_after_class_mapping",
                        condition=condition,
                        ingredients=rxnorm_ingredients,
                    )
                    continue
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
    parser.add_argument("--rxnorm-cache", default="data/treatment_terms/rxnorm_cache.json")
    parser.add_argument("--ingredient-class-map", default="data/treatment_terms/ingredient_class_map.json")
    parser.add_argument("--skip-log", default="data/patient_events_skipped.jsonl")
    parser.add_argument("--row-skip-log", default="data/patient_event_rows_skipped.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    row_skip_log_path = Path(args.row_skip_log) if args.row_skip_log else None
    if row_skip_log_path and row_skip_log_path.exists():
        row_skip_log_path.unlink()

    records = read_jsonl(input_path)
    normalized = normalize_records(
        records,
        rxnorm_cache=read_json_or_empty(Path(args.rxnorm_cache)) if args.rxnorm_cache else {},
        ingredient_class_map=read_json_or_empty(Path(args.ingredient_class_map)) if args.ingredient_class_map else {},
        skip_log_path=Path(args.skip_log) if args.skip_log else None,
        row_skip_log_path=row_skip_log_path,
    )
    write_jsonl(normalized, output_path)

    print(f"Wrote {len(normalized)} normalized events to {output_path}")
