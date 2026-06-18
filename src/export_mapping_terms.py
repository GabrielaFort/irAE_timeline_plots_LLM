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


def split_combo(value):
    return [part.strip() for part in str(value or "").split("+") if part.strip()]


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
            for part in split_combo(condition):
                add(terms, "immunotherapy", part)
        elif condition_type == "irae_treatment":
            for part in split_combo(condition):
                add(terms, "irae_treatment", part)

    return terms


def cache_entry_for(term, rxnorm_cache):
    return (rxnorm_cache or {}).get(str(term or "").strip().lower()) or {}


def normalized_term(entry):
    ingredients = entry.get("ingredients") or []
    return " + ".join(
        str(ingredient.get("name"))
        for ingredient in ingredients
        if ingredient.get("name")
    )


def ingredient_rxcuis(entry):
    ingredients = entry.get("ingredients") or []
    return " + ".join(
        str(ingredient.get("rxcui"))
        for ingredient in ingredients
        if ingredient.get("rxcui")
    )


def write_csv(terms, output_path, rxnorm_cache=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "category",
                "term",
                "raw_term",
                "count",
                "rxnorm_status",
                "rxnorm_match_method",
                "rxnorm_matched_name",
                "rxnorm_ingredients",
                "rxnorm_ingredient_rxcuis",
            ]
        )
        for (category, term, raw_term), count in sorted(terms.items()):
            entry = cache_entry_for(term, rxnorm_cache)
            status = entry.get("status")
            if not status and category in {"immunotherapy", "irae_treatment"} and rxnorm_cache is not None:
                status = "not_in_cache"
            writer.writerow(
                [
                    category,
                    term,
                    raw_term,
                    count,
                    status,
                    entry.get("match_method"),
                    entry.get("matched_name"),
                    normalized_term(entry),
                    ingredient_rxcuis(entry),
                ]
            )


def read_json(path):
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export unique terms for mapping file review.")
    parser.add_argument("--input", default="data/patient_events.jsonl")
    parser.add_argument("--output", default="data/mapping_terms_review.csv")
    parser.add_argument("--rxnorm-cache", default=None)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    terms = collect_terms(records)
    rxnorm_cache = read_json(Path(args.rxnorm_cache)) if args.rxnorm_cache else None
    write_csv(terms, Path(args.output), rxnorm_cache=rxnorm_cache)
    print(f"Wrote {len(terms)} unique terms to {args.output}")
