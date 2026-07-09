import argparse
import json
from pathlib import Path

import ollama

IMMUNOTHERAPY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_mapped": {"type": "boolean"},
        "therapy_type": {"type": "string"},
        "ici_class": {"type": ["string", "null"]},
    },
    "required": ["is_mapped", "therapy_type", "ici_class"],
    "additionalProperties": False,
}


IRAE_TREATMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_mapped": {"type": "boolean"},
        "irae_treatment_type": {"type": "string"},
    },
    "required": ["is_mapped", "irae_treatment_type"],
    "additionalProperties": False,
}


IMMUNOTHERAPY_PROMPT = """
You are classifying RxNorm ingredient names.

Return JSON only according to the schema.

Classify the ingredient if it is a cancer-directed therapy.

Allowed therapy_type values:
- ICI
- targeted therapy
- chemotherapy
- ADC

Allowed ici_class values, only when therapy_type is ICI:
- PD-1_inhibitor
- PD-L1_inhibitor
- CTLA-4_inhibitor
- LAG-3_inhibitor

Rules:
- Set is_mapped true if the ingredient is a cancer-directed antineoplastic therapy.
- Use therapy_type ICI only for immune checkpoint inhibitors.
- Use targeted therapy for kinase inhibitors, tumor-targeting antibodies, VEGF/EGFR/BRAF/MEK/ALK/RET/MET inhibitors, and similar targeted cancer drugs.
- Use chemotherapy for conventional cytotoxic chemotherapy.
- Use ADC for antibody-drug conjugates.
- If the ingredient is not cancer-directed or is too ambiguous, set is_mapped false.
- If is_mapped is false, set therapy_type to "" and ici_class to null.
- If therapy_type is not ICI, ici_class must be null.
"""


IRAE_TREATMENT_PROMPT = """
You are classifying RxNorm ingredient names.

Return JSON only according to the schema.

Classify the ingredient if it is a medication used to treat or manage immune-related adverse events.

Allowed irae_treatment_type values:
- Corticosteroids
- Non-steroid immunosuppressants/immunomodulators
- Endocrine/metabolic replacement or regulation

Rules:
- Set is_mapped true if the ingredient belongs in one of the allowed irAE treatment categories.
- Corticosteroids include systemic, topical, ophthalmic, inhaled, or local corticosteroids.
- Non-steroid immunosuppressants/immunomodulators include infliximab, vedolizumab, mycophenolate, methotrexate, tacrolimus, cyclosporine, abatacept, tocilizumab, anakinra, IVIG, and similar immune-modulating agents.
- Endocrine/metabolic replacement or regulation includes levothyroxine, insulin, metformin, testosterone, progesterone, adrenal replacement, and similar agents for immune endocrinopathies or metabolic irAEs.
- If the ingredient is supportive care, antibiotic, antiemetic, analgesic, prophylaxis, supplement, OTC supportive care, or too nonspecific, set is_mapped false.
- If is_mapped is false, set irae_treatment_type to "".
"""


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path):
    records = []
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as f:
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


def normalized_name(value):
    return str(value or "").strip().lower()


def display_name(value):
    value = str(value or "").strip()
    if not value:
        return ""
    return value[0].upper() + value[1:]


def split_combo(value):
    return [part.strip() for part in str(value or "").split("+") if part.strip()]


def accepted_rxnorm_entry(term, rxnorm_cache):
    return rxnorm_cache.get(normalized_name(term))



def collect_missing_ingredient_contexts(records, rxnorm_cache, ingredient_class_map):
    mapped = {normalized_name(key) for key in ingredient_class_map}
    contexts = {}

    for record in records:
        condition_type = record.get("condition_type")
        if condition_type not in {"immunotherapy", "irae_treatment"}:
            continue

        for part in split_combo(record.get("condition")):
            entry = accepted_rxnorm_entry(part, rxnorm_cache)
            if not entry or entry.get("status") != "accepted":
                continue

            for ingredient in entry.get("ingredients") or []:
                key = normalized_name(ingredient.get("name"))
                if not key or key in mapped:
                    continue

                contexts.setdefault(
                    key,
                    {
                        "name": ingredient.get("name"),
                        "condition_types": set(),
                    },
                )
                contexts[key]["condition_types"].add(condition_type)

    return dict(sorted(contexts.items()))


def ask_llm(model, temperature, schema, prompt, ingredient_name):
    response = ollama.chat(
        model=model,
        format=schema,
        options={"temperature": temperature},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"RxNorm ingredient name: {ingredient_name}"},
        ],
    )
    return json.loads(response["message"]["content"])


def clean_immunotherapy_entry(result, ingredient_name):
    if not result.get("is_mapped"):
        return None

    therapy_type = result.get("therapy_type")
    if therapy_type not in {"ICI", "targeted therapy", "chemotherapy", "ADC"}:
        return None

    name = display_name(ingredient_name)
    if not name:
        return None

    entry = {
        "name": name,
        "therapy_type": therapy_type,
    }

    if therapy_type == "ICI":
        ici_class = result.get("ici_class")
        if ici_class not in {
            "PD-1_inhibitor",
            "PD-L1_inhibitor",
            "CTLA-4_inhibitor",
            "LAG-3_inhibitor",
        }:
            return None
        entry["ici_class"] = ici_class

    return entry


def clean_irae_treatment_entry(result, ingredient_name):
    if not result.get("is_mapped"):
        return None

    irae_treatment_type = result.get("irae_treatment_type")
    if irae_treatment_type not in {
        "Corticosteroids",
        "Non-steroid immunosuppressants/immunomodulators",
        "Endocrine/metabolic replacement or regulation",
    }:
        return None

    name = display_name(ingredient_name)
    if not name:
        return None

    return {
        "name": name,
        "irae_treatment_type": irae_treatment_type,
    }


def suggest_entry(context, model, temperature):
    condition_types = context["condition_types"]
    ingredient_name = context["name"]

    if "immunotherapy" in condition_types:
        result = ask_llm(
            model=model,
            temperature=temperature,
            schema=IMMUNOTHERAPY_SCHEMA,
            prompt=IMMUNOTHERAPY_PROMPT,
            ingredient_name=ingredient_name,
        )
        entry = clean_immunotherapy_entry(result, ingredient_name)
        if entry is not None:
            return entry

    if "irae_treatment" in condition_types:
        result = ask_llm(
            model=model,
            temperature=temperature,
            schema=IRAE_TREATMENT_SCHEMA,
            prompt=IRAE_TREATMENT_PROMPT,
            ingredient_name=ingredient_name,
        )
        entry = clean_irae_treatment_entry(result, ingredient_name)
        if entry is not None:
            return entry

    return None


def build_suggestions(contexts, existing_output, model, temperature):
    suggestions = dict(existing_output)
    total = len(contexts)

    for index, (key, context) in enumerate(contexts.items(), start=1):
        if key in suggestions:
            print(f"[{index}/{total}] Cached suggestion: {key}", flush=True)
            continue

        print(f"[{index}/{total}] Classifying ingredient: {context['name']}", flush=True)
        entry = suggest_entry(
            context=context,
            model=model,
            temperature=temperature,
        )

        if entry is None:
            print(f"  No map-ready category: {context['name']}", flush=True)
            continue

        suggestions[key] = entry

    return dict(sorted(suggestions.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Suggest map-ready ingredient_class_map entries for unmapped RxNorm ingredients."
    )
    parser.add_argument("--input", default="data/patient_events.jsonl")
    parser.add_argument("--rxnorm-cache", default="data/treatment_terms/rxnorm_cache.json")
    parser.add_argument("--ingredient-class-map", default="data/treatment_terms/ingredient_class_map.json")
    parser.add_argument("--output", default="data/treatment_terms/ingredient_class_suggestions.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    rxnorm_cache = read_json(args.rxnorm_cache)
    ingredient_class_map = read_json(args.ingredient_class_map)
    existing_output = read_json(args.output)

    contexts = collect_missing_ingredient_contexts(
        records=records,
        rxnorm_cache=rxnorm_cache,
        ingredient_class_map=ingredient_class_map,
    )
    print(f"Found {len(contexts)} unmapped accepted RxNorm ingredients.")

    suggestions = build_suggestions(
        contexts=contexts,
        existing_output=existing_output,
        model=args.model,
        temperature=args.temperature,
    )

    write_json(suggestions, args.output)
    print(f"Wrote {len(suggestions)} map-ready suggestions to {args.output}")
