import argparse
import json
from pathlib import Path

from rxnorm_mapping import cache_key, resolve_term, split_combo


TREATMENT_CATEGORIES = {"immunotherapy", "irae_treatment"}


def read_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


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


def read_terms(path):
    terms = set()
    for record in read_jsonl(path):
        if record.get("condition_type") not in TREATMENT_CATEGORIES:
            continue
        for part in split_combo(record.get("condition")):
            terms.add(part)
    return sorted(terms, key=str.lower)


def normalized_custom_map(raw_map):
    return {
        str(key).strip().lower(): value
        for key, value in (raw_map or {}).items()
        if str(key).strip()
    }


def custom_mapped_term(term, custom_map):
    mapped = custom_map.get(cache_key(term))
    if mapped is None:
        return term
    return str(mapped).strip()


def combined_resolution(raw_term, lookup_term, resolutions):
    accepted = []
    unresolved = []
    ingredients = []
    seen_ingredients = set()

    for resolution in resolutions:
        if resolution.get("status") == "accepted":
            accepted.append(resolution)
            for ingredient in resolution.get("ingredients") or []:
                key = ingredient.get("rxcui") or str(ingredient.get("name", "")).lower()
                if not key or key in seen_ingredients:
                    continue
                seen_ingredients.add(key)
                ingredients.append(ingredient)
        else:
            unresolved.append(resolution)

    if not accepted:
        return {
            "status": "unresolved",
            "raw_term": raw_term,
            "custom_mapped_term": lookup_term,
            "match_method": "custom_combo_no_accepted_parts",
            "matched_rxcui": None,
            "matched_name": None,
            "matched_tty": None,
            "match_score": None,
            "llm_reason": None,
            "ingredients": [],
            "component_resolutions": resolutions,
        }

    return {
        "status": "accepted",
        "raw_term": raw_term,
        "custom_mapped_term": lookup_term,
        "match_method": "custom_combo",
        "matched_rxcui": " + ".join(
            str(resolution.get("matched_rxcui"))
            for resolution in accepted
            if resolution.get("matched_rxcui")
        ) or None,
        "matched_name": " + ".join(
            str(resolution.get("matched_name"))
            for resolution in accepted
            if resolution.get("matched_name")
        ) or None,
        "matched_tty": None,
        "match_score": None,
        "llm_reason": None,
        "ingredients": ingredients,
        "component_resolutions": resolutions,
        "unresolved_components": unresolved,
    }


def resolve_lookup_term(raw_term, lookup_term, model, temperature, max_approximate, llm_timeout, rxnorm_timeout):
    lookup_parts = split_combo(lookup_term)
    if len(lookup_parts) <= 1:
        resolution = resolve_term(
            lookup_term,
            model=model,
            temperature=temperature,
            max_approximate=max_approximate,
            llm_timeout=llm_timeout,
            rxnorm_timeout=rxnorm_timeout,
        )
        resolution["raw_term"] = raw_term
        if lookup_term != raw_term:
            resolution["custom_mapped_term"] = lookup_term
        return resolution

    resolutions = []
    for part in lookup_parts:
        print(f"  Resolving custom-map component: {part}", flush=True)
        resolutions.append(
            resolve_term(
                part,
                model=model,
                temperature=temperature,
                max_approximate=max_approximate,
                llm_timeout=llm_timeout,
                rxnorm_timeout=rxnorm_timeout,
            )
        )
    return combined_resolution(raw_term, lookup_term, resolutions)


def build_cache(
    terms,
    cache,
    custom_map,
    model,
    temperature,
    max_approximate,
    llm_timeout,
    rxnorm_timeout,
    refresh=False,
    cache_path=None,
):
    total = len(terms)
    for index, term in enumerate(terms, start=1):
        key = cache_key(term)
        if not refresh and key in cache:
            print(f"[{index}/{total}] Cached treatment term: {term}", flush=True)
            continue

        lookup_term = custom_mapped_term(term, custom_map)
        if lookup_term != term:
            print(
                f"[{index}/{total}] Resolving treatment term: {term} -> {lookup_term}",
                flush=True,
            )
        else:
            print(f"[{index}/{total}] Resolving treatment term: {term}", flush=True)
        cache[key] = resolve_lookup_term(
            raw_term=term,
            lookup_term=lookup_term,
            model=model,
            temperature=temperature,
            max_approximate=max_approximate,
            llm_timeout=llm_timeout,
            rxnorm_timeout=rxnorm_timeout,
        )
        if cache_path is not None:
            write_json(cache, cache_path)
    return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an RxNorm treatment cache from raw event JSONL."
    )
    parser.add_argument("--input", default="data/patient_events.jsonl")
    parser.add_argument("--cache", default="data/treatment_terms/rxnorm_cache.json")
    parser.add_argument("--custom-map", default="data/treatment_terms/custom_map.json")
    parser.add_argument("--model", required=True, help="Ollama model name for approximate-match validation.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60,
        help="Seconds to wait for each Ollama approximate-match validation.",
    )
    parser.add_argument(
        "--rxnorm-timeout",
        type=float,
        default=20,
        help="Seconds to wait for each RxNorm HTTP request.",
    )
    parser.add_argument(
        "--max-approximate",
        type=int,
        default=2,
        help="Number of RxNorm approximate candidates to ask the LLM about. Default is the top two matches.",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache = read_json(cache_path)
    custom_map = normalized_custom_map(read_json(Path(args.custom_map)) if args.custom_map else {})
    terms = read_terms(Path(args.input))
    print(f"Found {len(terms)} unique treatment terms.", flush=True)
    cache = build_cache(
        terms=terms,
        cache=cache,
        custom_map=custom_map,
        model=args.model,
        temperature=args.temperature,
        max_approximate=args.max_approximate,
        llm_timeout=args.llm_timeout,
        rxnorm_timeout=args.rxnorm_timeout,
        refresh=args.refresh,
        cache_path=cache_path,
    )
    write_json(cache, cache_path)

    accepted = sum(1 for item in cache.values() if item.get("status") == "accepted")
    unresolved = sum(1 for item in cache.values() if item.get("status") != "accepted")
    print(f"Wrote {len(cache)} cache entries to {cache_path}")
    print(f"Accepted: {accepted}; unresolved: {unresolved}")
