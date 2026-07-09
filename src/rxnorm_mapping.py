import json
from urllib.parse import urlencode
from urllib.request import urlopen

import ollama


RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

RXNORM_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "is_match": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_match", "reason"],
    "additionalProperties": False,
}

LLM_VALIDATION_PROMPT = """
You are validating RxNorm drug-name matches.

Return JSON only.

Decide whether the RxNorm candidate refers to the same medication as the extracted clinical term.

Accept if:
- it is the same generic ingredient
- it is a brand/trade name for the same drug
- it differs only by spelling, capitalization, dose, route, formulation, or abbreviation
- the extracted term is a common shorthand for the candidate
- it is generally ok if the dose or formulation is different as long as the active ingredient(s) match

Reject if:
- it is a different drug
- it is only the same drug class
- it is only an indication, toxicity, procedure, or treatment category
- the candidate adds a different active ingredient that is not implied by the extracted term
- the extracted term is too vague to identify the medication

Be conservative.
"""

def cache_key(term):
    return str(term or "").strip().lower()


def split_combo(value):
    return [part.strip() for part in str(value).split("+") if part.strip()]


def rxnorm_get(path, params=None, timeout=20):
    query = f"?{urlencode(params or {})}" if params else ""
    with urlopen(f"{RXNORM_BASE_URL}{path}{query}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def exact_rxcuis(term, rxnorm_timeout=20):
    data = rxnorm_get(
        "/rxcui.json",
        {"name": term, "search": "0", "allsrc": "0"},
        timeout=rxnorm_timeout,
    )
    return data.get("idGroup", {}).get("rxnormId", []) or []


def approximate_candidates(term, max_entries=2, rxnorm_timeout=20):
    data = rxnorm_get(
        "/approximateTerm.json",
        {"term": term, "maxEntries": str(max_entries), "option": "1"},
        timeout=rxnorm_timeout,
    )
    candidates = data.get("approximateGroup", {}).get("candidate", []) or []
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            int(candidate.get("rank") or 999),
            -float(candidate.get("score") or 0),
        ),
    )
    unique_candidates = []
    seen_rxcuis = set()
    for candidate in sorted_candidates:
        rxcui = str(candidate.get("rxcui") or "").strip()
        if not rxcui or rxcui in seen_rxcuis:
            continue
        seen_rxcuis.add(rxcui)
        unique_candidates.append(candidate)
        if len(unique_candidates) >= max_entries:
            break
    return unique_candidates


def concept_properties(rxcui, rxnorm_timeout=20):
    data = rxnorm_get(f"/rxcui/{rxcui}/properties.json", timeout=rxnorm_timeout)
    return data.get("properties") or {}


def ingredient_concepts(rxcui, rxnorm_timeout=20):
    props = concept_properties(rxcui, rxnorm_timeout=rxnorm_timeout)
    if props.get("tty") == "IN":
        return [{"rxcui": str(props.get("rxcui")), "name": str(props.get("name"))}]

    data = rxnorm_get(
        f"/rxcui/{rxcui}/related.json",
        {"tty": "IN"},
        timeout=rxnorm_timeout,
    )
    ingredients = []
    for group in data.get("relatedGroup", {}).get("conceptGroup", []) or []:
        for concept in group.get("conceptProperties", []) or []:
            ingredients.append(
                {
                    "rxcui": str(concept.get("rxcui")),
                    "name": str(concept.get("name")),
                }
            )
    return unique_ingredients(ingredients)


def unique_ingredients(ingredients):
    seen = set()
    out = []
    for ingredient in ingredients:
        key = ingredient.get("rxcui") or str(ingredient.get("name", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ingredient)
    return out


def accepted_resolution(
    raw_term,
    rxcui,
    match_method,
    match_score=None,
    llm_reason=None,
    rxnorm_timeout=20,
):
    props = concept_properties(rxcui, rxnorm_timeout=rxnorm_timeout)
    ingredients = ingredient_concepts(rxcui, rxnorm_timeout=rxnorm_timeout)
    if not ingredients:
        return unresolved_resolution(
            raw_term,
            match_method=f"{match_method}_no_ingredients",
            rejected_candidates=[
                {
                    "rxcui": str(rxcui),
                    "name": props.get("name"),
                    "reason": "RxNorm match had no ingredient concepts.",
                }
            ],
        )

    return {
        "status": "accepted",
        "raw_term": raw_term,
        "match_method": match_method,
        "matched_rxcui": str(rxcui),
        "matched_name": props.get("name"),
        "matched_tty": props.get("tty"),
        "match_score": match_score,
        "llm_reason": llm_reason,
        "ingredients": ingredients,
    }


def unresolved_resolution(raw_term, match_method, rejected_candidates=None, error=None):
    payload = {
        "status": "unresolved",
        "raw_term": raw_term,
        "match_method": match_method,
        "matched_rxcui": None,
        "matched_name": None,
        "matched_tty": None,
        "match_score": None,
        "llm_reason": None,
        "ingredients": [],
        "rejected_candidates": rejected_candidates or [],
    }
    if error:
        payload["error"] = str(error)
    return payload


def validate_approximate_match(raw_term, candidate, ingredients, model, temperature, llm_timeout):
    candidate_name = candidate.get("name") or candidate.get("resolved_name")
    ingredient_lines = "\n".join(
        f"- {item.get('name')}, RxCUI {item.get('rxcui')}" for item in ingredients
    ) or "- none"
    user_prompt = (
        f"Extracted term: {raw_term}\n\n"
        f"RxNorm candidate name: {candidate_name}\n"
        f"RxNorm candidate RxCUI: {candidate.get('rxcui')}\n"
        f"RxNorm candidate ingredients:\n{ingredient_lines}\n\n"
        "Is this a good match?"
    )
    client = ollama.Client(timeout=llm_timeout)
    response = client.chat(
        model=model,
        format=RXNORM_MATCH_SCHEMA,
        options={"temperature": temperature},
        messages=[
            {"role": "system", "content": LLM_VALIDATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def resolve_term(
    raw_term,
    model=None,
    temperature=0.0,
    max_approximate=2,
    llm_timeout=60,
    rxnorm_timeout=20,
):
    term = str(raw_term or "").strip()
    if not term:
        return unresolved_resolution(raw_term, "blank")

    try:
        print(f"  RxNorm exact lookup: {term}", flush=True)
        exact = exact_rxcuis(term, rxnorm_timeout=rxnorm_timeout)
        if exact:
            print(f"  RxNorm exact match: {exact[0]}", flush=True)
            return accepted_resolution(
                term,
                exact[0],
                "exact",
                rxnorm_timeout=rxnorm_timeout,
            )

        if not model:
            return unresolved_resolution(term, "no_exact_no_llm")

        rejected = []
        print(f"  RxNorm approximate lookup: {term}", flush=True)
        candidates = approximate_candidates(
            term,
            max_entries=max_approximate,
            rxnorm_timeout=rxnorm_timeout,
        )
        if not candidates:
            return unresolved_resolution(term, "no_rxnorm_match")

        for candidate in candidates:
            rxcui = str(candidate.get("rxcui"))
            props = concept_properties(rxcui, rxnorm_timeout=rxnorm_timeout)
            candidate_name = candidate.get("name") or props.get("name")
            candidate["resolved_name"] = candidate_name
            print(f"  RxNorm ingredient lookup: {candidate_name} ({rxcui})", flush=True)
            ingredients = ingredient_concepts(rxcui, rxnorm_timeout=rxnorm_timeout)
            if not candidate_name and not ingredients:
                rejected.append(
                    {
                        "rxcui": rxcui,
                        "name": None,
                        "score": candidate.get("score"),
                        "rank": candidate.get("rank"),
                        "reason": "RxNorm approximate candidate had no name or ingredient concepts.",
                    }
                )
                continue

            print(f"  LLM validating approximate match: {candidate_name}", flush=True)
            decision = validate_approximate_match(
                raw_term=term,
                candidate=candidate,
                ingredients=ingredients,
                model=model,
                temperature=temperature,
                llm_timeout=llm_timeout,
            )
            if decision.get("is_match"):
                return accepted_resolution(
                    term,
                    rxcui,
                    "approximate_llm_validated",
                    match_score=candidate.get("score"),
                    llm_reason=decision.get("reason"),
                    rxnorm_timeout=rxnorm_timeout,
                )

            rejected.append(
                {
                    "rxcui": rxcui,
                    "name": candidate_name,
                    "score": candidate.get("score"),
                    "rank": candidate.get("rank"),
                    "reason": decision.get("reason"),
                }
            )

        return unresolved_resolution(
            term,
            "approximate_llm_rejected",
            rejected_candidates=rejected,
        )
    except Exception as exc:
        return unresolved_resolution(term, "rxnorm_error", error=exc)
