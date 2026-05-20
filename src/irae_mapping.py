import json
from pathlib import Path

import ollama


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
    },
    "required": ["value"],
    "additionalProperties": False,
}


def parse_lines_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_json_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_response(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("value", "")

    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"', "`"}:
        value = value[1:-1].strip()
    return value


def canonical_match(value, valid_values):
    lookup = {valid_value.lower(): valid_value for valid_value in valid_values}
    return lookup.get(str(value).strip().lower(), "Unknown")


def generate_mapping_response(model, temperature, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options = {"temperature": temperature}

    
    response = ollama.chat(
        model=model,
        format=OUTPUT_SCHEMA,
        options=options,
        messages=messages,
    )

    raw_content = response["message"]["content"]
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = raw_content
    return clean_response(parsed)


def create_irae_name_prompt(irae_names):
    return (
        "You are mapping extracted immune-related adverse event terms to a "
        "canonical irAE terminology list.\n\n"
        "Rules:\n"
        "- Choose only from the provided list.\n"
        "- Return a JSON object with one key, `value`.\n"
        "- Set `value` to the best matching irAE name exactly as it appears in the list.\n"
        "- If there is no good match, set `value` to: Unknown.\n\n"
        "Accepted irAE names:\n"
        + "\n".join(irae_names)
    )


def map_irae_name(raw_irae, model, temperature, irae_names_path):
    irae_names = parse_lines_file(irae_names_path)
    direct_match = canonical_match(raw_irae, irae_names)
    if direct_match != "Unknown":
        return direct_match

    mapped_name = generate_mapping_response(
        model=model,
        temperature=temperature,
        system_prompt=create_irae_name_prompt(irae_names),
        user_prompt=f"Extracted irAE term: {raw_irae}",
    )
    return canonical_match(mapped_name, irae_names)


def map_irae_type(irae_name, irae_map_path):
    if not irae_name or irae_name == "Unknown":
        return "Unknown"

    irae_map = load_json_file(irae_map_path)
    canonical_name = canonical_match(irae_name, irae_map.keys())
    if canonical_name == "Unknown":
        return "Unknown"
    return irae_map.get(canonical_name, "Unknown")


def map_irae_events(events, model, temperature, irae_names_path, irae_map_path):
    mapped = []

    for event in events:
        if event.get("condition_type") != "irae":
            mapped.append(event)
            continue

        raw_condition = event.get("condition") or ""
        irae_name = map_irae_name(
            raw_irae=raw_condition,
            model=model,
            temperature=temperature,
            irae_names_path=irae_names_path,
        )
        irae_type = map_irae_type(
            irae_name=irae_name,
            irae_map_path=irae_map_path,
        )
        mapped.append(
            {
                **event,
                "raw_condition": raw_condition,
                "condition": irae_name,
                "irae_type": irae_type,
            }
        )

    return mapped
