import argparse
import json
from pathlib import Path

import ollama
from irae_mapping import map_irae_events
from oncotree_mapping import classify_oncotree


class EventJSONError(Exception):
    def __init__(self, message, content):
        super().__init__(message)
        self.content = content


EVENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition_type": {
                        "type": "string",
                        "enum": ["immunotherapy", "irae", "irae_treatment"],
                    },
                    "condition": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    }
                },
                "required": ["condition_type", "condition", "start_date", "end_date"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}


EVENTS_PROMPT = """
    You are extracting structured oncology timeline data from clinical notes.

    Return JSON according to the schema.

    Rules:
    1. Include all documented immunotherapy treatments, iraes (immune-related adverse events), and irae-directed treatments.
    2. `immunotherapy` includes **ONLY** checkpoint inhibitors or other immunotherapy (e.g., pembrolizumab, nivolumab, atezolizumab), not chemotherapy, targeted therapy, radiation, or other treatments.
    3. `irae_treatment` includes treatments used to manage irAEs, such as corticosteroids (e.g., prednisone), methylprednisolone, and other immunosuppressive agents (e.g., infliximab) used for irAE management. Do not include treatments used for other purposes (e.g., chemotherapy, targeted therapy, prophylaxis).
    4. `irae` includes **immune-related toxicities/events only**.
    5. If a date range is implied, set `start_date` and `end_date`.
    6. If only onset is known, set `start_date` only and `end_date` to null.
    7. Preserve date precision from source text (day > month > year).
    8. Format all dates using hyphens only: YYYY-MM-DD, YYYY-MM, or YYYY. Convert slash dates like 03/19/2018 to 2018-03-19.
    9. Important: each event must have its own entry in the JSON array. Do not combine multiple events into one entry.
    10. Use concise names without extranneous detail (e.g., "Atezolizumab", "Hepatitis").
    11. Use single, generic drug names with no qualifiers or extra information (e.g. "Prednisone", not "Prednisone 10mg daily").
    12. Use irAE names following ASCO terminology when possible (e.g., "Colitis", "Pneumonitis", "Hepatitis", "Rash", "Adrenal Insufficiency").
    13. If the therapy is a combination, create two separate entries with the same date for each drug (e.g., "Nivolumab" and "Ipilimumab", not "Nivolumab + Ipilimumab").
    """


def parse_json_object(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start:end + 1])


def extract_events(model, temperature, note):
    """Extract structured events from a clinical note using the LLM."""
    messages = [
        {'role': 'system', 'content': EVENTS_PROMPT},
        {'role': 'user', 'content': note},
    ]
    response = ollama.chat(
        model=model,
        format=EVENTS_SCHEMA,
        options={'temperature': temperature},
        messages=messages,
    )
    content = response['message']['content']
    print(f"Raw LLM response content preview: {content[:120]}...")
    try:
        data = parse_json_object(content)
    except json.JSONDecodeError:
        print("Could not parse event JSON. Retrying once.")
        response = ollama.chat(
            model=model,
            format=EVENTS_SCHEMA,
            options={'temperature': 0.0},
            messages=messages + [
                {
                    'role': 'user',
                    'content': 'Your previous response was invalid JSON. Return the complete JSON object again, with no extra text.',
                }
            ],
        )
        content = response['message']['content']
        print(f"Retry LLM response content preview: {content[:120]}...")
        try:
            data = parse_json_object(content)
        except json.JSONDecodeError as e:
            raise EventJSONError(str(e), content) from e

    events = data.get('events', []) if isinstance(data, dict) else []
    out = []

    for e in events:
        ctype = e.get('condition_type', '').strip().lower()
        condition = e.get('condition', '').strip().lower()
        start = e.get('start_date', '')
        end = e.get('end_date', None)

        start_clean = str(start).strip()
        end_clean = None if end is None else str(end).strip()
        if end_clean and end_clean.lower() in {"none", "null", "na", "n/a", ""}:
            end_clean = None

        out.append(
            {
                'condition_type': ctype,
                'condition': condition,
                'start_date': start_clean,
                'end_date': end_clean,
            }
        )

    return out


def has_valid_condition(events, condition_type):
    return any(
        event.get("condition_type") == condition_type
        and str(event.get("condition", "")).strip().lower() not in {"", "unknown", "none", "null", "na", "n/a"}
        for event in events
    )


def cohort_exclusion_reason(events):
    if not has_valid_condition(events, "irae"):
        return "no_irae"
    if not has_valid_condition(events, "immunotherapy"):
        return "no_valid_immunotherapy"
    return None


def append_skip_log(log_path, record):
    if log_path is None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def existing_source_files(output_path):
    """Return source files already present in an existing JSONL output."""
    if not output_path.exists():
        return set()

    sources = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_file = record.get("source_file")
            if source_file:
                sources.add(source_file)
    return sources


def next_patient_count(output_path):
    """Return the next numeric patient index for appending to an existing JSONL."""
    if not output_path.exists():
        return 1

    max_id = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            patient_id = str(record.get("patient_id", ""))
            if not patient_id.startswith("patient_"):
                continue
            try:
                max_id = max(max_id, int(patient_id.removeprefix("patient_")))
            except ValueError:
                continue

    return max_id + 1


def parse_patient_files(
    model,
    temperature,
    input_dir,
    pattern,
    tissue_list,
    oncotree_base,
    irae_names,
    irae_map,
    output_path,
    skip_log_path,
    patient_count,
    skip_files=None,
):
    """Parse files and append each result to output_path as soon as it completes."""
    files = sorted(input_dir.glob(pattern))
    skip_files = skip_files or set()
    files_to_process = [path for path in files if path.name not in skip_files]
    total_events = 0

    for index, path in enumerate(files_to_process, start=1):
        try:
            note = path.read_text(encoding="utf-8", errors="ignore")
            events = extract_events(model=model, temperature=temperature, note=note)
            parsed = {
                "events": map_irae_events(
                    events=events,
                    model=model,
                    temperature=temperature,
                    irae_names_path=irae_names,
                    irae_map_path=irae_map,
                ),
                "oncotree": classify_oncotree(
                    note=note,
                    model=model,
                    temperature=temperature,
                    tissue_list_path=tissue_list,
                    data_base_path=oncotree_base,
                ),
            }
            reason = cohort_exclusion_reason(parsed["events"])
            if reason:
                append_skip_log(
                    skip_log_path,
                    {
                        "source_file": path.name,
                        "candidate_patient_id": f"patient_{patient_count}",
                        "reason": reason,
                        "event_count": len(parsed["events"]),
                    },
                )
                print(f"Skipped {index} of {len(files_to_process)} notes: {path.name}: {reason}")
                continue

            total_events += append_events_jsonl(
                source_file=path.name,
                parsed=parsed,
                output_path=output_path,
                patient_id=f"patient_{patient_count}",
            )
            patient_count += 1
            print(f"Processed {index} of {len(files_to_process)} notes")
        except EventJSONError as e:
            append_skip_log(
                skip_log_path,
                {
                    "source_file": path.name,
                    "candidate_patient_id": f"patient_{patient_count}",
                    "reason": "invalid_event_json",
                    "error": str(e),
                    "llm_response": e.content,
                },
            )
            print(f"Failed {index} of {len(files_to_process)} notes: {path.name}: invalid_event_json")
        except Exception as e:
            append_skip_log(
                skip_log_path,
                {
                    "source_file": path.name,
                    "candidate_patient_id": f"patient_{patient_count}",
                    "reason": "processing_error",
                    "error": str(e),
                },
            )
            print(f"Failed {index} of {len(files_to_process)} notes: {path.name}: {e}")
    return files, len(files_to_process), total_events


def append_events_jsonl(source_file, parsed, output_path, patient_id):
    """Append one parsed file's event rows to the output JSONL."""
    total_events = 0
    with output_path.open("a", encoding="utf-8") as f:
        events = parsed.get("events", [])
        oncotree = parsed.get("oncotree", {})
        for event in events:
            record = {
                "patient_id": patient_id,
                "source_file": source_file,
                **oncotree,
                **event,
            }
            f.write(json.dumps(record) + "\n")
            total_events += 1

    return total_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-parse patient notes with Ollama.")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g., llama3.1:8b).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--input-dir", default="data/patient_notes", help="Directory containing patient note .txt files.")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern for note files.")
    parser.add_argument("--output", default="data/patient_events.jsonl", help="Path to output JSONL file.")
    parser.add_argument("--skip-log", default="data/patient_events_skipped.jsonl", help="Path to skipped-note JSONL log.")
    parser.add_argument("--tissue-list", default="data/oncotree_tissues/tissue_types.txt")
    parser.add_argument("--oncotree-base", default="data/oncotree_tissues")
    parser.add_argument("--irae-names", default="data/irae_terms/irae_names.txt")
    parser.add_argument("--irae-map", default="data/irae_terms/irae_map.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    skip_log_path = Path(args.skip_log) if args.skip_log else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skip_files = existing_source_files(output_path)
    patient_count = next_patient_count(output_path)

    files, processed_count, total_events = parse_patient_files(
        model=args.model,
        temperature=args.temperature,
        input_dir=input_dir,
        pattern=args.pattern,
        tissue_list=args.tissue_list,
        oncotree_base=args.oncotree_base,
        irae_names=args.irae_names,
        irae_map=args.irae_map,
        output_path=output_path,
        skip_log_path=skip_log_path,
        patient_count=patient_count,
        skip_files=skip_files,
    )

    print(f"Parsed {processed_count} files and wrote {total_events} events to {output_path}")
