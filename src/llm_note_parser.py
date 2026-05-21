import argparse
import json
from pathlib import Path

import ollama
from irae_mapping import map_irae_events
from oncotree_mapping import classify_oncotree


class EventJSONError(Exception):
    def __init__(self, message, content, event_type):
        super().__init__(message)
        self.content = content
        self.event_type = event_type


SINGLE_TYPE_EVENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "start_date": {"type": "string"}
                },
                "required": ["condition", "start_date"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}


IMMUNOTHERAPY_PROMPT = """
You are extracting cancer immunotherapy regimen timeline events from clinical notes.

Return JSON according to the schema. Return no explanations.

Extract ONLY cancer immunotherapy regimen START/RESTART/CHANGE events.
Do **NOT** extract every cycle, dose, infusion, administration, continuation, maintenance mention, follow-up mention, or medication adjustment.

Include:
- immune checkpoint inhibitors
- other cancer immunotherapies
- immunotherapy combinations

Do NOT include:
- chemotherapy
- targeted therapy
- radiation
- surgery
- supportive care
- steroids or immunosuppressants used to treat irAEs
- prophylaxis
- non-cancer treatments

New event rules:
1. Create an event when a new immunotherapy regimen starts.
2. Create an event when an immunotherapy regimen changes.
3. Create an event when an immunotherapy drug is added or removed.
4. Create an event when immunotherapy restarts after a documented pause or discontinuation.
5. IMPORTANT: Do **NOT** create new events for repeated cycles/doses/infusions/mentions of the same regimen.
6. Do **NOT** create events for continued therapy, maintenance mention, or planned next cycle unless the regimen changed.

Combination rules:
1. Combination is allowed ONLY for immunotherapy drugs.
2. If multiple immunotherapy drugs start as one regimen, write one condition with drugs joined by " + ".
3. If chemotherapy plus immunotherapy is listed, include ONLY the immunotherapy drug(s).
4. Never include chemotherapy or targeted therapy inside the condition.

Examples:
- "carboplatin/pemetrexed/pembrolizumab" -> "Pembrolizumab"
- "nivolumab plus ipilimumab" -> "Ipilimumab + Nivolumab"
- "ipi/nivo followed by nivolumab maintenance" -> two events:
  1. "Ipilimumab + Nivolumab" at combination start
  2. "Nivolumab" at maintenance monotherapy start
- repeated "C2 nivolumab", "C3 nivolumab", "C4 nivolumab" -> only one "Nivolumab" event at first start date

IMPORTANT Date rules:
- Extract only start/restart/change dates.
- Preserve source precision: YYYY-MM-DD, YYYY-MM, or YYYY.
- Convert slash dates to hyphenated dates.
- If an exact date is not stated but timing is clearly implied, estimate the most specific reasonable date from context.
- Interpret "early YEAR" as YEAR-01, "mid YEAR" as YEAR-06, and "late YEAR" as YEAR-12.
- Examples: March 2020 -> 2020-03, March 10, 2022 -> 2022-03-10, "in 2021" -> 2021, "started in early 2020" -> 2020-01, "10/05/2023" -> 2023-10-05

Condition rules:
- Use concise generic drug names (e.g. "Atezolizumab", "Nivolumab + Ipilimumab").
- No dose, route, frequency, cycle number, or other explanatory text.
"""

IRAE_PROMPT = """
You are extracting immune-related adverse event onset episodes from clinical notes.

Return JSON according to the schema. Return no explanations.

Extract ONLY immune-related adverse event START/ONSET events.

Do NOT extract:
- immunotherapy drugs
- chemotherapy or cancer treatments
- treatments used to manage irAEs
- symptoms that are clearly unrelated to immunotherapy
- cancer progression
- infections unless explicitly described as immune-related
- lab abnormalities unless they represent a named immune-related diagnosis

New event rules:
1. Create an event when an irAE first starts.
2. Create a new event only if the irAE clearly resolved and later recurred/restarted.
3. Do NOT create repeated events for follow-up mentions of the same ongoing irAE.
4. Do NOT create repeated events for persistent, improving, worsening, stable, or monitored toxicity unless a new episode starts.
5. Do NOT create a new event only because severity grade changed.

Combination rules:
1. Never combine multiple irAEs in one condition - you should create a separate event for each distinct irAE.
2. If multiple irAEs start on the same date or are mentioned together, create separate events.

Examples:
- "rash and colitis began on 3/1/2020" -> two events:
  1. "Rash", "2020-03-01"
  2. "Colitis", "2020-03-01"
- "rash persisted over several visits" -> one Rash event at onset only
- "colitis resolved, then recurred in July" -> two Colitis events if both onset dates are documented
- "arthralgias attributed to nivolumab" -> "Arthralgia"
- "hypothyroidism from pembrolizumab" -> "Hypothyroidism"

IMPORTANT Date rules:
- Extract only onset/start date.
- Preserve source precision: YYYY-MM-DD, YYYY-MM, or YYYY.
- Convert slash dates to hyphenated dates.
- If an exact date is not stated but timing is clearly implied, estimate the most specific reasonable date from context.
- If an event occurred "shortly after" a dated treatment start, use the day after the treatment start date.
- Interpret "early YEAR" as YEAR-01, "mid YEAR" as YEAR-06, and "late YEAR" as YEAR-12.
- Examples: March 2020 -> 2020-03, March 10, 2022 -> 2022-03-10, "in 2021" -> 2021, "started in early 2020" -> 2020-01, "10/05/2023" -> 2023-10-05

Condition rules:
- Use concise clinical toxicity names.
- Prefer ASCO-style irAE terms when possible.
- No grade, dose, route, severity, or explanatory text.
"""


IRAE_TREATMENT_PROMPT = """
You are extracting treatments used to manage immune-related adverse events (irAEs) from clinical notes.

Return JSON according to the schema. Return no explanations.

Extract ONLY medications, procedures, or interventions that are explicitly being used to treat or manage an irAE.

IMPORTANT:
An irAE treatment is the treatment/intervention, NOT the adverse event itself.
Do NOT extract symptoms, toxicities, diagnoses, or indications as treatments.

Extract:
- Systemic corticosteroids: Prednisone, Methylprednisolone, Dexamethasone, Hydrocortisone
- Topical/local steroids: Triamcinolone, Clobetasol, Hydrocortisone cream
- Immunosuppressants: Mycophenolate, Tacrolimus, Cyclosporine, Methotrexate
- Biologics/targeted immune treatments for irAEs: Infliximab, Vedolizumab, Tocilizumab, Abatacept, Dupilumab, IVIG
- Hormone replacement for immune endocrinopathies: Levothyroxine, Hydrocortisone, Insulin
- Symptom-directed treatments clearly used for irAE management: Loperamide for immune diarrhea, antihistamines for immune rash/pruritus, artificial tears or ophthalmic steroids for ocular irAE
- Procedures clearly used for irAE management: Plasmapheresis, physical therapy, hydration if explicitly used for irAE management

Do NOT extract:
- The irAE itself: Rash, Itching, Pruritus, Colitis, Diarrhea, Pneumonitis, Hepatitis, Arthralgia, Hypothyroidism, Adrenal insufficiency
- Cancer-directed treatments: immunotherapy, chemotherapy, targeted therapy, radiation, surgery
- Prophylaxis unless explicitly described as part of irAE management
- Chronic home medications unless clearly started for an irAE
- Dose changes, tapers, refills, or continued treatment mentions

Event rules:
1. Create an event when a new irAE-directed treatment starts.
2. Create a new event only if the treatment restarts after being stopped.
3. Do not create events for dose changes or tapers.
4. Do not create events for continued use.
5. Each treatment must be a separate event. Never combine treatments with "+".

Condition field rules:
- The condition must be the treatment name only.
- Do not include the irAE, symptom, indication, dose, route, frequency, or taper instructions.
- Use concise generic treatment names.

IMPORTANT Date rules:
- Extract only treatment start/restart date.
- Preserve source precision: YYYY-MM-DD, YYYY-MM, or YYYY.
- Convert slash dates to hyphenated dates.
- If an exact date is not stated but timing is clearly implied, estimate the most specific reasonable date from context.
- Interpret "early YEAR" as YEAR-01, "mid YEAR" as YEAR-06, and "late YEAR" as YEAR-12.
- Examples: March 2020 -> 2020-03, March 10, 2022 -> 2022-03-10, "in 2021" -> 2021, "started in early 2020" -> 2020-01, "10/05/2023" -> 2023-10-05
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


def extract_single_type_events(model, temperature, note, prompt, condition_type):
    """Extract one event type from a clinical note using the LLM."""
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': note},
    ]
    response = ollama.chat(
        model=model,
        format=SINGLE_TYPE_EVENTS_SCHEMA,
        options={'temperature': temperature},
        messages=messages,
    )
    content = response['message']['content']
    print(f"{condition_type} LLM response content preview: {content[:120]}...")
    try:
        data = parse_json_object(content)
    except json.JSONDecodeError:
        print(f"Could not parse {condition_type} JSON. Retrying once.")
        response = ollama.chat(
            model=model,
            format=SINGLE_TYPE_EVENTS_SCHEMA,
            options={'temperature': 0.0},
            messages=messages + [
                {
                    'role': 'user',
                    'content': 'Your previous response was invalid JSON. Return the complete JSON object again, with no extra text.',
                }
            ],
        )
        content = response['message']['content']
        print(f"Retry {condition_type} LLM response content preview: {content[:120]}...")
        try:
            data = parse_json_object(content)
        except json.JSONDecodeError as e:
            raise EventJSONError(str(e), content, condition_type) from e

    events = data.get('events', []) if isinstance(data, dict) else []
    out = []

    for e in events:
        condition = str(e.get('condition', '')).strip()
        start = e.get('start_date', '')
        start_clean = str(start).strip()
        if not condition or not start_clean:
            continue

        out.append(
            {
                'condition_type': condition_type,
                'condition': condition,
                'start_date': start_clean
            }
        )

    return out


def extract_events(model, temperature, note):
    """Extract structured events from a clinical note using separate LLM calls."""
    immunotherapy_events = extract_single_type_events(
        model=model,
        temperature=temperature,
        note=note,
        prompt=IMMUNOTHERAPY_PROMPT,
        condition_type="immunotherapy",
    )
    irae_events = extract_single_type_events(
        model=model,
        temperature=temperature,
        note=note,
        prompt=IRAE_PROMPT,
        condition_type="irae",
    )
    irae_treatment_events = extract_single_type_events(
        model=model,
        temperature=temperature,
        note=note,
        prompt=IRAE_TREATMENT_PROMPT,
        condition_type="irae_treatment",
    )
    print(
        "Extracted "
        f"{len(immunotherapy_events)} immunotherapy, "
        f"{len(irae_events)} irAE, "
        f"{len(irae_treatment_events)} irAE treatment events"
    )
    return immunotherapy_events + irae_events + irae_treatment_events


def has_valid_condition(events, condition_type):
    return any(
        event.get("condition_type") == condition_type
        and str(event.get("condition", "")).strip().lower() not in {"", "unknown", "none", "null", "na", "n/a"}
        for event in events
    )


def is_missing_value(value):
    return value is None or str(value).strip().lower() in {"", "unknown", "none", "null", "na", "n/a"}


def cohort_exclusion_reason(events):
    if not has_valid_condition(events, "irae"):
        return "no_irae"
    if not has_valid_condition(events, "immunotherapy"):
        return "no_valid_immunotherapy"
    return None


def extract_and_map_events(model, temperature, note, irae_names, irae_map):
    events = extract_events(model=model, temperature=temperature, note=note)
    return map_irae_events(
        events=events,
        model=model,
        temperature=temperature,
        irae_names_path=irae_names,
        irae_map_path=irae_map,
    )


def classify_oncotree_with_retry(note, model, temperature, tissue_list, oncotree_base):
    oncotree = classify_oncotree(
        note=note,
        model=model,
        temperature=temperature,
        tissue_list_path=tissue_list,
        data_base_path=oncotree_base,
    )
    if is_missing_value(oncotree.get("oncotree_tissue")):
        print("No OncoTree tissue mapped. Retrying OncoTree mapping once.")
        oncotree = classify_oncotree(
            note=note,
            model=model,
            temperature=temperature,
            tissue_list_path=tissue_list,
            data_base_path=oncotree_base,
        )
    return oncotree


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


def existing_logged_source_files(log_path):
    """Return source files already recorded in the skipped-note log."""
    if log_path is None or not log_path.exists():
        return set()

    sources = set()
    with log_path.open("r", encoding="utf-8") as f:
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
            events = extract_and_map_events(
                model=model,
                temperature=temperature,
                note=note,
                irae_names=irae_names,
                irae_map=irae_map,
            )
            reason = cohort_exclusion_reason(events)
            if reason:
                print(f"{path.name}: {reason}. Retrying event extraction once.")
                events = extract_and_map_events(
                    model=model,
                    temperature=temperature,
                    note=note,
                    irae_names=irae_names,
                    irae_map=irae_map,
                )
                reason = cohort_exclusion_reason(events)

            if reason:
                append_skip_log(
                    skip_log_path,
                    {
                        "source_file": path.name,
                        "candidate_patient_id": f"patient_{patient_count}",
                        "reason": reason,
                        "event_count": len(events),
                        "retried": True,
                    },
                )
                print(f"Skipped {index} of {len(files_to_process)} notes: {path.name}: {reason}")
                continue

            parsed = {
                "events": events,
                "oncotree": classify_oncotree_with_retry(
                    note=note,
                    model=model,
                    temperature=temperature,
                    tissue_list=tissue_list,
                    oncotree_base=oncotree_base,
                ),
            }
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
                    "event_type": e.event_type,
                    "error": str(e),
                    "llm_response": e.content,
                },
            )
            print(f"Failed {index} of {len(files_to_process)} notes: {path.name}: invalid_event_json ({e.event_type})")
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
    skip_files.update(existing_logged_source_files(skip_log_path))
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
