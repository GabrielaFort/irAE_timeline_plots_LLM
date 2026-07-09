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

DEFAULT_OPTIONS = {'temperature': 1.0,
                   'top_p': 0.95,
                   'top_k': 64}

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


DATE_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "start_date": {"type": "string"},
                },
                "required": ["index", "start_date"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


IMMUNOTHERAPY_PROMPT = """
You are extracting cancer immunotherapy regimen timeline events from clinical notes.

Return JSON according to the schema. Return no explanations.

Extract ONLY cancer immunotherapy regimen START/RESTART/CHANGE events. Do not extract STOP or DISCONTINUATION events.
Do **NOT** extract every cycle, dose, infusion, administration, continuation, maintenance mention, follow-up mention, or medication adjustment.

Include ONLY immune checkpoint inhibitors (ICI) drug combinations that include ICIs.
ICIs may include Relatlimab, Ipilimumab, Fianlimab, Avelumab, Durvalumab, Atezolizumab, Cemiplimab, Nivolumab, Pembrolizumab, Dostarlimab, Serplulimab, or any other checkpoint inhibitor (even less common investigational ones) mentioned in the note.

Do NOT include:
- surgery
- supportive care
- steroids or immunosuppressants used to treat irAEs
- prophylaxis
- non-cancer treatments
- other immunotherapy (i.e. cytokines, cell therapies, vaccines, etc)
- drug regimens that do not include ICIs (e.g. chemotherapy alone, targeted therapy alone, chemo + targeted therapy, etc)

New event rules:
1. Create an event when a new ICI regimen starts.
2. Create an event when an ICI regimen changes.
3. Create an event when an ICI drug is added or removed.
4. Create an event when ICI therapy restarts after a documented pause or discontinuation.
5. IMPORTANT: Do **NOT** create new events for repeated cycles/doses/infusions/mentions of the same regimen.
6. Do **NOT** create events for continued therapy, maintenance mention, or planned next cycle unless the regimen changed.

Combination rules:
1. Combination is allowed ONLY for combinations that include ICIs.
2. If multiple drugs start as one regimen, write one condition with drugs joined by " + ".
3. If chemotherapy, radiation, or targeted therapy plus ICI is listed, include the names of all drugs separated by " + ".

Examples:
- "carboplatin/pemetrexed/pembrolizumab" -> "Pembrolizumab + Carboplatin + Pemetrexed"
- "nivolumab plus ipilimumab" -> "Ipilimumab + Nivolumab"
- "ipi/nivo followed by nivolumab maintenance" -> two events:
  1. "Ipilimumab + Nivolumab" at combination start
  2. "Nivolumab" at maintenance monotherapy start
- repeated "C2 nivolumab", "C3 nivolumab", "C4 nivolumab" -> only one "Nivolumab" event at first start date
- "nivolumab was held for cycle 3, then restarted in cycle 4" -> one "Nivolumab" event at first start date, then one "Nivolumab" event at restart date
- "pragmatica trial (ramucimumab + pembrolizumab)" -> "Pembrolizumab + Ramucimumab", ignore the trial name
- "AC (Adriamycin/Cyclophosphamide) + Pembrolizumab" -> "Pembrolizumab + Adriamycin + Cyclophosphamide"

IMPORTANT Date rules:
- Extract only start/restart/change dates.
- Preserve source precision: YYYY-MM-DD, YYYY-MM, or YYYY.
- If an exact date is not stated but timing is clearly implied, estimate the most specific reasonable date from context.
- Examples: March 2020 -> 2020-03, March 10, 2022 -> 2022-03-10, "in 2021" -> 2021, "started in early 2020" -> 2020-01, "10/05/2023" -> 2023-10-05

Condition rules:
- Use concise generic drug names (e.g. "Atezolizumab", "Nivolumab + Ipilimumab", "Durvalumab").
- No dose, route, frequency, cycle number, or other explanatory text.
- If a trial is described, extract only the drug names, not the trial name.
- If the name of an ICI is not clear be sure to double check all parts of the note.
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
- "colitis/enteritis" -> two events "Colitis" and "Enteritis" 

IMPORTANT Date rules:
- Extract only onset/start date.
- Preserve source precision: YYYY-MM-DD, YYYY-MM, or YYYY.
- If an exact date is not stated but timing is clearly implied, estimate the most specific reasonable date from context.
- If an event occurred "shortly after" a dated treatment start, use the day after the treatment start date.
- Examples: March 2020 -> 2020-03, March 10, 2022 -> 2022-03-10, "in 2021" -> 2021, "started in early 2020" -> 2020-01, "10/05/2023" -> 2023-10-05

Condition rules:
- Use concise clinical toxicity names.
- Prefer ASCO-style irAE terms when possible.
- No grade, dose, route, severity, or explanatory text.
"""


IRAE_TREATMENT_PROMPT = """
You are extracting treatments used to manage immune-related adverse events (irAEs) from clinical notes.

Return JSON according to the schema. Return no explanations.

Extract ONLY MEDICATION NAMES that are being used to treat or manage an irAE.
Use the most specific term possible (e.g. "Prednisone" instead of "steroid", "Mycophenolate" instead of "immunosuppressant").

IMPORTANT:
An irAE treatment is the medication used to treat the irAE, NOT the adverse event itself.
Do NOT extract symptoms, toxicities, diagnoses, or indications.

Extract:
- Corticosteroids: e.g. 'Prednisone', 'Methylprednisolone', 'Dexamethasone', 'Hydrocortisone', 'Hydrocortisone Cream', 'Trimcinolone', 'Cortisone Cream', etc
- Non-steroid immunosuppressants/immunomodulators: e.g. 'Mycophenolate', 'Vedolizumab', 'Infliximab', 'Immunoglobulin G', 'Methotrexate', etc
- Endocrine/metabolic replacement or regulation: e.g. 'Levothyroxine', 'Insulin', 'Metformin', 'Progesterone', 'Testosterone', etc

Do NOT extract:
- The irAE itself: Rash, Itching, Pruritus, Colitis, Diarrhea, Pneumonitis, Hepatitis, Arthralgia, Hypothyroidism, Adrenal insufficiency
- Cancer-directed treatments: immunotherapy, chemotherapy, targeted therapy, radiation, surgery
- Dose changes, tapers, refills, or continued treatment mentions
- Supportive care or prophylaxis: antiemetics, pain medications, anti-infectives, etc.
- Other procedures or interventions used: Plasmapheresis, physical therapy, hydration, etc

Event rules:
1. Create an event when a new irAE-directed treatment starts.
2. Create a new event only if the treatment restarts after being stopped.
3. Do not create events for dose changes or tapers.
4. Do not create events for continued use.
5. Each treatment **must** be a separate event. You should create a separate event for each distinct treatment mentioned, even if they start on the same day.

Condition field rules:
- The condition must be the treatment name only.
- Do not include the irAE, symptom, indication, dose, route, frequency, or taper instructions.
- Use concise generic treatment names.

IMPORTANT Date rules:
- Extract only treatment start/restart date.
- Preserve source precision: YYYY-MM-DD, YYYY-MM, or YYYY.
- If an exact date is not stated but timing is clearly implied, estimate the most specific reasonable date from context.
- Examples: March 2020 -> 2020-03, March 10, 2022 -> 2022-03-10, "in 2021" -> 2021, "started in early 2020" -> 2020-01, "10/05/2023" -> 2023-10-05
"""

DATE_REVIEW_PROMPT = """
You are reviewing extracted oncology timeline event start dates against a patient note summary.

Return only valid JSON according to the provided schema. Do not include explanations outside JSON.

You will receive:
1. A patient note summary.
2. A list of already extracted timeline events, each with an immutable index.

Your task:
- Review each event's start_date against the patient note summary.
- Return exactly one review object for each input event index.
- Keep the same event count.
- Do not add, remove, merge, split, or reorder events.
- Do not edit condition_type or condition.
- Only edit start_date when the note summary clearly supports a better date.

Allowed date formats:
- YYYY-MM-DD
- YYYY-MM
- YYYY
- Unknown

Core rules:
- Prefer the date precision supported by the note.
- If the note gives an exact day, return YYYY-MM-DD.
- If the note gives only month/year, return YYYY-MM.
- If the note gives only year, return YYYY.
- Keep the extracted date unchanged if it is supported by the note.
- Do not use documentation, follow-up, scan, assessment, or clinic visit dates as event onset dates unless the note says the event started on that date.

Event-specific rules:
- For immunotherapy events, use the regimen start, restart, or change date.
- For irAE events, use the adverse event onset/start date.
- For irAE treatment events, use the treatment start/restart date.
- Do not use later cycles, continuation dates, taper dates, refill dates, response-assessment dates, or generic follow-up dates.

ICI sequencing and deduction rules:
- irAE events should not be dated before the relevant ICI exposure if the note clearly says the irAE occurred after ICI start.
- If the note gives an ICI start date and says the irAE occurred after that start, use the most specific defensible date from the note.
- If the note says the irAE occurred in the same month/year as ICI start but after ICI start, and no exact irAE day is given, you may infer the ICI start date as the earliest defensible irAE date.
- If the note uses phrases like "shortly after", "soon after", "following", "after starting", or "subsequent to" a dated ICI start, and no more specific date is given, use the day after the ICI start date.
- If the note says the irAE occurred after ICI start but gives no date/month/year and no clear relative timing phrase, return Unknown.
- Do not infer a date from ICI start if the note does not clearly link the irAE timing to that ICI start.
- Do not force irAE treatment dates to occur after ICI unless the treatment is clearly for an irAE caused by or occurring after ICI.

Partial date handling:
- If an irAE is stated as occurring "in March 2020" and ICI started on 2020-03-15, return 2020-03-15 only if the note clearly says the irAE occurred after that ICI start.
- If an irAE is stated as occurring "in March 2020" but there is no clear link that it occurred after ICI start, return 2020-03.
- If a year-only or month-only date would place an irAE before a clearly stated ICI exposure, refine it only when the note supports the sequence.
- Do not invent exact days just to make the timeline cleaner.

Approximate timing:
- "early YEAR" -> YYYY-01
- "mid YEAR" -> YYYY-06
- "late YEAR" -> YYYY-12
- "early MONTH YEAR" -> YYYY-MM
- "mid MONTH YEAR" -> YYYY-MM
- "late MONTH YEAR" -> YYYY-MM
- For "next day" after a dated event, use the next calendar day.
- For "within X days/weeks/months after" a dated event, estimate the earliest reasonable date in that interval.

Examples:
- ICI start: 2020-03-15. Note: "Colitis developed in March 2020 after starting nivolumab."
  Return Colitis start_date: 2020-03-15

- ICI start: 2020-03-15. Note: "Colitis developed shortly after starting nivolumab."
  Return Colitis start_date: 2020-03-16

- ICI start: 2020-03-15. Note: "Colitis developed in March 2020." No clear link to ICI timing.
  Return Colitis start_date: 2020-03

- ICI start: 2020-03-15. Note: "Colitis occurred after nivolumab." No month/year or relative timing.
  Return Colitis start_date: Unknown

- Note: "Pneumonitis was diagnosed on CT in July 2021, but cough began in June 2021."
  Return Pneumonitis start_date: 2021-06

- Note: "Prednisone was started the next day for colitis."
  Use the day after the colitis date if the colitis date is known.
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


def print_token_counts(step, response):
    input_tokens = response.get("prompt_eval_count", "unknown")
    output_tokens = response.get("eval_count", "unknown")
    print(f"{step} tokens: input={input_tokens}, output={output_tokens}")


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
        #options=DEFAULT_OPTIONS,
        think=True,
        stream=False,
        messages=messages,
    )
    print_token_counts(condition_type, response)
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
            #options=DEFAULT_OPTIONS,
            messages=messages + [
                {
                    'role': 'user',
                    'content': 'Your previous response was invalid JSON. Return the complete JSON object again, with no extra text.',
                }
            ],
        )
        print_token_counts(f"{condition_type} retry", response)
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


def event_payload_for_date_review(events):
    return [
        {
            "index": index,
            "condition_type": event.get("condition_type"),
            "condition": event.get("condition"),
            "irae_type": event.get("irae_type"),
            "start_date": event.get("start_date"),
        }
        for index, event in enumerate(events)
    ]


def date_review_user_prompt(note_summary, events):
    return (
        "Patient note summary:\n"
        f"{note_summary}\n\n"
        "Extracted events to review:\n"
        f"{json.dumps(event_payload_for_date_review(events), indent=2)}"
    )


def apply_date_reviews(events, reviews):
    reviews_by_index = {}
    for review in reviews:
        try:
            index = int(review.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(events):
            reviews_by_index[index] = review

    reviewed_events = []
    changed_count = 0
    for index, event in enumerate(events):
        reviewed = dict(event)
        review = reviews_by_index.get(index)
        if review is None:
            reviewed_events.append(reviewed)
            continue

        original_start = str(event.get("start_date", "")).strip()
        reviewed_start = str(review.get("start_date", "")).strip() or "Unknown"

        if reviewed_start != original_start:
            reviewed["start_date"] = reviewed_start
            changed_count += 1

        reviewed_events.append(reviewed)

    return reviewed_events, changed_count


def review_event_dates(model, temperature, note_summary, events):
    if not events:
        return events

    messages = [
        {"role": "system", "content": DATE_REVIEW_PROMPT},
        {"role": "user", "content": date_review_user_prompt(note_summary, events)},
    ]
    response = ollama.chat(
        model=model,
        format=DATE_REVIEW_SCHEMA,
        options={"temperature": temperature},
        stream=False,
        messages=messages,
    )
    print_token_counts("date review", response)
    content = response["message"]["content"]
    print(f"date review LLM response content preview: {content[:120]}...")
    try:
        data = parse_json_object(content)
    except json.JSONDecodeError:
        print("Could not parse date review JSON. Retrying once.")
        response = ollama.chat(
            model=model,
            format=DATE_REVIEW_SCHEMA,
            options={"temperature": 0.0},
            stream=False,
            messages=messages + [
                {
                    "role": "user",
                    "content": "Your previous response was invalid JSON. Return the complete JSON object again, with no extra text.",
                }
            ],
        )
        print_token_counts("date review retry", response)
        content = response["message"]["content"]
        print(f"Retry date review LLM response content preview: {content[:120]}...")
        try:
            data = parse_json_object(content)
        except json.JSONDecodeError as e:
            raise EventJSONError(str(e), content, "date_review") from e

    reviews = data.get("reviews", []) if isinstance(data, dict) else []
    reviewed_events, changed_count = apply_date_reviews(events, reviews)
    print(f"Date review completed: {changed_count} date edits")
    return reviewed_events


def unique_values(values):
    seen = set()
    out = []
    for value in values:
        value = str(value).strip()
        if not value or value.lower() in {"unknown", "none", "null", "na", "n/a"}:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def irae_treatment_prompt_for_mapped_iraes(mapped_events):
    mapped_iraes = unique_values(
        event.get("condition")
        for event in mapped_events
        if event.get("condition_type") == "irae"
    )

    prompt = IRAE_TREATMENT_PROMPT
    if mapped_iraes:
        prompt += (
            "\n\nFor this note, extract treatments ONLY if they are clearly directed toward "
            "one of these mapped irAEs. Do not extract treatments for any other condition, "
            "symptom, toxicity, or adverse event, even if the treatment would otherwise look "
            "like an irAE-directed treatment.\n\nMapped irAEs:\n- "
            + "\n- ".join(mapped_iraes)
        )
    else:
        prompt += (
            "\n\nNo extracted adverse event in this note mapped to a valid irAE. "
            "Return an empty events list for irAE-directed treatments."
        )

    return prompt


def extract_events(model, temperature, note, irae_names=None, irae_map=None):
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

    events = immunotherapy_events + irae_events
    if irae_names is not None and irae_map is not None:
        events = map_irae_events(
            events=events,
            model=model,
            temperature=temperature,
            irae_names_path=irae_names,
            irae_map_path=irae_map,
        )

    irae_treatment_events = extract_single_type_events(
        model=model,
        temperature=temperature,
        note=note,
        prompt=irae_treatment_prompt_for_mapped_iraes(events),
        condition_type="irae_treatment",
    )
    print(
        "Extracted "
        f"{len(immunotherapy_events)} immunotherapy, "
        f"{len(irae_events)} irAE, "
        f"{len(irae_treatment_events)} irAE treatment events"
    )
    events = events + irae_treatment_events
    return review_event_dates(
        model=model,
        temperature=temperature,
        note_summary=note,
        events=events,
    )


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
    return extract_events(
        model=model,
        temperature=temperature,
        note=note,
        irae_names=irae_names,
        irae_map=irae_map,
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
