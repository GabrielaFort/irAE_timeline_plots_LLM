import argparse
import json
from pathlib import Path

import ollama

PROMPT = """
You are extracting structured oncology timeline data from clinical notes.

Return JSON only with this exact schema:
{
  "events": [
    {
      "condition_type": "treatment" or "irae" or "disease"
      "condition": "short clinical label",
      "start_date": "YYYY-MM-DD or YYYY-MM or YYYY",
      "end_date": "YYYY-MM-DD or YYYY-MM or YYYY or null"
    }
  ]
}

Rules:
1. Include all dated immunotherapy-related treatments, iraes, and diseases.
2. `treatment` includes:
- checkpoint inhibitors (e.g., pembrolizumab, nivolumab, atezolizumab)
- other drugs (e.g., steroids, IV methylprednisolone, prophylaxis, chemotherapy, targeted therapy)
3. `irae` includes **immune-related toxicities/events only**.
4. `disease` includes **oncological conditions only**.
5. Sometimes there may be multiple entries per treatment, irae, or disease if there are multiple distinct time periods (e.g., treatment stopped and restarted, irae resolved and recurred).
6. If a date range is implied, set `start_date` and `end_date`.
7. If only onset is known, set `start_date` only and `end_date` to null.
8. Preserve date precision from source text (day > month > year).
9. Use concise condition names (e.g., "Atezolizumab", "Hepatitis").
10. Do not add keys beyond the schema.
11. ONLY return JSON, no explanations or extra text.
"""


def extract_events(model, temperature, note):
    """Extract structured events from a clinical note using the LLM."""
    reponse = ollama.chat(
        model=model,
        format='json',
        options={'temperature': temperature},
        messages=[
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': note},
        ],
    )
    data = json.loads(reponse['message']['content'])
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


def parse_patient_files(model, temperature, input_dir, pattern):
    """Parse all files matching pattern in input_dir and return per-file events."""
    results = {}
    files = sorted(input_dir.glob(pattern))

    for path in files:
        note = path.read_text(encoding="utf-8", errors="ignore")
        results[path.name] = extract_events(model=model, temperature=temperature, note=note)

    return files, results


def write_events_jsonl(results, output_path):
    """Write one JSON object per event line with patient metadata."""
    total_events = 0
    patient_count = 1
    with output_path.open("w", encoding="utf-8") as f:
        for source_file, events in results.items():
            patient_id = f"patient_{patient_count}"
            patient_count += 1
            for event in events:
                record = {
                    "patient_id": patient_id,
                    "source_file": source_file,
                    **event,
                }
                f.write(json.dumps(record) + "\n")
                total_events += 1

    return total_events



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-parse patient notes with Ollama.")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g., llama3.1:8b).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--input-dir", default="data", help="Directory containing patient note .txt files.")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern for note files.")
    parser.add_argument("--output", default="data/patient_events.jsonl", help="Path to output JSONL file.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    files, results = parse_patient_files(
        model=args.model,
        temperature=args.temperature,
        input_dir=input_dir,
        pattern=args.pattern,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_events = write_events_jsonl(results=results, output_path=output_path)

    print(f"Parsed {len(files)} files and wrote {total_events} events to {output_path}")
