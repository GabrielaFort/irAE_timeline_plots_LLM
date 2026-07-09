import json
from pathlib import Path

input_path = Path("data/events_060926_gemma4_e4b_normalized_rxnorm.jsonl")
output_path = Path("data/events_060926_gemma4_e4b_normalized_myositis_mg_myocarditis.jsonl")

target_iraes = {"myositis", "myasthenia gravis", "myocarditis"}

records = [
    json.loads(line)
    for line in input_path.read_text().splitlines()
    if line.strip()
]

patient_ids = {
    record["patient_id"]
    for record in records
    if record.get("condition_type") == "irae"
    and str(record.get("condition", "")).strip().lower() in target_iraes
}

subset = [
    record
    for record in records
    if record.get("patient_id") in patient_ids
]

with output_path.open("w", encoding="utf-8") as f:
    for record in subset:
        f.write(json.dumps(record) + "\n")

print(f"Patients: {len(patient_ids)}")
print(f"Rows written: {len(subset)}")
print(f"Wrote {output_path}")
