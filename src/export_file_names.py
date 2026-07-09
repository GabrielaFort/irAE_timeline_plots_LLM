
import json
from pathlib import Path

input_path = Path("data/events_060926_gemma4_e4b_normalized_rxnorm.jsonl")
output_path = Path("data/normalized_source_files.txt")

source_files = []
seen = set()

with input_path.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        source_file = record.get("source_file")
        if not source_file:
            continue

        name = Path(source_file).stem  # removes .txt
        if name not in seen:
            seen.add(name)
            source_files.append(name)

output_path.write_text("\n".join(source_files) + "\n", encoding="utf-8")
print(f"Wrote {len(source_files)} source files to {output_path}")
