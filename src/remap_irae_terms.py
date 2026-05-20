import argparse
import json
from pathlib import Path

from irae_mapping import map_irae_name, map_irae_type


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def remap_record(record, model, temperature, irae_names_path, irae_map_path):
    if record.get("condition_type") != "irae":
        return record

    raw_condition = record.get("raw_condition") or record.get("condition") or ""
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

    return {
        **record,
        "raw_condition": raw_condition,
        "condition": irae_name,
        "irae_type": irae_type,
    }


def remap_file(input_path, output_path, model, temperature, irae_names_path, irae_map_path):
    total_rows = count_rows(input_path)
    remapped_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for index, record in enumerate(read_jsonl(input_path), start=1):
            if record.get("condition_type") == "irae":
                remapped_rows += 1

            new_record = remap_record(
                record,
                model=model,
                temperature=temperature,
                irae_names_path=irae_names_path,
                irae_map_path=irae_map_path,
            )
            out.write(json.dumps(new_record) + "\n")

            if index % 25 == 0 or index == total_rows:
                print(f"Processed {index} of {total_rows} rows")

    print(f"Wrote {total_rows} rows to {output_path}")
    print(f"Remapped {remapped_rows} irAE rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap irAE terms in an existing extracted JSONL.")
    parser.add_argument("--input", default="data/patient_events.jsonl")
    parser.add_argument("--output", default="data/patient_events_remapped_irae.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--irae-names", default="data/irae_terms/irae_names.txt")
    parser.add_argument("--irae-map", default="data/irae_terms/irae_map.json")
    args = parser.parse_args()

    remap_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model=args.model,
        temperature=args.temperature,
        irae_names_path=args.irae_names,
        irae_map_path=args.irae_map,
    )
