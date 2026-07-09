import argparse
import json
import os

import pandas as pd


def normalize_mrn(mrn):
    value = str(mrn).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(8) if value.isdigit() else value


def mrn_from_source_file(source_file):
    filename = os.path.basename(str(source_file))
    return normalize_mrn(filename.split("_", 1)[1].rsplit(".", 1)[0])


def filter_normalized_jsonl(normalized_data_path, survival_data_path, output_path):
    survival_df = pd.read_csv(survival_data_path)
    survival_mrns = {normalize_mrn(mrn) for mrn in survival_df["MRN"]}

    kept = 0
    with open(normalized_data_path, encoding="utf-8") as input_file, open(output_path, "w", encoding="utf-8") as output_file:
        for line in input_file:
            record = json.loads(line)
            if mrn_from_source_file(record.get("source_file")) in survival_mrns:
                output_file.write(json.dumps(record) + "\n")
                kept += 1

    return kept


def main():
    parser = argparse.ArgumentParser(description="Filter normalized JSONL to patients in the survival cohort.")
    parser.add_argument("--normalized-data", default="data/events_060926_gemma4_e4b_normalized_rxnorm.jsonl")
    parser.add_argument("--survival-data", default="data/survival_data_labeled.csv")
    parser.add_argument("--output", default="data/events_060926_gemma4_e4b_normalized_rxnorm_survival_cohort.jsonl")
    args = parser.parse_args()

    kept = filter_normalized_jsonl(args.normalized_data, args.survival_data, args.output)
    print(f"Wrote {kept} records to {args.output}")


if __name__ == "__main__":
    main()
