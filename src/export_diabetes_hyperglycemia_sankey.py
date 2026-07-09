import argparse
from pathlib import Path

import export_tripleM_sankey as sankey


TARGET_IRAES = {
    "diabetes mellitus": "Diabetes mellitus",
    "hyperglycemia": "Hyperglycemia",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export a patient-level Sankey-style PNG for cancer type -> associated ICI class -> "
            "diabetes mellitus/hyperglycemia phenotype."
        )
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output", default="outputs/diabetes_hyperglycemia_sankey.png")
    parser.add_argument("--cancer-field", default="oncotree_tissue")
    parser.add_argument("--top-cancers", type=int, default=12, help="Collapse lower-frequency cancer groups into Other.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    sankey.TARGET_IRAES = TARGET_IRAES

    records = sankey.read_jsonl(Path(args.input))
    paths = sankey.top_cancers(
        sankey.sankey_paths(records, cancer_field=args.cancer_field),
        top_n=args.top_cancers,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sankey.write_sankey(
        paths,
        output_path,
        "Cancer Type to ICI Class to Diabetes Mellitus/Hyperglycemia Phenotype",
        dpi=args.dpi,
    ):
        print(f"Wrote {output_path}")
        print(f"Patients included: {sum(paths.values())}")
    else:
        print("Skipped Sankey: no patients with target irAEs found.")
