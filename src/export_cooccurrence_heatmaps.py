import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


HEATMAP_SPECS = [
    ("iraes", "irAE Co-occurrence", "condition", "irae"),
    ("irae_types", "irAE Type Co-occurrence", "irae_type", "irae"),
    ("icis", "ICI Co-occurrence", "condition", "immunotherapy"),
    ("ici_classes", "ICI Class Co-occurrence", "ici_class", "immunotherapy"),
    ("associated_ici_classes", "Associated ICI Class Co-occurrence", "associated_ici_class", "irae"),
    ("irae_treatments", "irAE Treatment Co-occurrence", "condition", "irae_treatment"),
    ("irae_treatment_types", "irAE Treatment Type Co-occurrence", "irae_treatment_type", "irae_treatment"),
    ("oncotree_tissues", "OncoTree Tissue Co-occurrence", "oncotree_tissue", None),
    ("oncotree_names", "OncoTree Name Co-occurrence", "oncotree_name", None),
]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_unknown(value):
    return value is None or str(value).strip().lower() in {"", "unknown", "none", "null", "na", "n/a"}


def split_combo_value(value):
    return [part.strip() for part in str(value).split("+") if part.strip()]


def record_values(record, field):
    value = record.get(field)
    if field in {"condition", "ici_class", "associated_ici", "associated_ici_class"}:
        return split_combo_value(value)
    return [value]


def patient_sets(records, field, condition_type=None, include_unknown=False):
    by_value = {}
    for record in records:
        if condition_type and record.get("condition_type") != condition_type:
            continue

        patient_id = record.get("patient_id")
        if not patient_id:
            continue

        for value in record_values(record, field):
            if is_unknown(value):
                if not include_unknown:
                    continue
                value = "Unknown"

            by_value.setdefault(str(value), set()).add(patient_id)

    return by_value


def top_patient_sets(by_value, top_n):
    return dict(
        sorted(
            by_value.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:top_n]
    )


def cooccurrence_matrices(by_value):
    labels = list(by_value)
    count_matrix = []
    jaccard_matrix = []

    for row_label in labels:
        count_row = []
        jaccard_row = []
        for col_label in labels:
            overlap = len(by_value[row_label] & by_value[col_label])
            union = len(by_value[row_label] | by_value[col_label])
            count_row.append(overlap)
            jaccard_row.append(overlap / union if union else 0)
        count_matrix.append(count_row)
        jaccard_matrix.append(jaccard_row)

    return labels, count_matrix, jaccard_matrix


def wrap_label(value, width=22):
    return textwrap.fill(str(value), width=width)


def write_heatmap(labels, count_matrix, jaccard_matrix, title, output_path, dpi):
    if len(labels) < 2:
        return False

    size = max(8, min(16, 0.65 * len(labels) + 4))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(jaccard_matrix, cmap="Blues", vmin=0, vmax=1)

    wrapped = [wrap_label(label) for label in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(wrapped, rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticklabels(wrapped, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    for row_index, row in enumerate(count_matrix):
        for col_index, value in enumerate(row):
            jaccard = jaccard_matrix[row_index][col_index]
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if jaccard > 0.5 else "black",
                fontsize=8,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Jaccard overlap", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def export_heatmaps(records, output_dir, top_n, dpi, include_unknown):
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, title, field, condition_type in HEATMAP_SPECS:
        by_value = patient_sets(
            records,
            field=field,
            condition_type=condition_type,
            include_unknown=include_unknown,
        )
        by_value = top_patient_sets(by_value, top_n=top_n)
        labels, count_matrix, jaccard_matrix = cooccurrence_matrices(by_value)
        output_path = output_dir / f"{filename}_cooccurrence.png"

        if write_heatmap(labels, count_matrix, jaccard_matrix, title, output_path, dpi=dpi):
            print(f"Wrote {output_path}")
        else:
            print(f"Skipped {title}: fewer than 2 values")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export patient-level metadata co-occurrence heatmaps.")
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs/cooccurrence_heatmaps")
    parser.add_argument("--top-n", type=int, default=20, help="Maximum number of terms per heatmap.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-unknown", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    export_heatmaps(
        records=records,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
        dpi=args.dpi,
        include_unknown=args.include_unknown,
    )
