import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from primary_irae_filter import filter_primary_iraes


HEATMAP_SPECS = [
    ("iraes", "irAE Co-occurrence", "condition", "irae"),
    ("irae_types", "irAE Type Co-occurrence", "irae_type", "irae"),
    ("treatment_regimens", "Full Normalized Treatment Regimen Co-occurrence", "condition", "immunotherapy"),
    ("ici_regimens", "ICI Regimen Co-occurrence", "ici_combo", "immunotherapy"),
    ("treatment_categories", "Treatment Category Co-occurrence", "therapy_type_consolidated", "immunotherapy"),
    ("ici_classes", "ICI Class Co-occurrence", "ici_class", "immunotherapy"),
    ("irae_treatments", "irAE Treatment Co-occurrence", "condition", "irae_treatment"),
    ("irae_treatment_types", "irAE Treatment Type Co-occurrence", "irae_treatment_type", "irae_treatment"),
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
    if field in {"condition", "ici_combo", "ici_class", "associated_treatment", "associated_ici", "associated_ici_class"}:
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
    score_matrix = []

    for row_label in labels:
        count_row = []
        score_row = []
        for col_label in labels:
            overlap = len(by_value[row_label] & by_value[col_label])
            smaller_count = min(len(by_value[row_label]), len(by_value[col_label]))
            count_row.append(overlap)
            score_row.append(overlap / smaller_count if smaller_count else 0)
        count_matrix.append(count_row)
        score_matrix.append(score_row)

    return labels, count_matrix, score_matrix


def wrap_label(value, width=22):
    return textwrap.fill(str(value), width=width)


def write_heatmap(labels, count_matrix, score_matrix, title, output_path, dpi):
    if len(labels) < 2:
        return False

    size = max(8, min(16, 0.65 * len(labels) + 4))
    wrapped = [wrap_label(label) for label in labels]
    scores = pd.DataFrame(score_matrix, index=wrapped, columns=wrapped)
    counts = pd.DataFrame(count_matrix, index=wrapped, columns=wrapped)

    grid = sns.clustermap(
        scores,
        annot=counts,
        fmt="",
        cmap="Reds",
        vmin=0,
        vmax=1,
        figsize=(size, size),
        cbar_kws={"label": "Co-occurrence score"},
        cbar_pos=(0.97, 0.25, 0.02, 0.5),
        linewidths=0,
        dendrogram_ratio=(0.08, 0.08),
    )
    #grid.ax_heatmap.set_title(title, fontsize=16, fontweight="bold", pad=20)
    grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9, fontweight="bold")
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=9, fontweight="bold")
    grid.fig.subplots_adjust(right=0.82)
    grid.ax_cbar.set_position([0.97, 0.25, 0.02, 0.5])
    grid.fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(grid.fig)
    return True


def export_heatmaps(records, output_dir, top_n, dpi, include_unknown, primary_only=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, title, field, condition_type in HEATMAP_SPECS:
        heatmap_records = filter_primary_iraes(records, primary_only) if condition_type == "irae" else records
        by_value = patient_sets(
            heatmap_records,
            field=field,
            condition_type=condition_type,
            include_unknown=include_unknown,
        )
        by_value = top_patient_sets(by_value, top_n=top_n)
        labels, count_matrix, score_matrix = cooccurrence_matrices(by_value)
        output_path = output_dir / f"{filename}_cooccurrence.png"

        if write_heatmap(labels, count_matrix, score_matrix, title, output_path, dpi=dpi):
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
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    export_heatmaps(
        records=records,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
        dpi=args.dpi,
        include_unknown=args.include_unknown,
        primary_only=args.primary_only,
    )
