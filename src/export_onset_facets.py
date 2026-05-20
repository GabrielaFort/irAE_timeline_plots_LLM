import argparse
import json
import random
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


WEEKS_PER_MONTH = 4.34524
LINE_COLOR = "#0072B2"
MEDIAN_COLOR = "#D55E00"


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def field_value(record, field):
    if field == "all":
        return "All"

    value = record.get(field)
    if value is None or str(value).strip().lower() in {"", "unknown", "none", "null", "na", "n/a"}:
        return "Unknown"
    return str(value)


def top_values(records, field, max_values):
    counts = Counter(field_value(record, field) for record in records)
    return [value for value, _ in counts.most_common(max_values)]


def onset_value(record, unit):
    months = record.get("time_to_onset_months")
    if months is None:
        return None
    value = float(months)
    if unit == "weeks":
        return value * WEEKS_PER_MONTH
    return value


def median_value(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def jitter_positions(count, seed):
    if count == 1:
        return [0]

    rng = random.Random(seed)
    return [rng.uniform(-0.18, 0.18) for _ in range(count)]


def wrapped(value, width=18):
    return textwrap.fill(value, width=width)


def export_onset_facets(
    records,
    output_path,
    row_field,
    col_field,
    max_rows,
    max_cols,
    unit,
    max_x,
    dpi,
):
    irae_records = [
        record
        for record in records
        if record.get("condition_type") == "irae" and record.get("time_to_onset_months") is not None
    ]
    if not irae_records:
        raise ValueError("No irAE records with time_to_onset_months found.")

    row_values = top_values(irae_records, row_field, max_rows)
    col_values = top_values(irae_records, col_field, max_cols)

    fig_width = max(10, 2.0 * len(col_values) + 2)
    fig_height = max(5, 1.55 * len(row_values) + 2)
    fig, axes = plt.subplots(
        len(row_values),
        len(col_values),
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row_index, row_value in enumerate(row_values):
        for col_index, col_value in enumerate(col_values):
            ax = axes[row_index][col_index]
            values = [
                onset_value(record, unit)
                for record in irae_records
                if field_value(record, row_field) == row_value and field_value(record, col_field) == col_value
            ]
            values = [value for value in values if value is not None and 0 <= value <= max_x]

            if values:
                y_values = jitter_positions(len(values), seed=f"{row_value}|{col_value}")
                ax.scatter(values, y_values, color=LINE_COLOR, alpha=0.65, s=18, linewidth=0)
                median = median_value(values)
                ax.axvline(median, color=MEDIAN_COLOR, linestyle="--", linewidth=1.4)
                label = f"median {median:.1f}  n={len(values)}"
            else:
                label = "n=0"

            ax.axvline(0, color="0.65", linestyle=":", linewidth=1)
            ax.set_xlim(0, max_x)
            ax.set_ylim(-0.35, 0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("0.8")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.tick_params(axis="x", labelsize=8)
            ax.text(0.04, 0.78, label, transform=ax.transAxes, fontsize=8, fontweight="bold")

            if row_index == 0:
                ax.set_title(wrapped(col_value), fontsize=10, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(wrapped(row_value), fontsize=10, fontweight="bold", rotation=0, ha="right", va="center")

    x_label = f"Time to Onset ({unit.title()})"
    fig.supxlabel(x_label, fontsize=13, fontweight="bold")
    fig.suptitle("Onset of irAEs", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export faceted irAE time-to-onset plots.")
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output", default="outputs/onset_facets/onset_by_ici_and_irae_type.png")
    parser.add_argument("--row-field", default="associated_ici")
    parser.add_argument("--col-field", default="irae_type")
    parser.add_argument("--max-rows", type=int, default=6)
    parser.add_argument("--max-cols", type=int, default=8)
    parser.add_argument("--unit", choices=["weeks", "months"], default="weeks")
    parser.add_argument("--max-x", type=float, default=52)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    export_onset_facets(
        records=read_jsonl(Path(args.input)),
        output_path=Path(args.output),
        row_field=args.row_field,
        col_field=args.col_field,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        unit=args.unit,
        max_x=args.max_x,
        dpi=args.dpi,
    )
    print(f"Wrote {args.output}")
