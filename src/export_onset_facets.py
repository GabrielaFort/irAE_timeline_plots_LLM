import argparse
import json
import random
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


WEEKS_PER_MONTH = 4.34524
LINE_COLOR = "#0072B2"
MEDIAN_COLOR = "#D55E00"
RIDGE_COLORS = [
    "#8B1A1A",
    "#20D6A0",
    "#4B61D1",
    "#FF6A21",
    "#FF9E2C",
    "#A8EE1D",
    "#27C9D8",
    "#7B4CC2",
    "#D84C8A",
    "#2A9D55",
]


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


def wrapped(value, width=18):
    return textwrap.fill(value, width=width)


def short_number(value):
    return f"{value:.1f}".rstrip("0").rstrip(".")


def jitter_positions(count, seed):
    if count == 1:
        return [0]

    rng = random.Random(seed)
    return [rng.uniform(-0.18, 0.18) for _ in range(count)]


def export_onset_distribution(
    records,
    output_path,
    row_field,
    max_rows,
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
    values_by_row = []
    kept_rows = []
    for row_value in row_values:
        values = [
            min(value, max_x)
            for record in irae_records
            if field_value(record, row_field) == row_value
            for value in [onset_value(record, unit)]
            if value is not None and value >= 0
        ]
        if values:
            kept_rows.append(row_value)
            values_by_row.append(values)

    rows_with_values = sorted(zip(kept_rows, values_by_row), key=lambda item: median_value(item[1]))
    kept_rows = [row for row, _ in rows_with_values]
    values_by_row = [values for _, values in rows_with_values]
    fig_height = max(3.6, 0.42 * len(kept_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    plot_rows = list(reversed(kept_rows))
    values_by_label = dict(zip(kept_rows, values_by_row))
    palette = {row: RIDGE_COLORS[index % len(RIDGE_COLORS)] for index, row in enumerate(kept_rows)}

    for y_position, row_value in enumerate(plot_rows):
        values = values_by_label[row_value]
        color = palette[row_value]
        median = median_value(values)
        before_collections = len(ax.collections)
        before_lines = len(ax.lines)

        if len(set(values)) > 1:
            sns.kdeplot(x=values, ax=ax, color=color, fill=True, alpha=0.25, cut=0, linewidth=1.1)
            new_collections = ax.collections[before_collections:]
            new_lines = ax.lines[before_lines:]
            peak = 0
            for collection in new_collections:
                for path in collection.get_paths():
                    peak = max(peak, max(path.vertices[:, 1]))
            for line in new_lines:
                _, line_y = line.get_data()
                if len(line_y):
                    peak = max(peak, max(line_y))
            peak = peak or 1

            for collection in new_collections:
                collection.set_edgecolor(color)
                collection.set_linewidth(1.1)
                for path in collection.get_paths():
                    path.vertices[:, 1] = y_position + (path.vertices[:, 1] / peak * 0.36)
            for line in new_lines:
                line_x, line_y = line.get_data()
                line.set_data(line_x, [y_position + (y / peak * 0.36) for y in line_y])
        else:
            ax.plot(values, [y_position] * len(values), color=color, linewidth=1.1)

        ax.vlines(median, y_position, y_position + 0.34, color=color, linestyle="--", linewidth=1.0)
        ax.text(
            median + max_x * 0.012,
            y_position + 0.18,
            short_number(median),
            fontsize=9,
            va="center",
        )
        ax.text(
            max_x * 0.985,
            y_position + 0.18,
            f"n = {len(values)}",
            fontsize=9,
            ha="right",
            va="center",
        )

    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels([wrapped(value, width=24) for value in plot_rows], fontsize=10)
    ax.set_xlim(0, max_x)
    ax.set_ylim(-0.45, len(kept_rows) - 0.25)
    ax.set_xlabel(f"Time to Onset ({unit.title()})", fontsize=10)
    ax.set_ylabel("")
    ax.set_title("Onset of Adverse Events", fontsize=16, pad=10)
    if unit == "weeks":
        ax.set_xticks([0, 10, 20, 30, 40, max_x])
        ax.set_xticklabels(["0", "10", "20", "30", "40", f">{int(max_x)}"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("0.7")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


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

    fig.supxlabel(f"Time to Onset ({unit.title()})", fontsize=13, fontweight="bold")
    fig.suptitle("Onset of irAEs", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export irAE time-to-onset plots.")
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output", default="outputs/onset_facets/onset_distribution_by_irae_type.png")
    parser.add_argument("--row-field", default="irae_type")
    parser.add_argument("--col-field", default="all")
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--max-cols", type=int, default=8)
    parser.add_argument("--unit", choices=["weeks", "months"], default="weeks")
    parser.add_argument("--max-x", type=float, default=52)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_path = Path(args.output)
    facets_output_path = output_path.with_name(f"{output_path.stem}_facets{output_path.suffix}")
    records = read_jsonl(Path(args.input))

    export_onset_distribution(
        records=records,
        output_path=output_path,
        row_field=args.row_field,
        max_rows=args.max_rows,
        unit=args.unit,
        max_x=args.max_x,
        dpi=args.dpi,
    )
    export_onset_facets(
        records=records,
        output_path=facets_output_path,
        row_field=args.row_field,
        col_field=args.col_field,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        unit=args.unit,
        max_x=args.max_x,
        dpi=args.dpi,
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {facets_output_path}")
