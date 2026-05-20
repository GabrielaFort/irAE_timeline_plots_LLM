import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


BAR_COLOR = "#0072B2"
FIG_SIZE = (12, 9)

SUMMARY_SPECS = [
    ("iraes", "Most Common irAEs", "condition", "irae"),
    ("irae_types", "Most Common irAE Types", "irae_type", "irae"),
    ("icis", "Most Common ICIs", "condition", "immunotherapy"),
    ("irae_treatments", "Most Common irAE Treatments", "condition", "irae_treatment"),
    ("irae_treatment_types", "Most Common irAE Treatment Types", "irae_treatment_type", "irae_treatment"),
    ("oncotree_tissues", "OncoTree Tissues", "oncotree_tissue", None),
    ("oncotree_names", "OncoTree Names", "oncotree_name", None),
]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def patient_sets(records, field, condition_type=None):
    by_value = {}
    for record in records:
        if condition_type and record.get("condition_type") != condition_type:
            continue

        value = record.get(field) or "Unknown"
        by_value.setdefault(value, set()).add(record.get("patient_id"))

    return by_value


def grouped_counts(by_value, total_patients, min_percent):
    kept = {}
    other_patients = set()

    for value, patients in by_value.items():
        percent = 100 * len(patients) / total_patients if total_patients else 0
        if percent < min_percent:
            other_patients.update(patients)
        else:
            kept[value] = patients

    if other_patients:
        kept["Other"] = other_patients

    return sorted(
        [(value, len(patients), 100 * len(patients) / total_patients) for value, patients in kept.items()],
        key=lambda item: (-item[1], item[0]),
    )


def wrap_label(value, width=36):
    return textwrap.fill(str(value), width=width)


def write_bar(rows, title, output_path, dpi):
    if not rows:
        return False

    labels = [wrap_label(row[0]) for row in rows]
    counts = [row[1] for row in rows]
    percents = [row[2] for row in rows]
    y_positions = range(len(rows))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.barh(y_positions, percents, color=BAR_COLOR, edgecolor="white")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    x_max = max(100, max(percents) + 15)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Percent of patients", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count, percent in zip(bars, counts, percents):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.1f}% (n={count})",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.34, right=0.92, top=0.88, bottom=0.12)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def export_charts(records, output_dir, min_percent, dpi):
    total_patients = len({record.get("patient_id") for record in records if record.get("patient_id")})
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, title, field, condition_type in SUMMARY_SPECS:
        rows = grouped_counts(
            patient_sets(records, field, condition_type=condition_type),
            total_patients=total_patients,
            min_percent=min_percent,
        )
        output_path = output_dir / f"{filename}.png"
        if write_bar(rows, f"{title} (N={total_patients})", output_path, dpi=dpi):
            print(f"Wrote {output_path}")
        else:
            print(f"Skipped {title}: no data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export summary bar charts from normalized event JSONL.")
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs/summary_bars")
    parser.add_argument("--min-percent", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    export_charts(
        records=records,
        output_dir=Path(args.output_dir),
        min_percent=args.min_percent,
        dpi=args.dpi,
    )
