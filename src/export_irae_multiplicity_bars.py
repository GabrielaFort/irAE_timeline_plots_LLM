import argparse
import json
import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir())
os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(cache_dir)

import matplotlib.pyplot as plt


BAR_COLOR = "#0072B2"
FIG_SIZE = (8, 5)
MULTIPLICITY_SPECS = [
    ("irae_name_multiplicity", "Patients With One vs Multiple irAEs", "name_count"),
    ("irae_type_multiplicity", "Patients With One vs Multiple irAE Organ Systems", "type_count"),
]
COUNT_SPECS = [
    ("irae_event_count_distribution", "Number of irAE Events per Patient", "event_count"),
    ("irae_name_count_distribution", "Number of Distinct irAEs per Patient", "name_count"),
    ("irae_type_count_distribution", "Number of Distinct irAE Organ Systems per Patient", "type_count"),
]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def irae_records(records):
    return [record for record in records if record.get("condition_type") == "irae"]


def patient_burdens(records):
    burdens = {}
    for record in irae_records(records):
        patient_id = record.get("patient_id")
        if not patient_id:
            continue

        entry = burdens.setdefault(
            patient_id,
            {
                "event_count": 0,
                "names": set(),
                "types": set(),
            },
        )
        entry["event_count"] += 1
        entry["names"].add(str(record.get("condition") or "Unknown"))
        entry["types"].add(str(record.get("irae_type") or "Unknown"))

    return {
        patient_id: {
            "event_count": entry["event_count"],
            "name_count": len(entry["names"]),
            "type_count": len(entry["types"]),
        }
        for patient_id, entry in burdens.items()
    }


def multiplicity_rows(burdens, count_field):
    denominator = len(burdens)
    one_count = sum(1 for entry in burdens.values() if entry[count_field] == 1)
    multiple_count = sum(1 for entry in burdens.values() if entry[count_field] > 1)

    return [
        ("One", one_count, 100 * one_count / denominator if denominator else 0),
        ("Multiple", multiple_count, 100 * multiple_count / denominator if denominator else 0),
    ], denominator


def count_distribution_rows(burdens, count_field, max_bin):
    denominator = len(burdens)
    counts = {}
    for entry in burdens.values():
        value = entry[count_field]
        label = f"{max_bin}+" if value >= max_bin else str(value)
        counts[label] = counts.get(label, 0) + 1

    rows = []
    for value in range(1, max_bin):
        label = str(value)
        count = counts.get(label, 0)
        rows.append((label, count, 100 * count / denominator if denominator else 0))

    label = f"{max_bin}+"
    count = counts.get(label, 0)
    rows.append((label, count, 100 * count / denominator if denominator else 0))
    return rows, denominator


def write_vertical_bar(rows, title, ylabel, output_path, dpi):
    if not rows:
        return False

    labels = [row[0] for row in rows]
    counts = [row[1] for row in rows]
    percents = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(labels, percents, color=BAR_COLOR, edgecolor="white", width=0.6)

    ax.set_ylim(0, max(100, max(percents) + 15))
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count, percent in zip(bars, counts, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{percent:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def export_charts(records, output_dir, max_count_bin, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    burdens = patient_burdens(records)
    denominator = len(burdens)

    for filename, title, count_field in MULTIPLICITY_SPECS:
        rows, _ = multiplicity_rows(burdens, count_field)
        output_path = output_dir / f"{filename}.png"
        if write_vertical_bar(
            rows,
            f"{title} (N={denominator})",
            "Percent of patients with any irAE",
            output_path,
            dpi=dpi,
        ):
            print(f"Wrote {output_path}")
        else:
            print(f"Skipped {title}: no data")

    for filename, title, count_field in COUNT_SPECS:
        rows, _ = count_distribution_rows(burdens, count_field, max_count_bin)
        output_path = output_dir / f"{filename}.png"
        if write_vertical_bar(
            rows,
            f"{title} (N={denominator})",
            "Percent of patients with any irAE",
            output_path,
            dpi=dpi,
        ):
            print(f"Wrote {output_path}")
        else:
            print(f"Skipped {title}: no data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export patient-level irAE burden bar charts."
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs/irae_multiplicity_bars")
    parser.add_argument("--max-count-bin", type=int, default=5, help="Collapse counts at this value into a plus bin.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    export_charts(
        records=records,
        output_dir=Path(args.output_dir),
        max_count_bin=args.max_count_bin,
        dpi=args.dpi,
    )
