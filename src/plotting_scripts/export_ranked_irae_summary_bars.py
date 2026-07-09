import argparse
import json
import os
import tempfile
import textwrap
from pathlib import Path

cache_dir = Path(tempfile.gettempdir())
os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(cache_dir)

import matplotlib.pyplot as plt

BAR_COLOR = "#0072B2"
FIG_SIZE = (12, 9)
GROUP_SPECS = [
    ("primary", "Primary irAEs", 1),
    ("secondary", "Secondary irAEs", 2),
    ("non_primary", "Non-primary irAEs", None),
]
FIELD_SPECS = [
    ("names", "irAE Names", "condition"),
    ("types", "irAE Types", "irae_type"),
]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def irae_episode_key(record):
    time_start = parse_float(record.get("time_start"))
    time_to_onset = parse_float(record.get("time_to_onset_months"))
    if time_start is None or time_to_onset is None:
        return None

    episode_start = round(time_start - time_to_onset, 2)
    return (
        record.get("patient_id"),
        episode_start,
        record.get("associated_ici"),
        record.get("associated_treatment"),
    )


def ranked_irae_records(records):
    iraes = []
    onsets_by_episode = {}

    for record in records:
        if record.get("condition_type") != "irae":
            continue

        key = irae_episode_key(record)
        onset = parse_float(record.get("time_to_onset_months"))
        if key is None or onset is None:
            continue

        iraes.append((key, onset, record))
        onsets_by_episode.setdefault(key, set()).add(onset)

    ranks_by_episode = {
        key: {onset: index + 1 for index, onset in enumerate(sorted(onsets))}
        for key, onsets in onsets_by_episode.items()
    }

    return [
        (ranks_by_episode[key][onset], record)
        for key, onset, record in iraes
    ]


def records_for_group(ranked_records, onset_rank):
    if onset_rank is None:
        return [record for rank, record in ranked_records if rank != 1]
    return [record for rank, record in ranked_records if rank == onset_rank]


def patient_sets(records, field):
    by_value = {}
    for record in records:
        patient_id = record.get("patient_id")
        if not patient_id:
            continue

        value = record.get(field) or "Unknown"
        by_value.setdefault(value, set()).add(patient_id)
    return by_value


def grouped_counts(by_value, denominator_patients, min_percent):
    kept = {}
    other_patients = set()

    for value, patients in by_value.items():
        percent = 100 * len(patients) / denominator_patients if denominator_patients else 0
        if percent >= min_percent:
            kept[value] = patients
        else:
            other_patients.update(patients)

    if other_patients:
        kept["Other"] = other_patients

    rows = [
        (value, len(patients), 100 * len(patients) / denominator_patients if denominator_patients else 0)
        for value, patients in kept.items()
    ]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


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
    ax.set_xlabel("Percent of group patients", fontsize=12, fontweight="bold")
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
    cohort_patients = len({record.get("patient_id") for record in records if record.get("patient_id")})
    ranked_records = ranked_irae_records(records)

    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name, group_title, onset_rank in GROUP_SPECS:
        group_records = records_for_group(ranked_records, onset_rank)
        group_patients = len({record.get("patient_id") for record in group_records if record.get("patient_id")})

        for field_name, field_title, field in FIELD_SPECS:
            rows = grouped_counts(
                patient_sets(group_records, field),
                denominator_patients=group_patients,
                min_percent=min_percent,
            )
            output_path = output_dir / f"{group_name}_irae_{field_name}.png"
            title = f"{group_title}: {field_title} (group N={group_patients}; cohort N={cohort_patients})"
            if write_bar(rows, title, output_path, dpi=dpi):
                print(f"Wrote {output_path}")
            else:
                print(f"Skipped {group_title}: {field_title}: no data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export patient-level bar charts for primary, secondary, and non-primary irAEs. "
            "Primary and secondary are based on distinct time-to-onset ranks within each ICI episode; "
            "ties are retained."
        )
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs/ranked_irae_summary_bars")
    parser.add_argument("--min-percent", type=float, default=0.0, help="Minimum group percentage to include.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    export_charts(
        records=records,
        output_dir=Path(args.output_dir),
        min_percent=args.min_percent,
        dpi=args.dpi,
    )
