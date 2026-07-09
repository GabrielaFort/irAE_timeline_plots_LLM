import argparse
import json
import os
import tempfile
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

cache_dir = Path(tempfile.gettempdir())
os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(cache_dir)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIELD_SPECS = [
    ("irae_names", "irAE Name", "condition"),
    ("irae_types", "irAE Type", "irae_type"),
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


def patient_temporal_states(records, field):
    by_patient_time = defaultdict(lambda: defaultdict(set))

    for record in records:
        if record.get("condition_type") != "irae":
            continue

        patient_id = record.get("patient_id")
        time_start = parse_float(record.get("time_start"))
        value = record.get(field) or "Unknown"
        if not patient_id or time_start is None:
            continue

        by_patient_time[patient_id][time_start].add(str(value))

    pathways = {}
    for patient_id, by_time in by_patient_time.items():
        states = [
            " + ".join(sorted(values))
            for _, values in sorted(by_time.items(), key=lambda item: item[0])
        ]
        if states:
            pathways[patient_id] = tuple(states)

    return pathways


def transition_counts(pathways):
    counts = Counter()
    for pathway in pathways.values():
        for source, target in zip(pathway, pathway[1:]):
            counts[(source, target)] += 1
    return counts


def transition_labels(counts, max_labels):
    totals = Counter()
    for (source, target), count in counts.items():
        totals[source] += count
        totals[target] += count
    return [
        label
        for label, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:max_labels]
    ]


def wrap_label(value, width=48):
    return textwrap.fill(str(value), width=width)


def write_transition_heatmap(counts, labels, title, output_path, dpi):
    if not counts or len(labels) < 2:
        return False

    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for (source, target), count in counts.items():
        if source in matrix.index and target in matrix.columns:
            matrix.loc[source, target] = count

    if matrix.to_numpy().sum() == 0:
        return False

    size = max(8, min(16, 0.55 * len(labels) + 4))
    wrapped = [wrap_label(label, width=22) for label in labels]
    matrix.index = wrapped
    matrix.columns = wrapped

    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Patients"},
        ax=ax,
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("Next irAE state", fontsize=12, fontweight="bold")
    ax.set_ylabel("Previous irAE state", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def export_charts(records, output_dir, max_transition_labels, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, title_label, field in FIELD_SPECS:
        pathways = patient_temporal_states(records, field)
        counts = transition_counts(pathways)
        labels = transition_labels(counts, max_transition_labels)
        transition_output = output_dir / f"{filename}_transitions.png"
        transition_title = f"Adjacent irAE Transitions by {title_label}"
        if write_transition_heatmap(counts, labels, transition_title, transition_output, dpi=dpi):
            print(f"Wrote {transition_output}")
        else:
            print(f"Skipped {transition_title}: fewer than two transition states")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export patient-level irAE temporal pathway plots. Tied onset times are grouped "
            "into a single pathway state."
        )
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs/irae_temporal_pathways")
    parser.add_argument("--max-transition-labels", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    export_charts(
        records=records,
        output_dir=Path(args.output_dir),
        max_transition_labels=args.max_transition_labels,
        dpi=args.dpi,
    )
