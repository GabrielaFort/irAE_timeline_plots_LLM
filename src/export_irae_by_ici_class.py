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
import pandas as pd
import seaborn as sns


PLOT_SPECS = [
    ("irae_names_by_ici_class", "irAE Names by Associated ICI Class", "condition", "names", "associated_ici_class", True, "Associated ICI class", "Percent of ICI-class patients", None),
    ("irae_types_by_ici_class", "irAE Types by Associated ICI Class", "irae_type", "types", "associated_ici_class", True, "Associated ICI class", "Percent of ICI-class patients", None),
    ("irae_names_by_oncotree_tissue", "Top irAE Names by OncoTree Tissue", "condition", "names", "oncotree_tissue", False, "OncoTree tissue", "Percent of tissue patients", "tissues"),
    ("irae_types_by_oncotree_tissue", "irAE Types by OncoTree Tissue", "irae_type", "types", "oncotree_tissue", False, "OncoTree tissue", "Percent of tissue patients", "tissues"),
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


def group_values(record, group_field, split_group):
    value = record.get(group_field)
    if split_group:
        return split_combo_value(value)
    return [value]


def patient_sets_by_group(records, field, group_field, split_group=False, include_unknown=False):
    group_patients = {}
    term_group_patients = {}

    for record in records:
        if record.get("condition_type") != "irae":
            continue

        patient_id = record.get("patient_id")
        if not patient_id:
            continue

        term = record.get(field)
        if is_unknown(term):
            if not include_unknown:
                continue
            term = "Unknown"

        values = group_values(record, group_field, split_group)
        if not values and include_unknown:
            values = ["Unknown"]

        for group_value in values:
            if is_unknown(group_value):
                if not include_unknown:
                    continue
                group_value = "Unknown"

            group_value = str(group_value)
            group_patients.setdefault(group_value, set()).add(patient_id)
            term_group_patients.setdefault(str(term), {}).setdefault(group_value, set()).add(patient_id)

    return group_patients, term_group_patients


def filtered_terms(term_group_patients, min_patients, top_n=None):
    terms = []
    for term, by_group in term_group_patients.items():
        patients = set()
        for group_patients in by_group.values():
            patients.update(group_patients)
        if len(patients) >= min_patients:
            terms.append((term, len(patients)))

    terms = [
        term
        for term, _ in sorted(terms, key=lambda item: (-item[1], item[0]))
    ]
    return terms[:top_n] if top_n else terms


def filtered_groups(group_patients, min_group_patients, top_groups=None):
    groups = [
        (group, len(patients))
        for group, patients in group_patients.items()
        if len(patients) >= min_group_patients
    ]
    groups = [
        group
        for group, _ in sorted(groups, key=lambda item: (-item[1], item[0]))
    ]
    return groups[:top_groups] if top_groups else groups


def heatmap_matrices(group_patients, term_group_patients, groups, terms):
    percent_rows = []
    count_rows = []

    for group in groups:
        denominator = len(group_patients[group])
        percent_row = []
        count_row = []
        for term in terms:
            count = len(term_group_patients.get(term, {}).get(group, set()))
            percent_row.append(100 * count / denominator if denominator else 0)
            count_row.append(count)
        percent_rows.append(percent_row)
        count_rows.append(count_row)

    return percent_rows, count_rows


def sorted_groups(group_patients):
    return [
        group
        for group, _ in sorted(
            group_patients.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]


def wrap_label(value, width=24):
    return textwrap.fill(str(value), width=width)


def write_heatmap(groups, terms, percent_matrix, count_matrix, title, group_axis_label, colorbar_label, output_path, dpi):
    if len(groups) < 1 or len(terms) < 1:
        return False

    wrapped_groups = [wrap_label(value, width=20) for value in groups]
    wrapped_terms = [wrap_label(value, width=22) for value in terms]
    percents = pd.DataFrame(percent_matrix, index=wrapped_groups, columns=wrapped_terms)
    counts = pd.DataFrame(count_matrix, index=wrapped_groups, columns=wrapped_terms)
    annotations = counts.astype(str) + "\n(" + percents.round(1).astype(str) + "%)"

    width = max(10, min(24, 0.45 * len(terms) + 5))
    height = max(5, min(18, 0.55 * len(groups) + 3))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        percents,
        annot=annotations,
        fmt="",
        cmap="Reds",
        vmin=0,
        vmax=max(100, float(percents.to_numpy().max()) if not percents.empty else 0),
        linewidths=0.5,
        cbar_kws={"label": colorbar_label},
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel(None)
    ax.set_ylabel(group_axis_label, fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def export_heatmaps(records, output_dir, min_patients, min_group_patients, top_names, top_tissues, dpi, include_unknown):
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, title, field, plot_type, group_field, split_group, group_axis_label, colorbar_label, group_limit_type in PLOT_SPECS:
        group_patients, term_group_patients = patient_sets_by_group(
            records,
            field=field,
            group_field=group_field,
            split_group=split_group,
            include_unknown=include_unknown,
        )
        top_n = top_names if plot_type == "names" else None
        top_groups = top_tissues if group_limit_type == "tissues" else None
        terms = filtered_terms(term_group_patients, min_patients=min_patients, top_n=top_n)
        groups = filtered_groups(
            group_patients,
            min_group_patients=min_group_patients,
            top_groups=top_groups,
        )
        percent_matrix, count_matrix = heatmap_matrices(
            group_patients,
            term_group_patients,
            groups,
            terms,
        )
        output_path = output_dir / f"{filename}.png"
        if write_heatmap(groups, terms, percent_matrix, count_matrix, title, group_axis_label, colorbar_label, output_path, dpi=dpi):
            print(f"Wrote {output_path}")
        else:
            print(f"Skipped {title}: no data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export patient-level irAE and irAE type distributions by associated ICI class."
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs/irae_by_ici_class")
    parser.add_argument("--min-patients", type=int, default=1, help="Minimum patients required for an irAE term to be shown.")
    parser.add_argument("--min-group-patients", type=int, default=1, help="Minimum patients required for a row group to be shown.")
    parser.add_argument("--top-names", type=int, default=20, help="Maximum number of irAE names to show.")
    parser.add_argument("--top-tissues", type=int, default=8, help="Maximum number of OncoTree tissue rows to show.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-unknown", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    export_heatmaps(
        records=records,
        output_dir=Path(args.output_dir),
        min_patients=args.min_patients,
        min_group_patients=args.min_group_patients,
        top_names=args.top_names,
        top_tissues=args.top_tissues,
        dpi=args.dpi,
        include_unknown=args.include_unknown,
    )
