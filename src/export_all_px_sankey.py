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
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


COLUMN_X = {
    "OncoTree tissue": 0.05,
    "ICI class": 0.48,
    "irAE Count": 0.88,
}
NODE_WIDTH = 0.035
NODE_GAP = 0.02
FLOW_ALPHA = 0.35
MIN_FLOW_POINTS = 1.6


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


def first_known_value(records, field):
    for record in records:
        value = record.get(field)
        if not is_unknown(value):
            return str(value)
    return "Unknown"


def split_combo_value(value):
    return [part.strip() for part in str(value).split("+") if part.strip()]


def patient_records(records):
    by_patient = defaultdict(list)
    for record in records:
        patient_id = record.get("patient_id")
        if patient_id:
            by_patient[patient_id].append(record)
    return by_patient


def distinct_irae_count(records):
    iraes = set()
    for record in records:
        if record.get("condition_type") != "irae":
            continue
        condition = record.get("condition")
        if not is_unknown(condition):
            iraes.add(str(condition))

    count = len(iraes)
    if count == 0:
        return "0 irAEs"
    if count == 1:
        return "1 irAE"
    return "Multiple irAEs"


def all_ici_classes(records):
    classes = []

    for record in records:
        if record.get("condition_type") != "immunotherapy":
            continue

        ici_class = record.get("ici_class")
        if is_unknown(ici_class):
            continue

        classes.extend(split_combo_value(ici_class))

    if not classes:
        return "Unknown"

    return " + ".join(sorted(set(classes)))


def sankey_paths(records, cancer_field, min_tissue_patients):
    paths = Counter()
    patients = patient_records(records)
    tissue_counts = Counter(first_known_value(events, cancer_field) for events in patients.values())

    for events in patients.values():
        tissue = first_known_value(events, cancer_field)
        if tissue_counts[tissue] < min_tissue_patients:
            tissue = "Other"
        ici_class = all_ici_classes(events)
        irae_count = distinct_irae_count(events)
        paths[(tissue, ici_class, irae_count)] += 1

    return paths


def link_counts(paths):
    links = Counter()
    for (tissue, ici_class, irae_count), count in paths.items():
        links[(("OncoTree tissue", tissue), ("ICI class", ici_class))] += count
        links[(("ICI class", ici_class), ("irAE Count", irae_count))] += count
    return links


def node_values(links):
    incoming = Counter()
    outgoing = Counter()
    for source, target in links:
        count = links[(source, target)]
        outgoing[source] += count
        incoming[target] += count

    nodes = set(incoming) | set(outgoing)
    return {node: max(incoming[node], outgoing[node]) for node in nodes}


def irae_count_sort_key(node):
    return {
        "0 irAEs": 0,
        "1 irAE": 1,
        "Multiple irAEs": 2,
    }.get(node[1], 999)


def node_layout(values):
    layout = {}
    for column in COLUMN_X:
        column_nodes = [
            (node, value)
            for node, value in values.items()
            if node[0] == column
        ]
        if column == "irAE Count":
            column_nodes.sort(key=lambda item: irae_count_sort_key(item[0]))
        else:
            column_nodes.sort(key=lambda item: (-item[1], item[0][1]))

        total = sum(value for _, value in column_nodes)
        if not total:
            continue

        available = 1.0 - NODE_GAP * max(0, len(column_nodes) - 1)
        y_top = 1.0
        for node, value in column_nodes:
            height = available * value / total
            y_bottom = y_top - height
            layout[node] = {
                "x": COLUMN_X[column],
                "y0": y_bottom,
                "y1": y_top,
                "value": value,
            }
            y_top = y_bottom - NODE_GAP
    return layout


def link_segments(links, layout):
    source_offsets = {node: layout[node]["y1"] for node in layout}
    target_offsets = {node: layout[node]["y1"] for node in layout}
    segments = []

    for (source, target), count in sorted(
        links.items(),
        key=lambda item: (item[0][0][0], item[0][0][1], item[0][1][1]),
    ):
        source_height = layout[source]["y1"] - layout[source]["y0"]
        target_height = layout[target]["y1"] - layout[target]["y0"]
        source_band = source_height * count / layout[source]["value"]
        target_band = target_height * count / layout[target]["value"]

        source_y1 = source_offsets[source]
        source_y0 = source_y1 - source_band
        target_y1 = target_offsets[target]
        target_y0 = target_y1 - target_band
        source_offsets[source] = source_y0
        target_offsets[target] = target_y0

        segments.append((source, target, count, source_y0, source_y1, target_y0, target_y1))

    return segments


def node_color(node):
    column = node[0]
    return {
        "OncoTree tissue": "#4C78A8",
        "ICI class": "#F58518",
        "irAE Count": "#E45756",
    }.get(column, "#777777")

def draw_flow(ax, source_x, source_y0, source_y1, target_x, target_y0, target_y1, color):
    x0 = source_x + NODE_WIDTH
    x1 = target_x
    curve = (x1 - x0) * 0.45
    vertices = [
        (x0, source_y0),
        (x0 + curve, source_y0),
        (x1 - curve, target_y0),
        (x1, target_y0),
        (x1, target_y1),
        (x1 - curve, target_y1),
        (x0 + curve, source_y1),
        (x0, source_y1),
        (x0, source_y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor=color,
        edgecolor="none",
        alpha=FLOW_ALPHA,
    )
    ax.add_patch(patch)


def visible_band(y0, y1, min_band):
    if y1 - y0 >= min_band:
        return y0, y1

    midpoint = (y0 + y1) / 2
    half_band = min_band / 2
    return max(0, midpoint - half_band), min(1, midpoint + half_band)


def wrap_label(label, width=24):
    return textwrap.fill(str(label), width=width)


def write_sankey(paths, output_path, title, dpi, width, height):
    links = link_counts(paths)
    if not links:
        return False

    values = node_values(links)
    layout = node_layout(values)
    segments = link_segments(links, layout)
    min_band = MIN_FLOW_POINTS / (height * dpi)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for source, target, _, source_y0, source_y1, target_y0, target_y1 in segments:
        visible_source_y0, visible_source_y1 = visible_band(source_y0, source_y1, min_band)
        visible_target_y0, visible_target_y1 = visible_band(target_y0, target_y1, min_band)
        draw_flow(
            ax,
            layout[source]["x"],
            visible_source_y0,
            visible_source_y1,
            layout[target]["x"],
            visible_target_y0,
            visible_target_y1,
            node_color(source),
        )

    for node, item in layout.items():
        x = item["x"]
        y0 = item["y0"]
        y1 = item["y1"]
        color = node_color(node)
        ax.add_patch(
            Rectangle(
                (x, y0),
                NODE_WIDTH,
                y1 - y0,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
            )
        )
        text_x = x - 0.01 if node[0] == "irAE Count" else x + NODE_WIDTH + 0.01
        ha = "right" if node[0] == "irAE Count" else "left"
        ax.text(
            text_x,
            (y0 + y1) / 2,
            wrap_label(node[1]),
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
            linespacing=0.85,
        )

    for column, x in COLUMN_X.items():
        ax.text(
            x + NODE_WIDTH / 2,
            1.04,
            column,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export a patient-level Sankey-style PNG for OncoTree tissue -> ICI class -> "
            "irAE count."
        )
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output", default="outputs/all_px_sankey.png")
    parser.add_argument("--cancer-field", default="oncotree_tissue")
    parser.add_argument("--min-tissue-patients", type=int, default=10)
    parser.add_argument("--width", type=float, default=18, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=18, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    paths = sankey_paths(
        records,
        cancer_field=args.cancer_field,
        min_tissue_patients=args.min_tissue_patients,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if write_sankey(
        paths,
        output_path,
        "OncoTree Tissue to ICI Class to irAE Count",
        dpi=args.dpi,
        width=args.width,
        height=args.height,
    ):
        print(f"Wrote {output_path}")
        print(f"Patients included: {sum(paths.values())}")
    else:
        print("Skipped Sankey: no patient paths found.")
