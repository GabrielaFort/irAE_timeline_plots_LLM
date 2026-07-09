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


TARGET_IRAES = {
    "myocarditis": "Myocarditis",
    "myositis": "Myositis",
    "myasthenia gravis": "Myasthenia gravis",
}
COLUMN_X = {
    "Cancer": 0.05,
    "ICI class": 0.48,
    "Phenotype": 0.88,
}
NODE_WIDTH = 0.035
NODE_GAP = 0.02
FLOW_ALPHA = 0.35


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


def patient_records(records):
    by_patient = defaultdict(list)
    for record in records:
        patient_id = record.get("patient_id")
        if patient_id:
            by_patient[patient_id].append(record)
    return by_patient


def target_phenotype(records):
    phenotypes = set()
    for record in records:
        if record.get("condition_type") != "irae":
            continue
        condition = str(record.get("condition") or "").strip().lower()
        if condition in TARGET_IRAES:
            phenotypes.add(TARGET_IRAES[condition])

    if not phenotypes:
        return None
    return " + ".join(sorted(phenotypes))


def associated_ici_class(records):
    target_classes = []
    any_classes = []

    for record in records:
        if record.get("condition_type") != "irae":
            continue

        ici_class = record.get("associated_ici_class")
        if is_unknown(ici_class):
            continue

        any_classes.append(str(ici_class))
        condition = str(record.get("condition") or "").strip().lower()
        if condition in TARGET_IRAES:
            target_classes.append(str(ici_class))

    values = target_classes or any_classes
    if not values:
        return "Unknown"

    return " + ".join(sorted(set(values)))


def sankey_paths(records, cancer_field):
    paths = Counter()

    for events in patient_records(records).values():
        phenotype = target_phenotype(events)
        if phenotype is None:
            continue

        cancer = first_known_value(events, cancer_field)
        ici_class = associated_ici_class(events)
        paths[(cancer, ici_class, phenotype)] += 1

    return paths


def top_cancers(paths, top_n):
    if not top_n:
        return paths

    cancer_counts = Counter()
    for (cancer, _, _), count in paths.items():
        cancer_counts[cancer] += count

    kept = {cancer for cancer, _ in cancer_counts.most_common(top_n)}
    collapsed = Counter()
    for (cancer, ici_class, phenotype), count in paths.items():
        collapsed[(cancer if cancer in kept else "Other cancer types", ici_class, phenotype)] += count
    return collapsed


def link_counts(paths):
    links = Counter()
    for (cancer, ici_class, phenotype), count in paths.items():
        links[(("Cancer", cancer), ("ICI class", ici_class))] += count
        links[(("ICI class", ici_class), ("Phenotype", phenotype))] += count
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


def node_layout(values):
    layout = {}
    for column in COLUMN_X:
        column_nodes = [
            (node, value)
            for node, value in values.items()
            if node[0] == column
        ]
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
        band = min(source_band, target_band)

        source_y1 = source_offsets[source]
        source_y0 = source_y1 - band
        target_y1 = target_offsets[target]
        target_y0 = target_y1 - band
        source_offsets[source] = source_y0
        target_offsets[target] = target_y0

        segments.append((source, target, count, source_y0, source_y1, target_y0, target_y1))

    return segments


def node_color(node):
    column = node[0]
    return {
        "Cancer": "#4C78A8",
        "ICI class": "#F58518",
        "Phenotype": "#E45756",
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


def wrap_label(label, width=24):
    return textwrap.fill(str(label), width=width)


def write_sankey(paths, output_path, title, dpi):
    links = link_counts(paths)
    if not links:
        return False

    values = node_values(links)
    layout = node_layout(values)
    segments = link_segments(links, layout)

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for source, target, _, source_y0, source_y1, target_y0, target_y1 in segments:
        draw_flow(
            ax,
            layout[source]["x"],
            source_y0,
            source_y1,
            layout[target]["x"],
            target_y0,
            target_y1,
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
                edgecolor="white",
                linewidth=1,
            )
        )
        text_x = x - 0.01 if node[0] == "Phenotype" else x + NODE_WIDTH + 0.01
        ha = "right" if node[0] == "Phenotype" else "left"
        ax.text(
            text_x,
            (y0 + y1) / 2,
            f"{wrap_label(node[1])}\n(n={item['value']})",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
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
            "Export a patient-level Sankey-style PNG for cancer type -> associated ICI class -> "
            "myocarditis/myositis/myasthenia gravis phenotype."
        )
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output", default="outputs/tripleM_sankey.png")
    parser.add_argument("--cancer-field", default="oncotree_tissue")
    parser.add_argument("--top-cancers", type=int, default=12, help="Collapse lower-frequency cancer groups into Other.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    paths = top_cancers(
        sankey_paths(records, cancer_field=args.cancer_field),
        top_n=args.top_cancers,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if write_sankey(
        paths,
        output_path,
        "Cancer Type to ICI Class to Myocarditis/Myositis/Myasthenia Gravis Phenotype",
        dpi=args.dpi,
    ):
        print(f"Wrote {output_path}")
        print(f"Patients included: {sum(paths.values())}")
    else:
        print("Skipped Sankey: no patients with target irAEs found.")
