import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import plotly.graph_objects as go


COLUMNS = ["OncoTree tissue", "ICI class", "irAE Count"]
COLORS = {
    "OncoTree tissue": "rgba(76, 120, 168, 0.85)",
    "ICI class": "rgba(245, 133, 24, 0.85)",
    "irAE Count": "rgba(228, 87, 86, 0.85)",
}


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


def patient_records(records):
    by_patient = defaultdict(list)
    for record in records:
        patient_id = record.get("patient_id")
        if patient_id:
            by_patient[patient_id].append(record)
    return by_patient


def first_known_value(records, field):
    for record in records:
        value = record.get(field)
        if not is_unknown(value):
            return str(value)
    return "Unknown"


def all_ici_classes(records):
    classes = []
    for record in records:
        if record.get("condition_type") != "immunotherapy":
            continue
        if not is_unknown(record.get("ici_class")):
            classes.extend(split_combo_value(record["ici_class"]))
    return " + ".join(sorted(set(classes))) if classes else "Unknown"


def irae_count(records):
    iraes = {
        str(record.get("condition"))
        for record in records
        if record.get("condition_type") == "irae" and not is_unknown(record.get("condition"))
    }
    count = len(iraes)
    return f"{count} irAE" if count == 1 else f"{count} irAEs"


def patient_paths(records, cancer_field):
    paths = Counter()
    for events in patient_records(records).values():
        paths[
            (
                first_known_value(events, cancer_field),
                all_ici_classes(events),
                irae_count(events),
            )
        ] += 1
    return paths


def node_index(label, column, labels, keys, colors):
    key = (column, label)
    if key not in keys:
        keys[key] = len(labels)
        labels.append(label)
        colors.append(COLORS[column])
    return keys[key]


def sankey_figure(paths, title, width, height):
    labels = []
    keys = {}
    colors = []
    links = Counter()

    for (tissue, ici_class, count_label), patient_count in paths.items():
        tissue_idx = node_index(tissue, "OncoTree tissue", labels, keys, colors)
        ici_idx = node_index(ici_class, "ICI class", labels, keys, colors)
        count_idx = node_index(count_label, "irAE Count", labels, keys, colors)
        links[(tissue_idx, ici_idx)] += patient_count
        links[(ici_idx, count_idx)] += patient_count

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "color": colors,
                "pad": 18,
                "thickness": 18,
                "line": {"color": "rgba(0,0,0,0.35)", "width": 0.5},
            },
            link={
                "source": [source for source, _ in links],
                "target": [target for _, target in links],
                "value": list(links.values()),
            },
        )
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        font={"size": 11},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def write_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".html":
        fig.write_html(output_path)
        return
    try:
        fig.write_image(output_path)
    except ValueError as exc:
        raise RuntimeError(
            "Static Plotly image export requires kaleido. "
            "Install it in the plotting environment or use an .html output."
        ) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a Plotly patient-level Sankey for OncoTree tissue -> ICI class -> irAE count."
    )
    parser.add_argument("--input", default="data/patient_events_normalized.jsonl")
    parser.add_argument("--output", default="outputs/all_px_sankey_plotly.html")
    parser.add_argument("--cancer-field", default="oncotree_tissue")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    paths = patient_paths(records, cancer_field=args.cancer_field)
    fig = sankey_figure(
        paths,
        "OncoTree Tissue to ICI Class to irAE Count",
        width=args.width,
        height=args.height,
    )
    write_figure(fig, Path(args.output))
    print(f"Wrote {args.output}")
    print(f"Patients included: {sum(paths.values())}")
