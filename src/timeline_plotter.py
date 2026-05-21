import plotly.graph_objects as go


COLORS = {
    "immunotherapy": "#1f77b4",
    "irae": "#d62728",
    "irae_treatment": "#2ca02c",
}

TYPE_ORDER = {
    "immunotherapy": 0,
    "irae": 1,
    "irae_treatment": 2,
}

def make_plot(events):
    rows = []
    patient_id = None
    oncotree_code = None
    oncotree_name = None

    for event in events:
        patient_id = event.get("patient_id")
        oncotree_code = oncotree_code or event.get("oncotree_code")
        oncotree_name = oncotree_name or event.get("oncotree_name")
        ctype = str(event.get("condition_type", "")).strip().lower()
        condition = event.get("condition")
        time_start = event.get("time_start")

        if not patient_id or not ctype or not condition or time_start is None:
            continue

        rows.append(
            {
                "type": ctype,
                "condition": condition,
                "time_start": time_start,
            }
        )

    fig = go.Figure()
    title = f"Treatment and irAE Timeline: {patient_id}"
    if oncotree_code:
        title += f" | {oncotree_code}"
    if oncotree_name:
        title += f" ({oncotree_name})"

    if not rows:
        fig.update_layout(title=title)
        return fig

    label_start = {}
    label_type = {}
    for row in rows:
        label_start[row["condition"]] = min(
            label_start.get(row["condition"], row["time_start"]),
            row["time_start"],
        )
        label_type.setdefault(row["condition"], row["type"])

    order = [
        label
        for label, _ in sorted(label_start.items(), key=lambda item: item[1], reverse=True)
    ]
    order = sorted(
        order,
        key=lambda label: (
            TYPE_ORDER.get(label_type.get(label), 99),
            label_start[label],
        )
    )

    legend_shown = set()

    for ctype in sorted({row["type"] for row in rows}, key=lambda value: TYPE_ORDER.get(value, 99)):
        subset = [row for row in rows if row["type"] == ctype]
        fig.add_trace(
            go.Scatter(
                x=[row["time_start"] for row in subset],
                y=[row["condition"] for row in subset],
                mode="markers",
                marker={"size": 11, "color": COLORS.get(ctype, "#555"), "line": {"width": 1, "color": "white"}},
                name=ctype,
                legendgroup=ctype,
                showlegend=ctype not in legend_shown,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Type: " + ctype + "<br>"
                    "Month: %{x:.2f}<extra></extra>"
                ),
            )
        )
        legend_shown.add(ctype)

    fig.update_yaxes(
        title="Event",
        autorange="reversed",
        categoryorder="array",
        categoryarray=order,
    )
    fig.update_xaxes(title="Months from first event")
    fig.update_layout(
        title=title,
        template="plotly_white",
        legend_title_text="Event Type",
        height=max(420, 70 + 38 * len(label_start)),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig
