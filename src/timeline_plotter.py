import plotly.graph_objects as go


COLORS = {
    "treatment": "#1f77b4",
    "irae": "#d62728",
    "disease": "#2ca02c",
}


def make_plot(events):
    rows = []
    patient_id = None

    for event in events:
        patient_id = event.get("patient_id")
        ctype = str(event.get("condition_type", "")).strip().lower()
        condition = event.get("condition")
        time_start = event.get("time_start")
        time_stop = event.get("time_stop")

        if not patient_id or not ctype or not condition or time_start is None:
            continue

        if time_stop is None or time_stop < time_start:
            time_stop = time_start

        rows.append(
            {
                "type": ctype,
                "condition": condition,
                "time_start": time_start,
                "time_stop": time_stop,
                "is_point": time_start == time_stop,
            }
        )

    fig = go.Figure()
    if not rows:
        fig.update_layout(title=f"Treatment and irAE Timeline: {patient_id}")
        return fig

    label_start = {}
    for row in rows:
        label_start[row["condition"]] = min(
            label_start.get(row["condition"], row["time_start"]),
            row["time_start"],
        )

    order = [
        label
        for label, _ in sorted(label_start.items(), key=lambda item: item[1], reverse=True)
    ]

    for ctype in sorted({row["type"] for row in rows if not row["is_point"]}):
        subset = [row for row in rows if row["type"] == ctype and not row["is_point"]]
        fig.add_trace(
            go.Bar(
                x=[row["time_stop"] - row["time_start"] for row in subset],
                y=[row["condition"] for row in subset],
                base=[row["time_start"] for row in subset],
                orientation="h",
                marker={"color": COLORS.get(ctype, "#555")},
                name=ctype,
                customdata=[[row["time_start"], row["time_stop"]] for row in subset],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Type: " + ctype + "<br>"
                    "Start month: %{customdata[0]:.2f}<br>"
                    "Stop month: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )

    for ctype in sorted({row["type"] for row in rows if row["is_point"]}):
        subset = [row for row in rows if row["type"] == ctype and row["is_point"]]
        fig.add_trace(
            go.Scatter(
                x=[row["time_start"] for row in subset],
                y=[row["condition"] for row in subset],
                mode="markers",
                marker={"size": 10, "color": COLORS.get(ctype, "#555")},
                name=f"{ctype} (point)",
                showlegend=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Type: " + ctype + "<br>"
                    "Month: %{x:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_yaxes(
        title="Event",
        autorange="reversed",
        categoryorder="array",
        categoryarray=order,
    )
    fig.update_xaxes(title="Months from first event")
    fig.update_layout(
        title=f"Treatment and irAE Timeline: {patient_id}",
        barmode="overlay",
        bargap=0.35,
        template="plotly_white",
        legend_title_text="Event Type",
        height=max(420, 70 + 38 * len(label_start)),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig
