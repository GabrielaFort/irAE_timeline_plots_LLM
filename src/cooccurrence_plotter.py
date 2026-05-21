import plotly.graph_objects as go


COOCCURRENCE_OPTIONS = {
    "irAE": ("condition", "irae"),
    "irAE Type": ("irae_type", "irae"),
    "ICI": ("condition", "immunotherapy"),
    "ICI Class": ("ici_class", "immunotherapy"),
    "irAE treatment": ("condition", "irae_treatment"),
    "irAE Treatment Type": ("irae_treatment_type", "irae_treatment"),
}


def is_unknown(value):
    return value is None or str(value).strip().lower() in {"", "unknown", "none", "null", "na", "n/a"}


def split_combo_value(value):
    return [part.strip() for part in str(value).split("+") if part.strip()]


def event_values(event, field):
    value = event.get(field)
    if field in {"condition", "ici_class", "associated_ici", "associated_ici_class"}:
        return split_combo_value(value)
    return [value]


def patient_sets(events, field, condition_type=None, include_unknown=False):
    by_value = {}
    for event in events:
        if condition_type and event.get("condition_type") != condition_type:
            continue

        patient_id = event.get("patient_id")
        if not patient_id:
            continue

        for value in event_values(event, field):
            if is_unknown(value):
                if not include_unknown:
                    continue
                value = "Unknown"

            by_value.setdefault(str(value), set()).add(patient_id)

    return by_value


def top_patient_sets(by_value, top_n):
    return dict(
        sorted(
            by_value.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:top_n]
    )


def cooccurrence_matrices(by_value):
    labels = list(by_value)
    counts = []
    jaccard = []

    for row_label in labels:
        count_row = []
        jaccard_row = []
        for col_label in labels:
            overlap = len(by_value[row_label] & by_value[col_label])
            union = len(by_value[row_label] | by_value[col_label])
            count_row.append(overlap)
            jaccard_row.append(overlap / union if union else 0)
        counts.append(count_row)
        jaccard.append(jaccard_row)

    return labels, counts, jaccard


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(template="plotly_white", height=500)
    return fig


def make_cooccurrence_heatmap(events, field, condition_type=None, top_n=20, include_unknown=False):
    by_value = patient_sets(
        events,
        field=field,
        condition_type=condition_type,
        include_unknown=include_unknown,
    )
    by_value = top_patient_sets(by_value, top_n)
    labels, counts, jaccard = cooccurrence_matrices(by_value)

    if len(labels) < 2:
        return empty_figure("Fewer than two values found for this cohort.")

    fig = go.Figure(
        data=go.Heatmap(
            z=jaccard,
            x=labels,
            y=labels,
            text=counts,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "Jaccard"},
            hovertemplate=(
                "Row: %{y}<br>"
                "Column: %{x}<br>"
                "Patients with both: %{text}<br>"
                "Jaccard: %{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(560, 34 * len(labels) + 220),
        margin={"l": 160, "r": 40, "t": 40, "b": 160},
    )
    fig.update_xaxes(tickangle=45)
    return fig
