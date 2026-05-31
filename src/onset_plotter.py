import random
import textwrap
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


WEEKS_PER_MONTH = 4.34524
POINT_COLOR = "#0072B2"
MEDIAN_COLOR = "#D55E00"

FACET_FIELDS = {
    "irAE Type": "irae_type",
    "OncoTree Tissue": "oncotree_tissue",
    "Full Treatment Regimen": "associated_treatment",
    "ICI Regimen": "associated_ici",
    "Treatment Category": "associated_therapy_type_consolidated",
    "Immunotherapy Class": "associated_ici_class",
}


def field_value(record, field):
    if field == "all":
        return "All"

    value = record.get(field)
    if value is None or str(value).strip().lower() in {"", "unknown", "none", "null", "na", "n/a"}:
        return "Unknown"
    return str(value)


def onset_value(record, unit):
    months = record.get("time_to_onset_months")
    if months is None:
        return None

    value = float(months)
    if unit == "weeks":
        return value * WEEKS_PER_MONTH
    return value


def median_value(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def top_values(records, field, max_values):
    counts = Counter(field_value(record, field) for record in records)
    return [value for value, _ in counts.most_common(max_values)]


def jitter_positions(count, seed):
    if count == 1:
        return [0]

    rng = random.Random(seed)
    return [rng.uniform(-0.18, 0.18) for _ in range(count)]


def wrapped_title(value, width=14):
    title = "<br>".join(textwrap.wrap(str(value), width=width)) or str(value)
    return f"<b>{title}</b>"


def onset_records(events):
    return [
        event
        for event in events
        if event.get("condition_type") == "irae" and event.get("time_to_onset_months") is not None
    ]


def onset_summary(events, row_field, col_field, unit, max_x, max_rows, max_cols):
    records = onset_records(events)
    row_values = top_values(records, row_field, max_rows)
    col_values = top_values(records, col_field, max_cols)
    rows = []

    for row_value in row_values:
        for col_value in col_values:
            values = [
                onset_value(record, unit)
                for record in records
                if field_value(record, row_field) == row_value and field_value(record, col_field) == col_value
            ]
            values = [value for value in values if value is not None and 0 <= value <= max_x]
            if not values:
                continue
            rows.append(
                {
                    "facet_row": row_value,
                    "facet_column": col_value,
                    "n": len(values),
                    f"median_onset_{unit}": round(median_value(values), 1),
                }
            )

    return pd.DataFrame(rows)


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(template="plotly_white", height=500)
    return fig


def make_onset_facets(events, row_field, col_field, unit="weeks", max_x=52, max_rows=6, max_cols=8):
    records = onset_records(events)
    if not records:
        return empty_figure("No irAE records with time-to-onset values found.")

    row_values = top_values(records, row_field, max_rows)
    col_values = top_values(records, col_field, max_cols)
    if not row_values or not col_values:
        return empty_figure("No facet values found.")

    fig = make_subplots(
        rows=len(row_values),
        cols=len(col_values),
        shared_xaxes=True,
        shared_yaxes=False,
        subplot_titles=[wrapped_title(value) for value in col_values],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for row_index, row_value in enumerate(row_values, start=1):
        for col_index, col_value in enumerate(col_values, start=1):
            values = [
                onset_value(record, unit)
                for record in records
                if field_value(record, row_field) == row_value and field_value(record, col_field) == col_value
            ]
            values = [value for value in values if value is not None and 0 <= value <= max_x]

            if values:
                y_values = jitter_positions(len(values), seed=f"{row_value}|{col_value}")
                median = median_value(values)
                fig.add_trace(
                    go.Scatter(
                        x=values,
                        y=y_values,
                        mode="markers",
                        marker={"color": POINT_COLOR, "opacity": 0.65, "size": 7},
                        name=f"{row_value} / {col_value}",
                        showlegend=False,
                        hovertemplate=(
                            f"{row_value}<br>{col_value}<br>"
                            f"Onset: %{{x:.1f}} {unit}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=col_index,
                )
                fig.add_vline(
                    x=median,
                    line_color=MEDIAN_COLOR,
                    line_dash="dash",
                    line_width=1.5,
                    row=row_index,
                    col=col_index,
                )
                label = f"median {median:.1f}<br>n={len(values)}"
            else:
                label = "n=0"

            fig.add_annotation(
                text=label,
                xref=f"x{'' if row_index == 1 and col_index == 1 else (row_index - 1) * len(col_values) + col_index} domain",
                yref=f"y{'' if row_index == 1 and col_index == 1 else (row_index - 1) * len(col_values) + col_index} domain",
                x=0.04,
                y=0.95,
                showarrow=False,
                align="left",
                font={"size": 10},
            )

            if col_index == 1 and row_field != "all":
                fig.update_yaxes(title_text=row_value, row=row_index, col=col_index)
    padding = 2
    fig.update_xaxes(range=[(0-padding), (max_x+padding)], title_text=f"Time to Onset ({unit.title()})")
    fig.update_yaxes(range=[-0.35, 0.35], showticklabels=False, zeroline=False)
    fig.update_layout(
        template="plotly_white",
        height=max(520, 210 * len(row_values)),
        margin={"l": 60 if row_field == "all" else 170, "r": 40, "t": 110, "b": 70},
    )
    fig.update_annotations(font={"size": 11})
    return fig
