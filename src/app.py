import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from timeline_plotter import make_plot


DEFAULT_EVENTS_PATH = Path("data/patient_events_normalized.jsonl")

st.set_page_config(page_title="irAE Timelines", layout="wide", initial_sidebar_state="collapsed")

@st.cache_data
def load_patient_events(events_path):
    """Load event-level JSONL and group records by patient_id."""
    path = Path(events_path)
    patients = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e

            patient_id = record.get("patient_id") or "unknown"
            patients.setdefault(patient_id, []).append(record)

    return patients


def natural_sort_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def patient_field_value(events, field):
    for event in events:
        value = event.get(field)
        if value:
            return str(value)
    return "Unknown"


def is_condition_filter(field):
    return isinstance(field, tuple)


def event_conditions(events, condition_type):
    return {
        event.get("condition") or "Unknown"
        for event in events
        if event.get("condition_type") == condition_type
    }


def events_of_type(events, condition_type):
    return [event for event in events if event.get("condition_type") == condition_type]


def filter_options(patient_events, field):
    if field == "patient_id":
        return ["All"] + sorted(patient_events, key=natural_sort_key)

    if is_condition_filter(field):
        condition_type = field[1]
        values = {
            condition
            for events in patient_events.values()
            for condition in event_conditions(events, condition_type)
        }
        return ["All"] + sorted(values, key=natural_sort_key)

    if field == "irae_type":
        values = {
            event.get("irae_type") or "Unknown"
            for events in patient_events.values()
            for event in events_of_type(events, "irae")
        }
        return ["All"] + sorted(values, key=natural_sort_key)

    values = {
        patient_field_value(events, field)
        for events in patient_events.values()
    }
    return ["All"] + sorted(values, key=natural_sort_key)


def matching_patient_ids(patient_events, field, value):
    if value == "All":
        return sorted(patient_events, key=natural_sort_key)

    if field == "patient_id":
        return [value]

    if is_condition_filter(field):
        condition_type = field[1]
        return [
            patient_id
            for patient_id, events in sorted(patient_events.items(), key=lambda item: natural_sort_key(item[0]))
            if value in event_conditions(events, condition_type)
        ]

    if field == "irae_type":
        return [
            patient_id
            for patient_id, events in sorted(patient_events.items(), key=lambda item: natural_sort_key(item[0]))
            if any((event.get("irae_type") or "Unknown") == value for event in events_of_type(events, "irae"))
        ]

    return [
        patient_id
        for patient_id, events in sorted(patient_events.items(), key=lambda item: natural_sort_key(item[0]))
        if patient_field_value(events, field) == value
    ]


def condition_summary(events, condition_type, total_patients):
    counts = {}
    patients = {}

    for event in events:
        if event.get("condition_type") != condition_type:
            continue

        condition = event.get("condition") or "Unknown"
        counts[condition] = counts.get(condition, 0) + 1
        patients.setdefault(condition, set()).add(event.get("patient_id"))

    rows = [
        {
            "condition": condition,
            "event_count": counts[condition],
            "patient_count": len(patients[condition]),
            "patient_percent": round(100 * len(patients[condition]) / total_patients, 1)
            if total_patients else 0,
        }
        for condition in counts
    ]
    if not rows:
        return pd.DataFrame(columns=["condition", "event_count", "patient_count", "patient_percent"])

    return pd.DataFrame(rows).sort_values(
        ["patient_count", "event_count", "condition"],
        ascending=[False, False, True],
    )


def show_condition_summary(title, events, condition_type, total_patients):
    st.subheader(title)
    summary = condition_summary(events, condition_type, total_patients)

    if summary.empty:
        st.info("No events found.")
        return

    fig = px.bar(
        summary,
        x="condition",
        y="patient_count",
        hover_data=["event_count", "patient_percent"],
    )
    fig.update_layout(
        xaxis={"categoryorder": "array", "categoryarray": summary["condition"].tolist()},
        xaxis_title=None,
        yaxis_title="Patients",
        template="plotly_white",
        height=460,
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")
    with st.expander("See summary table"):
        st.dataframe(summary, width="stretch", hide_index=True)


def field_summary(events, field, total_patients):
    patients = {}

    for event in events:
        value = event.get(field) or "Unknown"
        patients.setdefault(value, set()).add(event.get("patient_id"))

    rows = [
        {
            field: value,
            "patient_count": len(patient_ids),
            "patient_percent": round(100 * len(patient_ids) / total_patients, 1)
            if total_patients else 0,
        }
        for value, patient_ids in patients.items()
    ]
    if not rows:
        return pd.DataFrame(columns=[field, "patient_count", "patient_percent"])

    return pd.DataFrame(rows).sort_values(
        ["patient_count", field],
        ascending=[False, True],
    )


def show_field_summary(title, events, field, total_patients):
    st.subheader(title)
    summary = field_summary(events, field, total_patients)

    if summary.empty:
        st.info("No values found.")
        return

    fig = px.bar(
        summary,
        x=field,
        y="patient_count",
        hover_data=["patient_percent"],
    )
    fig.update_layout(
        xaxis={"categoryorder": "array", "categoryarray": summary[field].tolist()},
        xaxis_title=None,
        yaxis_title="Patients",
        template="plotly_white",
        height=460,
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")
    with st.expander("See summary table"):
        st.dataframe(summary, width="stretch", hide_index=True)


st.title("irAE Treatment Timelines Explorer")

path = DEFAULT_EVENTS_PATH 

if not path.exists():
    st.error(f"Could not find events file: {path}")
    st.stop()

try:
    patient_events = load_patient_events(str(path))
except Exception as e:
    st.error(f"Could not load events JSONL: {e}")
    st.stop()

if not patient_events:
    st.info("No patient events found in the JSONL file.")
    st.stop()

filter_fields = {
    "Patient": "patient_id",
    "OncoTree Name": "oncotree_name",
    "OncoTree Code": "oncotree_code",
    "OncoTree Tissue": "oncotree_tissue",
    "irAE Type": "irae_type",
    "Immunotherapy": ("condition", "immunotherapy"),
    "irAE": ("condition", "irae"),
    "irAE Treatment": ("condition", "irae_treatment"),
}
filter_col, value_col = st.columns([2, 3])
with filter_col:
    st.markdown("**Filter by**")
    filter_label = st.radio(
        "Filter by",
        options=list(filter_fields),
        horizontal=True,
        label_visibility="collapsed",
    )
filter_field = filter_fields[filter_label]
with value_col:
    st.markdown(f"**{filter_label}**")
    selected_value = st.selectbox(
        filter_label,
        options=filter_options(patient_events, filter_field),
        label_visibility="collapsed",
    )
selected_patient_ids = matching_patient_ids(patient_events, filter_field, selected_value)

selected_events = [
    event
    for patient_id in selected_patient_ids
    for event in patient_events[patient_id]
]

plots_tab, table_tab, summary_tab = st.tabs(["Timeline Plots", "Table", "Summary"])

with plots_tab:
    for index, patient_id in enumerate(selected_patient_ids):
        if index > 0:
            st.divider()
        events = patient_events[patient_id]
        fig = make_plot(events)
        st.plotly_chart(fig, width="stretch")

with table_tab:
    hidden_columns = {"source_file", "raw_condition"}
    events_display = [
        {key: value for key, value in event.items() if key not in hidden_columns}
        for event in selected_events
    ]
    st.dataframe(events_display, width="stretch", hide_index=True)

with summary_tab:
    irae_events = events_of_type(selected_events, "irae")
    st.metric("Patients", len(selected_patient_ids))
    show_field_summary("OncoTree Tissues", selected_events, "oncotree_tissue", len(selected_patient_ids))
    show_field_summary("OncoTree Names", selected_events, "oncotree_name", len(selected_patient_ids))
    show_field_summary("irAE Types", irae_events, "irae_type", len(selected_patient_ids))
    show_condition_summary("irAEs", selected_events, "irae", len(selected_patient_ids))
    show_condition_summary("ICIs", selected_events, "immunotherapy", len(selected_patient_ids))
    show_condition_summary("irAE Treatments", selected_events, "irae_treatment", len(selected_patient_ids))
