import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.cooccurrence_plotter import COOCCURRENCE_OPTIONS, make_cooccurrence_heatmap
from streamlit_app.onset_plotter import FACET_FIELDS, make_onset_facets
from streamlit_app.timeline_plotter import make_plot


#DEFAULT_EVENTS_PATH = Path("data/patient_events_normalized.jsonl")
DEFAULT_EVENTS_PATH = Path("data/events_060926_gemma4_e4b_normalized_rxnorm.jsonl")
#DEFAULT_EVENTS_PATH = Path("data/events_060926_gemma4_e4b_normalized_rxnorm.jsonl")
#DEFAULT_EVENTS_PATH = Path("data/normalized_demo_data_for_grant_pneumonitis.jsonl")
ONCOTREE_DIR = Path("data/oncotree_tissues")

st.set_page_config(page_title="irAE Timelines", layout="wide", initial_sidebar_state="expanded")

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


@st.cache_data
def load_oncotree_branches(oncotree_dir):
    """Return OncoTree code -> branch code list from local OncoTree JSON files."""
    branches = {}
    decoder = json.JSONDecoder()

    for path in Path(oncotree_dir).glob("*.json"):
        if path.name.endswith("_oncotree_map.json"):
            continue

        text = path.read_text(encoding="utf-8")
        index = 0
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break

            obj, index = decoder.raw_decode(text, index)
            code = obj.get("oncotree_code")
            if code:
                branches[code] = obj.get("oncotree_branch_codes", [])

    return branches


def natural_sort_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def patient_field_value(events, field):
    for event in events:
        value = event.get(field)
        if value:
            return str(value)
    return "Unknown"


def oncotree_name_code_value(events):
    name = patient_field_value(events, "oncotree_name")
    code = patient_field_value(events, "oncotree_code")
    if name == "Unknown" and code == "Unknown":
        return "Unknown"
    return f"{name} | {code}"


def oncotree_code_from_value(value):
    if value == "Unknown" or " | " not in value:
        return None
    return value.rsplit(" | ", 1)[1]


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


def event_count(events, condition_type):
    return len(events_of_type(events, condition_type))


def filter_options(patient_events, field):
    if field == "patient_id":
        return ["All"] + sorted(patient_events, key=natural_sort_key)

    if field == "oncotree_name_code":
        values = {
            oncotree_name_code_value(events)
            for events in patient_events.values()
        }
        return ["All"] + sorted(values, key=natural_sort_key)

    if is_condition_filter(field):
        condition_type = field[1]
        values = {
            condition
            for events in patient_events.values()
            for condition in event_conditions(events, condition_type)
        }
        return ["All"] + sorted(values, key=natural_sort_key)

    if field in {"irae_type", "ici_class", "irae_treatment_type"}:
        condition_type = {
            "irae_type": "irae",
            "ici_class": "immunotherapy",
            "irae_treatment_type": "irae_treatment",
        }[field]
        values = {
            event.get(field) or "Unknown"
            for events in patient_events.values()
            for event in events_of_type(events, condition_type)
        }
        return ["All"] + sorted(values, key=natural_sort_key)

    values = {
        patient_field_value(events, field)
        for events in patient_events.values()
    }
    return ["All"] + sorted(values, key=natural_sort_key)


def matching_patient_ids(patient_events, field, value, oncotree_branches=None):
    if value == "All":
        return sorted(patient_events, key=natural_sort_key)

    if field == "patient_id":
        return [value]

    if field == "oncotree_name_code":
        selected_code = oncotree_code_from_value(value)
        return [
            patient_id
            for patient_id, events in sorted(patient_events.items(), key=lambda item: natural_sort_key(item[0]))
            if (
                oncotree_name_code_value(events) == value
                or (
                    selected_code
                    and selected_code in (oncotree_branches or {}).get(patient_field_value(events, "oncotree_code"), [])
                )
            )
        ]

    if is_condition_filter(field):
        condition_type = field[1]
        return [
            patient_id
            for patient_id, events in sorted(patient_events.items(), key=lambda item: natural_sort_key(item[0]))
            if value in event_conditions(events, condition_type)
        ]

    if field in {"irae_type", "ici_class", "irae_treatment_type"}:
        condition_type = {
            "irae_type": "irae",
            "ici_class": "immunotherapy",
            "irae_treatment_type": "irae_treatment",
        }[field]
        return [
            patient_id
            for patient_id, events in sorted(patient_events.items(), key=lambda item: natural_sort_key(item[0]))
            if any((event.get(field) or "Unknown") == value for event in events_of_type(events, condition_type))
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


def show_field_summary(title, events, field, total_patients, value_label=None):
    st.subheader(title)
    summary = field_summary(events, field, total_patients)

    if summary.empty:
        st.info("No values found.")
        return

    display_field = value_label or field
    plot_summary = summary.rename(columns={field: display_field})

    fig = px.bar(
        plot_summary,
        x=display_field,
        y="patient_count",
        hover_data=["patient_percent"],
    )
    fig.update_layout(
        xaxis={"categoryorder": "array", "categoryarray": plot_summary[display_field].tolist()},
        xaxis_title=None,
        yaxis_title="Patients",
        template="plotly_white",
        height=460,
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")
    with st.expander("See summary table"):
        st.dataframe(plot_summary, width="stretch", hide_index=True)


st.title("irAE Timeline Explorer")

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

oncotree_branches = load_oncotree_branches(ONCOTREE_DIR)

filter_fields = {
    "Patient": "patient_id",
    "irAE": ("condition", "irae"),
    "irAE Type": "irae_type",
    "Immunotherapy": ("condition", "immunotherapy"),
    "Immunotherapy Class": "ici_class",
    "OncoTree Tissue": "oncotree_tissue",
    "OncoTree Name | Code": "oncotree_name_code",
    "irAE Treatment Type": "irae_treatment_type",
}
with st.sidebar:
    st.header("Cohort Filter")
    filter_label = st.radio(
        "Filter by",
        options=list(filter_fields),
    )
filter_field = filter_fields[filter_label]

with st.sidebar:
    selected_value = st.selectbox(
        filter_label,
        options=filter_options(patient_events, filter_field),
    )
selected_patient_ids = matching_patient_ids(
    patient_events,
    filter_field,
    selected_value,
    oncotree_branches=oncotree_branches,
)

selected_events = [
    event
    for patient_id in selected_patient_ids
    for event in patient_events[patient_id]
]

with st.sidebar:
    st.divider()
    st.metric("Patients", len(selected_patient_ids))
    st.metric("Events", len(selected_events))

selected_label = "All patients" if selected_value == "All" else f"{filter_label}: {selected_value}"
st.caption(f"Selected cohort: {selected_label}")
metric_cols = st.columns(5)
metric_cols[0].metric("Patients", len(selected_patient_ids))
metric_cols[1].metric("Events", len(selected_events))
metric_cols[2].metric("irAEs", event_count(selected_events, "irae"))
metric_cols[3].metric("ICIs", event_count(selected_events, "immunotherapy"))
metric_cols[4].metric("irAE Treatments", event_count(selected_events, "irae_treatment"))

plots_tab, summary_tab, cooccurrence_tab, onset_tab, table_tab = st.tabs(
    ["Patient Timelines", "Cohort Summary", "Co-occurrence", "Time to Onset", "Data Table"]
)

with plots_tab:
    for index, patient_id in enumerate(selected_patient_ids):
        if index > 0:
            st.divider()
        events = patient_events[patient_id]
        fig = make_plot(events)
        plotly_config = {"displayModeBar": True,
                                "scrollZoom": True,
                                "responsive": True,
                                "editable": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "timeline_plot",
                                    "scale": 5
                                }}
        st.plotly_chart(fig, width="stretch", config=plotly_config)


with table_tab:
    hidden_columns = {"source_file", "raw_condition"}
    events_display = [
        {key: value for key, value in event.items() if key not in hidden_columns}
        for event in selected_events
    ]
    st.dataframe(events_display, width="stretch", hide_index=True)

with summary_tab:
    irae_events = events_of_type(selected_events, "irae")
    immunotherapy_events = events_of_type(selected_events, "immunotherapy")
    irae_treatment_events = events_of_type(selected_events, "irae_treatment")

    st.header("Cancer")
    show_field_summary("OncoTree Tissues", selected_events, "oncotree_tissue", len(selected_patient_ids))
    show_field_summary("OncoTree Names", selected_events, "oncotree_name", len(selected_patient_ids))

    st.header("irAEs")
    show_field_summary("irAE Types", irae_events, "irae_type", len(selected_patient_ids))
    show_condition_summary("irAEs", selected_events, "irae", len(selected_patient_ids))

    st.header("Treatment")
    show_condition_summary("Full Normalized Treatment Regimens", selected_events, "immunotherapy", len(selected_patient_ids))
    show_field_summary("ICI Regimens", immunotherapy_events, "ici_combo", len(selected_patient_ids), value_label="ICI regimen")
    show_field_summary(
        "Treatment Categories",
        immunotherapy_events,
        "therapy_type_consolidated",
        len(selected_patient_ids),
        value_label="Treatment category",
    )
    show_field_summary("ICI Classes", immunotherapy_events, "ici_class", len(selected_patient_ids))
    show_condition_summary("irAE Treatments", selected_events, "irae_treatment", len(selected_patient_ids))
    show_field_summary("irAE Treatment Types", irae_treatment_events, "irae_treatment_type", len(selected_patient_ids))

with cooccurrence_tab:
    st.subheader("Co-occurrence Heatmap")
    option_col, top_col, primary_col = st.columns([3, 1, 1])
    with option_col:
        cooccurrence_label = st.selectbox(
            "Label type",
            options=list(COOCCURRENCE_OPTIONS),
            index=0,
        )
    with top_col:
        cooccurrence_top_n = st.slider("Top N", min_value=5, max_value=30, value=15)
    with primary_col:
        cooccurrence_primary_only = st.checkbox("Primary irAEs", value=False, key="cooccurrence_primary_only")

    heatmap_field, heatmap_condition_type = COOCCURRENCE_OPTIONS[cooccurrence_label]
    heatmap_fig = make_cooccurrence_heatmap(
        selected_events,
        field=heatmap_field,
        condition_type=heatmap_condition_type,
        top_n=cooccurrence_top_n,
        include_unknown=False,
        primary_only=cooccurrence_primary_only,
    )
    st.plotly_chart(heatmap_fig, width="stretch")

with onset_tab:
    st.subheader("Time to irAE Onset")
    facet_col, limit_col, primary_col = st.columns([3, 1, 1])
    with facet_col:
        col_label = st.selectbox(
            "Facet by",
            options=list(FACET_FIELDS),
            index=list(FACET_FIELDS).index("irAE Type"),
        )
    with limit_col:
        max_cols = st.slider("Max columns", min_value=1, max_value=12, value=8)
    with primary_col:
        onset_primary_only = st.checkbox("Primary irAEs", value=False, key="onset_primary_only")

    row_field = "all"
    col_field = FACET_FIELDS[col_label]
    onset_fig = make_onset_facets(
        selected_events,
        row_field=row_field,
        col_field=col_field,
        unit="weeks",
        max_x=52,
        max_rows=1,
        max_cols=max_cols,
        primary_only=onset_primary_only,
    )
    st.plotly_chart(onset_fig, width="stretch")
