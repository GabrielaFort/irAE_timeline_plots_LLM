import json
import re
from pathlib import Path

import streamlit as st

from timeline_plotter import make_plot


DEFAULT_EVENTS_PATH = Path("data/patient_events_normalized.jsonl")


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


st.set_page_config(page_title="irAE Timelines", layout="wide", initial_sidebar_state="expanded")
st.title("Patient Timelines")

events_path = st.sidebar.text_input("Events JSONL", value=str(DEFAULT_EVENTS_PATH))
path = Path(events_path)

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

patient_ids = sorted(patient_events, key=natural_sort_key)
selected_patient = st.sidebar.selectbox("Patient", options=patient_ids)
events = patient_events[selected_patient]

st.caption(f"{len(patient_ids)} patients loaded from {path}")

fig = make_plot(events)
st.plotly_chart(fig, width="stretch")

st.subheader("Structured Events")
hidden_columns = {"source_file"}
events_display = [
    {key: value for key, value in event.items() if key not in hidden_columns}
    for event in events
]
st.dataframe(events_display, width="stretch")
