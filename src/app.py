import json
import streamlit as st

from llm_note_parser import extract_events
from timeline_plotter import make_plot
from tools import discover_local_ollama_models


st.set_page_config(page_title="irAE Timeline Parser", layout="wide")
st.title("Patient Note -> LLM Parse -> Timeline")

# LLM settings (kept minimal)
# call once at startup
available_models = discover_local_ollama_models()

# show a helpful sidebar message if empty
if not available_models:
    st.sidebar.error(
        "No local Ollama models discovered. Make sure Ollama is installed and running, "
        "and that the Python `ollama` package can reach it."
    )

# model selection as a dropdown in the sidebar
model = st.sidebar.selectbox("Model", options=available_models or ["(no models)"], index=0)

# Temperature slider to select temp
temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

uploaded = st.file_uploader("Upload patient note (.txt)", type=["txt"])

if uploaded is not None:
    note_text = uploaded.read().decode("utf-8", errors="ignore")
    with st.expander("Preview note text", expanded=False):
        st.text(note_text)

    if st.button("Parse and Plot", type="primary"):
        with st.spinner("Parsing note with Ollama LLM..."):
            try:
                events = extract_events(model=model, temperature=temperature, note=note_text)
            except Exception as e:
                st.error(f"Parsing failed: {e}")
                st.stop()

        st.success(f"Parsed {len(events)} events")
        st.subheader("Structured Events")
        st.dataframe(events, width = "stretch")

        st.subheader("Timeline")
        fig = make_plot(events, uploaded.name)
        st.plotly_chart(fig, width = "stretch")

else:
    st.info("Upload a .txt patient note to begin.")
