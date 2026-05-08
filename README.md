# irAE Timeline Plotter

This project uses a local Ollama model to parse oncology patient notes and generate per-patient timelines of diseases, treatments, and immune-related adverse events (irAEs) in a Streamlit app.

The current workflow is a three-step pipeline:

1. **Extract** structured events from a directory of patient `.txt` notes using a local Ollama model (`src/llm_note_parser.py`).
2. **Normalize** the extracted event dates to months-since-first-event per patient (`src/normalize_data.py`).
3. **Visualize** the normalized events in a Streamlit app, with a sidebar to switch between patients (`src/app.py`).

Extraction and normalization are run once up front from the command line. The Streamlit app then reads the pre-computed JSONL, so it does not call the LLM at runtime.

## What the app extracts

The parser asks the model to return JSON events with this structure:

```json
{
  "condition_type": "treatment" or "irae" or "disease",
  "condition": "short clinical label",
  "start_date": "YYYY-MM-DD or YYYY-MM or YYYY",
  "end_date": "YYYY-MM-DD or YYYY-MM or YYYY or null"
}
```

Examples:

- `{"condition_type": "treatment", "condition": "pembrolizumab", "start_date": "2022-03-03", "end_date": null}`
- `{"condition_type": "irae", "condition": "hepatitis", "start_date": "2020-02", "end_date": "2020-04"}`
- `{"condition_type": "disease", "condition": "melanoma", "start_date": "2019", "end_date": null}`

After normalization, each event also has `time_start` and `time_stop` fields expressed in months relative to the patient's earliest event.

## Requirements

You need the following installed on your computer:

- Python 3.10+ recommended
- [Ollama](https://ollama.com/)
- A local Ollama model pulled onto your machine

## Setup

Clone the repository and move into the project folder:

```bash
git clone https://github.com/GabrielaFort/irAE_timeline_plots_LLM.git
cd irAE_timeline_plots_LLM
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Install and start Ollama

Install Ollama from the official site:

```text
https://ollama.com/download
```

After installation, start Ollama if it is not already running.

Pull at least one local model. Example:

```bash
ollama pull llama3.1:8b
```

You can confirm your local models with:

```bash
ollama list
```

## Input file format

Place one plain-text clinical note per patient in the `data/` directory, named like `patient_1.txt`, `patient_2.txt`, etc. Each file should contain a single patient's note or note summary.

## Step 1: Extract events from notes

Run the parser against the directory of patient notes. This calls the local Ollama model on each `.txt` file and writes one JSON event per line to `data/patient_events.jsonl`.

```bash
python src/llm_note_parser.py --model llama3.1:8b
```

Useful flags:

- `--model` (required): Ollama model name.
- `--temperature`: sampling temperature (default `0.0`).
- `--input-dir`: directory of patient note files (default `data`).
- `--pattern`: glob pattern for note files (default `*.txt`).
- `--output`: output JSONL path (default `data/patient_events.jsonl`).

Each output record includes `patient_id`, `source_file`, `condition_type`, `condition`, `start_date`, and `end_date`.

## Step 2: Normalize event dates

Convert raw dates to months-since-first-event per patient. This produces `data/patient_events_normalized.jsonl`, which is what the app reads.

```bash
python src/normalize_data.py
```

Useful flags:

- `--input`: input JSONL path (default `data/patient_events.jsonl`).
- `--output`: normalized output JSONL path (default `data/patient_events_normalized.jsonl`).

The normalizer parses partial dates (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`), drops events with no parseable start date, and emits `time_start` and `time_stop` in months relative to each patient's earliest event.

## Step 3: Run the app

From the repository root:

```bash
streamlit run src/app.py
```

Then open the local URL shown by Streamlit in your browser.

In the sidebar you can:

- Point to a different normalized events JSONL file.
- Select which patient's timeline to display.

The main view shows a Plotly timeline (treatments, irAEs, and diseases colored differently) along with the structured event table.

## Project structure

```text
.
├── data/
│   ├── patient_1.txt
│   ├── ...
│   ├── patient_events.jsonl              # output of step 1
│   └── patient_events_normalized.jsonl   # output of step 2
├── outputs/
├── requirements.txt
├── README.md
└── src/
    ├── app.py                # Streamlit viewer
    ├── llm_note_parser.py    # step 1: LLM extraction (CLI)
    ├── normalize_data.py     # step 2: date normalization (CLI)
    └── timeline_plotter.py   # Plotly figure builder used by app.py
```
