# irAE Timeline Plotter

This project uses a local Ollama model to parse oncology patient notes and generate timelines and cohort summaries of immunotherapy, immune-related adverse events (irAEs), and irAE-directed treatments.

The current workflow is a four-step pipeline:

1. **Extract** structured events and terminology mappings from patient `.txt` notes using a local Ollama model (`src/llm_note_parser.py`).
2. **Build an RxNorm treatment cache** with exact RxNorm matches first and LLM validation for the top approximate matches (`src/build_rxnorm_treatment_cache.py`).
3. **Normalize** treatment names/classes and event dates to months-since-first-event per patient (`src/normalize_data.py`).
4. **Visualize** the normalized events in a Streamlit app with filters, timelines, tables, and summaries (`src/app.py`).

Extraction, RxNorm cache building, and normalization are run once up front from the command line. The Streamlit app then reads the pre-computed JSONL, so it does not call the LLM at runtime.

## What the parser extracts

The parser asks the model to return dated events with this structure:

```json
{
  "condition_type": "immunotherapy" or "irae" or "irae_treatment",
  "condition": "short clinical label",
  "start_date": "YYYY-MM-DD or YYYY-MM or YYYY",
  "end_date": "YYYY-MM-DD or YYYY-MM or YYYY or null"
}
```

Examples:

- `{"condition_type": "immunotherapy", "condition": "pembrolizumab", "start_date": "2022-03-03", "end_date": null}`
- `{"condition_type": "irae", "condition": "hepatitis", "start_date": "2020-02", "end_date": "2020-04"}`
- `{"condition_type": "irae_treatment", "condition": "prednisone", "start_date": "2020-02", "end_date": null}`

During extraction, the pipeline also maps cancer diagnoses to OncoTree terminology and maps irAE terms to canonical irAE names/types. After normalization, each event also has a `time_start` field expressed in months relative to the patient's earliest event.

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
ollama pull gemma4:e4b
```

You can confirm your local models with:

```bash
ollama list
```

## Input file format

Place one plain-text clinical note per patient in the `data/patient_notes/` directory. Each file should contain a single patient's note or note summary.

## Step 1: Extract events from notes

Run the parser against the directory of patient notes. This calls the local Ollama model on each `.txt` file and writes one JSON event per line to `data/patient_events.jsonl`.

```bash
python src/llm_note_parser.py --model llama3.1:8b
```

Useful flags:

- `--model` (required): Ollama model name.
- `--temperature`: sampling temperature (default `0.0`).
- `--input-dir`: directory of patient note files (default `data/patient_notes`).
- `--pattern`: glob pattern for note files (default `*.txt`).
- `--output`: output JSONL path (default `data/patient_events.jsonl`).

Each output record includes patient metadata, OncoTree fields, event fields, and dates. The parser skips source files already present in the output JSONL and appends only newly extracted notes.

## Optional QC: Export treatment terms

Export unique treatment terms from the raw event JSONL for review. This splits combination regimens such as `Nivolumab + Ipilimumab` into individual terms. This step is optional; the RxNorm cache builder can read `data/patient_events.jsonl` directly.

```bash
python src/export_mapping_terms.py \
  --input data/patient_events.jsonl \
  --output data/mapping_terms_review.csv
```

The export includes immunotherapy and irAE-treatment terms. irAE terms are not sent to RxNorm; they continue to use the irAE mapping workflow from extraction.

## Step 2: Build the RxNorm treatment cache

Build or update the local RxNorm cache from the raw event JSONL. For each individual treatment term, the script:

1. tries an exact RxNorm match;
2. if no exact match exists, asks RxNorm for the top approximate matches;
3. asks the local Ollama model whether each approximate candidate is a good medication match, stopping at the first yes;
4. stores accepted matches with their RxNorm ingredient names.

```bash
python src/build_rxnorm_treatment_cache.py \
  --input data/patient_events.jsonl \
  --cache data/treatment_terms/rxnorm_cache.json \
  --custom-map data/treatment_terms/custom_map.json \
  --model llama3.1:8b
```

Useful flags:

- `--model` (required): Ollama model name used only to validate approximate RxNorm matches.
- `--custom-map`: optional pre-RxNorm rewrite map (default `data/treatment_terms/custom_map.json`).
- `--temperature`: validation sampling temperature (default `0.0`).
- `--max-approximate`: number of approximate candidates to validate (default `2`, the top two matches).
- `--refresh`: rebuild entries that are already present in the cache.

Cache entries with `status: "accepted"` are used during normalization. Entries with `status: "unresolved"` are skipped.

The custom map is applied before RxNorm lookup while preserving the original extracted term as the cache key. For example:

```json
{
  "taxol": "paclitaxel",
  "cortisone": "hydrocortisone"
}
```

If an event contains `Taxol`, the cache key remains `taxol`, but RxNorm is queried with `paclitaxel`.

After building the cache, you can rerun the export with cache annotations:

```bash
python src/export_mapping_terms.py \
  --input data/patient_events.jsonl \
  --output data/mapping_terms_review.csv \
  --rxnorm-cache data/treatment_terms/rxnorm_cache.json
```

This adds columns such as `rxnorm_status`, `rxnorm_match_method`, `rxnorm_matched_name`, and `rxnorm_ingredients`.

## Treatment class map

Treatment classes are assigned during normalization with:

```text
data/treatment_terms/ingredient_class_map.json
```

This file is keyed by lowercase RxNorm ingredient name. Each mapped ingredient can define fields such as:

```json
{
  "pembrolizumab": {
    "name": "Pembrolizumab",
    "therapy_type": "ICI",
    "ici_class": "PD-1_inhibitor"
  },
  "prednisone": {
    "name": "Prednisone",
    "irae_treatment_type": "Systemic corticosteroids"
  },
  "carboplatin": {
    "name": "Carboplatin",
    "therapy_type": "chemotherapy"
  }
}
```

The `name` value is the canonical display name used in normalized output. Accepted RxNorm ingredients missing from this map remain in lowercase and are assigned `unmapped`, which makes them easier to find and add later.

## Step 3: Normalize event dates and treatments

Convert raw dates to months-since-first-event per patient. This produces `data/patient_events_normalized.jsonl`, which is what the app reads.

```bash
python src/normalize_data.py
```

Useful flags:

- `--input`: input JSONL path (default `data/patient_events.jsonl`).
- `--output`: normalized output JSONL path (default `data/patient_events_normalized.jsonl`).
- `--rxnorm-cache`: treatment term cache (default `data/treatment_terms/rxnorm_cache.json`).
- `--ingredient-class-map`: ingredient class map (default `data/treatment_terms/ingredient_class_map.json`).
- `--skip-log`: skipped-patient log (default `data/patient_events_skipped.jsonl`).
- `--row-skip-log`: skipped treatment-row log (default `data/patient_event_rows_skipped.jsonl`).

The normalizer parses partial dates (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`) and common slash-date formats, drops events with no parseable start date, filters unmapped irAE rows, and emits `time_start` in months relative to each patient's earliest event.

Treatment normalization rules:

- `immunotherapy` rows must have accepted RxNorm ingredient(s) and at least one ingredient mapped with `therapy_type: "ICI"`.
- `irae_treatment` rows must have accepted RxNorm ingredient(s) and at least one ingredient mapped with `irae_treatment_type`.
- accepted ingredients missing from `ingredient_class_map.json` remain lowercase and appear as `unmapped` in regimen therapy types.
- patients with no remaining valid irAE or no remaining valid ICI after normalization are excluded and written to the skip log.

Treatment rows removed during normalization are written to the row skip log with the original term, normalized ingredients, and reason. Common reasons include `no_accepted_rxnorm_ingredients`, `no_ici_after_class_mapping`, and `no_irae_treatment_type_after_class_mapping`.

## Step 4: Run the app

From the repository root:

```bash
streamlit run src/app.py
```

Then open the local URL shown by Streamlit in your browser.

The app includes:

- filters by patient, OncoTree fields, immunotherapy, irAE, irAE type, and irAE treatment
- per-patient timeline plots
- a filtered event table
- cohort summary plots and tables

The app reads the normalized JSONL file and does not call the LLM at runtime.

## Project structure

```text
.
├── data/
│   ├── patient_notes/
│   ├── oncotree_tissues/
│   ├── irae_terms/
│   ├── treatment_terms/
│   │   ├── rxnorm_cache.json             # output of step 2
│   │   └── ingredient_class_map.json     # treatment class map for normalization
│   ├── patient_events.jsonl              # output of step 1
│   ├── mapping_terms_review.csv          # optional QC output
│   └── patient_events_normalized.jsonl   # output of step 3
├── requirements.txt
├── README.md
└── src/
    ├── app.py                # Streamlit viewer
    ├── llm_note_parser.py    # step 1: LLM extraction (CLI)
    ├── export_mapping_terms.py
    ├── build_rxnorm_treatment_cache.py
    ├── rxnorm_mapping.py
    ├── normalize_data.py     # step 3: treatment/date normalization (CLI)
    ├── oncotree_mapping.py   # OncoTree mapping helpers
    ├── irae_mapping.py       # irAE terminology mapping helpers
    └── timeline_plotter.py   # Plotly figure builder used by app.py
```
