# irAE Timeline Plotter

This project uses a local Ollama model to parse oncology patient notes and generate timelines and cohort summaries of immunotherapy, immune-related adverse events (irAEs), and irAE-directed treatments.

The current workflow is a staged pipeline:

1. **Extract** structured events from patient `.txt` notes, including OncoTree mapping, irAE term mapping, and an LLM date-review pass (`src/llm_note_parser.py`).
2. **Review extracted terms** with a CSV export for mapping/QC (`src/export_mapping_terms.py`).
3. **Build an RxNorm treatment cache** with exact RxNorm matches first and LLM validation for top approximate matches (`src/build_rxnorm_treatment_cache.py`).
4. **Suggest missing treatment ingredient classes** for accepted RxNorm ingredients not yet in `ingredient_class_map.json` (`src/suggest_ingredient_classes.py`), then merge reviewed suggestions into the class map.
5. **Normalize** treatment names/classes and event dates to months-since-first-event per patient (`src/normalize_data.py`).
6. **Visualize** the normalized events in a Streamlit app with filters, timelines, tables, and summaries (`src/app.py`).

Extraction, mapping review, RxNorm cache building, ingredient-class suggestion, and normalization are run up front from the command line. The Streamlit app then reads the pre-computed JSONL, so it does not call the LLM at runtime.

## What the parser extracts

The parser asks the model to return dated events with this structure:

```json
{
  "condition_type": "immunotherapy" or "irae" or "irae_treatment",
  "condition": "short clinical label",
  "start_date": "YYYY-MM-DD or YYYY-MM or YYYY"
}
```

Examples:

- `{"condition_type": "immunotherapy", "condition": "pembrolizumab", "start_date": "2022-03-03"}`
- `{"condition_type": "irae", "condition": "hepatitis", "start_date": "2020-02"}`
- `{"condition_type": "irae_treatment", "condition": "prednisone", "start_date": "2020-02"}`

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
python src/llm_note_parser.py --model gemma4:e4b
```

Useful flags:

- `--model` (required): Ollama model name.
- `--temperature`: sampling temperature (default `0.0`).
- `--input-dir`: directory of patient note files (default `data/patient_notes`).
- `--pattern`: glob pattern for note files (default `*.txt`).
- `--output`: output JSONL path (default `data/patient_events.jsonl`).

Each output record includes patient metadata, OncoTree fields, event fields, and dates. After the initial event extraction, the parser runs a second LLM date-review pass over all extracted events for that patient using the note summary as context; this pass may update `start_date` before rows are written. The parser skips source files already present in the output JSONL and appends only newly extracted notes.

## Step 2: Export terms for mapping review

Export unique terms from the raw event JSONL for review. This splits combination regimens such as `Nivolumab + Ipilimumab` into individual terms and counts repeated terms.

```bash
python src/export_mapping_terms.py \
  --input data/patient_events.jsonl \
  --output data/mapping_terms_review.csv
```

The export includes mapped/unmapped irAE terms, immunotherapy terms, and irAE-treatment terms. irAE terms are not sent to RxNorm; they use the irAE mapping workflow during extraction. The treatment terms are useful for reviewing `custom_map.json` before building the RxNorm cache.

## Step 3: Build the RxNorm treatment cache

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
  --model gemma4:e4b
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

## Step 4: Suggest and merge ingredient classes

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
    "irae_treatment_type": "Corticosteroids"
  },
  "carboplatin": {
    "name": "Carboplatin",
    "therapy_type": "chemotherapy"
  }
}
```

The key is what `normalize_data.py` uses for lookup. The `name` value is the display name used in normalized output. Accepted RxNorm ingredients missing from this map remain lowercase and may be assigned `unmapped` or dropped, depending on row type.

After building the RxNorm cache, suggest map-ready class-map entries for accepted RxNorm ingredients that are missing from `ingredient_class_map.json`:

```bash
python src/suggest_ingredient_classes.py \
  --input data/patient_events.jsonl \
  --rxnorm-cache data/treatment_terms/rxnorm_cache.json \
  --ingredient-class-map data/treatment_terms/ingredient_class_map.json \
  --output data/treatment_terms/ingredient_class_suggestions.json \
  --model gemma4:e4b
```

This script uses the original event `condition_type` to choose the LLM prompt:

- ingredients seen in `immunotherapy` rows are classified as cancer-directed therapy (`ICI`, `targeted therapy`, `chemotherapy`, or `ADC`);
- ingredients seen in `irae_treatment` rows are classified as irAE-treatment categories (`Corticosteroids`, `Non-steroid immunosuppressants/immunomodulators`, or `Endocrine/metabolic replacement or regulation`).

The LLM does not choose display names. The output `name` comes from the RxNorm ingredient name, with the first letter capitalized. Existing entries in `ingredient_class_suggestions.json` are reused, so reruns only call the LLM for new unmapped ingredients.

Review `ingredient_class_suggestions.json`, then merge approved suggestions into `ingredient_class_map.json`. To merge all suggestions:

```bash
python - <<'PY'
import json
from pathlib import Path

map_path = Path("data/treatment_terms/ingredient_class_map.json")
suggestions_path = Path("data/treatment_terms/ingredient_class_suggestions.json")

with map_path.open(encoding="utf-8") as f:
    ingredient_class_map = json.load(f)

with suggestions_path.open(encoding="utf-8") as f:
    suggestions = json.load(f)

ingredient_class_map.update(suggestions)

with map_path.open("w", encoding="utf-8") as f:
    json.dump(dict(sorted(ingredient_class_map.items())), f, indent=2)
    f.write("\n")

print(f"Merged {len(suggestions)} suggestions into {map_path}")
PY
```

## Step 5: Normalize event dates and treatments

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

The normalizer parses partial dates (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`) and additional formats supported by `dateparser`, preferring the first day/month for partial dates. It drops events with no parseable start date, filters unmapped irAE rows, and emits `time_start` in months relative to each patient's earliest event. The normalized output stores relative timing fields, not a normalized calendar `start_date`.

Treatment normalization rules:

- `immunotherapy` rows must have accepted RxNorm ingredient(s) and at least one ingredient mapped with `therapy_type: "ICI"`.
- `irae_treatment` rows must have accepted RxNorm ingredient(s) and at least one ingredient mapped with `irae_treatment_type`.
- immunotherapy ingredients missing from `ingredient_class_map.json` remain lowercase and appear as `unmapped` in regimen therapy types; irAE-treatment rows without a mapped `irae_treatment_type` are dropped.
- patients with no remaining valid irAE or no remaining valid ICI after normalization are excluded and written to the skip log.

Treatment rows removed during normalization are written to the row skip log with the original term, normalized ingredients, and reason. Common reasons include `no_accepted_rxnorm_ingredients`, `no_ici_after_class_mapping`, `no_irae_treatment_type_after_class_mapping`, and `irae_treatment_before_first_ici`.

## Step 6: Run the app

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
│   │   ├── custom_map.json
│   │   ├── rxnorm_cache.json             # output of step 3
│   │   ├── ingredient_class_suggestions.json
│   │   └── ingredient_class_map.json     # treatment class map for normalization
│   ├── patient_events.jsonl              # output of step 1
│   ├── mapping_terms_review.csv          # output of step 2
│   └── patient_events_normalized.jsonl   # output of step 5
├── requirements.txt
├── README.md
└── src/
    ├── app.py                # Streamlit viewer
    ├── llm_note_parser.py    # step 1: LLM extraction (CLI)
    ├── export_mapping_terms.py
    ├── build_rxnorm_treatment_cache.py
    ├── suggest_ingredient_classes.py
    ├── rxnorm_mapping.py
    ├── normalize_data.py     # step 5: treatment/date normalization (CLI)
    ├── oncotree_mapping.py   # OncoTree mapping helpers
    ├── irae_mapping.py       # irAE terminology mapping helpers
    └── timeline_plotter.py   # Plotly figure builder used by app.py
```
