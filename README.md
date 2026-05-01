# irAE Timeline Plotter

This project uses a local Ollama model to parse oncology patient notes from plain text and generate a timeline of treatments and immune-related adverse events (irAEs) in a Streamlit app.

The current workflow is:

1. Upload a `.txt` file containing a patient note (from one patient).
2. Select a local Ollama model.
3. Let the LLM extract structured events.
4. Display a Plotly timeline with treatments and irAEs colored differently.

## What the app extracts

The parser asks the model to return JSON events with this structure:

```json
{
  "condition_type": "treatment or irAE",
  "condition": "short clinical label",
  "start_date": "YYYY-MM-DD or YYYY-MM or YYYY",
  "end_date": "YYYY-MM-DD or YYYY-MM or YYYY or null"
}
```

Examples:

- `{"condition_type": "treatment", "condition": "pembrolizumab", "start_date": "2022-03-03", "end_date": null}`
- `{"condition_type": "irae", "condition": "hepatitis", "start_date": "2020-02", "end_date": "2020-04"}`

## Requirements

You need the following installed on your computer:

- Python 3.10+ recommended
- [Ollama](https://ollama.com/)
- A local Ollama model pulled onto your machine

Python packages for this repo are listed in [requirements.txt](/Users/gabbyfort/Documents/Tan_Lab/Projects/irAE_timeline_plots_LLM/irAE_timeline_plots_LLM/requirements.txt).

## Setup

Clone the repository and move into the project folder:

```bash
git clone <your-repo-url>
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
ollama pull llama3.1
```

You can confirm your local models with:

```bash
ollama list
```

The Streamlit app reads your locally available models and shows them in the sidebar.

## Run the app

From the repository root:

```bash
streamlit run src/app.py
```

Then open the local URL shown by Streamlit in your browser.

## How to use

1. Launch the app.
2. Choose an Ollama model from the sidebar.
3. Optionally adjust temperature.
4. Upload a patient note as a `.txt` file.
5. Click `Parse and Plot`.
6. Review the extracted event table and generated timeline.

## Input file format

The app expects a plain text clinical note or plain text note summary.

Examples are in the [`data`](./data) directory:

- `data/patient_1.txt`
- `data/patient_2.txt`
- `data/patient_3.txt`

These files are useful for testing the app after setup.

## Project structure

```text
.
├── data/
├── requirements.txt
├── README.md
└── src/
    ├── app.py
    ├── llm_note_parser.py
    ├── timeline_plotter.py
    └── tools.py
```

Key files:

- [src/app.py](/Users/gabbyfort/Documents/Tan_Lab/Projects/irAE_timeline_plots_LLM/irAE_timeline_plots_LLM/src/app.py): Streamlit UI
- [src/llm_note_parser.py](/Users/gabbyfort/Documents/Tan_Lab/Projects/irAE_timeline_plots_LLM/irAE_timeline_plots_LLM/src/llm_note_parser.py): Ollama-based event extraction
- [src/timeline_plotter.py](/Users/gabbyfort/Documents/Tan_Lab/Projects/irAE_timeline_plots_LLM/irAE_timeline_plots_LLM/src/timeline_plotter.py): Plotly timeline generation
- [src/tools.py](/Users/gabbyfort/Documents/Tan_Lab/Projects/irAE_timeline_plots_LLM/irAE_timeline_plots_LLM/src/tools.py): local Ollama model discovery

## Troubleshooting

If the app says no Ollama models were found:

- make sure Ollama is installed
- make sure the Ollama app or daemon is running
- make sure you have pulled at least one model with `ollama pull <model_name>`

If parsing fails:

- verify the selected model exists in `ollama list`
- try temperature `0.0`
- try a shorter or cleaner `.txt` note

If the timeline is empty:

- inspect the structured events table in the app
- confirm the model returned valid dates
- confirm the note actually includes dated treatments or irAEs

## Notes

- All inference is intended to run locally through Ollama.
- No external API key is required for the current app.
- Output quality depends heavily on note quality and model choice.

