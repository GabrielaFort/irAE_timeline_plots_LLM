import json
import re
from pathlib import Path

import ollama

def parse_lines_file(path):
    """Read a non-empty line-delimited text file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def parse_tissue_list(tissue_list_path):
    """Read canonical OncoTree tissue names."""
    return parse_lines_file(tissue_list_path)

def parse_oncotree_list(tissue_name, base_path):
    """Read canonical OncoTree names for a tissue."""
    path = Path(base_path) / f"{tissue_name}_oncotree_names.txt"
    return parse_lines_file(path)

def load_oncotree_name_to_code(tissue_name, data_base_path):
    """Load a canonical OncoTree name -> OncoTree code mapping."""
    path = Path(data_base_path) / f"{tissue_name}_oncotree_map.json"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def predict_tissue_from_list(tissue_list_path, note, model, temperature):
    """Select the best matching tissue from the canonical tissue list."""
    tissues = parse_tissue_list(tissue_list_path)
    system_prompt = create_tissue_prompt(tissues)
    return generate_oncotree_response(
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        user_prompt=note,
    )

def predict_oncotree_name_from_tissue(tissue_name, note, model, temperature, data_base_path):
    """Select the best matching canonical OncoTree name within one tissue."""
    oncotree_names = parse_oncotree_list(tissue_name, data_base_path)
    system_prompt = create_oncotree_name_prompt()
    user_prompt = (
        "Clinical note:\n"
        f"{note}\n\n"
        "Accepted OncoTree names, delimited by $:\n"
        f"{'$'.join(oncotree_names)}"
    )
    return generate_oncotree_response(
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def extract_primary_diagnosis(note, model, temperature):
    """Extract one short primary cancer diagnosis phrase from the note."""
    return generate_oncotree_response(
        model=model,
        temperature=temperature,
        system_prompt=(
            "Identify the patient's primary cancer diagnosis. Return a JSON "
            "object with one key, `value`. Use a simple phrase around one sentence long for the diagnosis. "
            "If no primary cancer diagnosis is available, set `value` to: Unknown. "
            "This will be used for mapping to OncoTree tissue and OncoTree name mapping, so try to be as specific as possible if a diagnosis is present to allow for mapping to OncoTree terms. "
            "If there are multiple primary tumors mentioned, choose the one that appears most prominently (e.g., in the summary or assessment) or that is most likely to be the indication for immunotherapy. "
            "Prefer primary tumor over metastatic sites. "
        ),
        user_prompt=note,
    )


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
    },
    "required": ["value"],
    "additionalProperties": False,
}

def clean_response(value):
    """Normalize one string response from Ollama."""
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("value", "")

    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"', "`"}:
        value = value[1:-1].strip()
    return value


def print_token_counts(step, response):
    input_tokens = response.get("prompt_eval_count", "unknown")
    output_tokens = response.get("eval_count", "unknown")
    print(f"{step} tokens: input={input_tokens}, output={output_tokens}")


def canonical_match(value, valid_values):
    lookup = {valid_value.lower(): valid_value for valid_value in valid_values}
    return lookup.get(str(value).strip().lower())


def normalized_match(value, valid_values):
    def normalize(text):
        return re.sub(r"\s+", " ", str(text).replace("_", " ").replace("/", " ")).strip().lower()

    lookup = {normalize(valid_value): valid_value for valid_value in valid_values}
    return lookup.get(normalize(value))


def tissue_level_oncotree(tissue, name_to_code):
    oncotree_name = canonical_match(tissue, name_to_code.keys())
    if oncotree_name is None:
        oncotree_name = normalized_match(tissue, name_to_code.keys())

    if oncotree_name is None:
        return None, None

    return oncotree_name, name_to_code[oncotree_name]


def create_tissue_prompt(tissues):
    return (
        "You are an expert pathologist specializing in tumor classification.\n\n"
        "Select the single tissue from the provided list that best matches the "
        "primary cancer diagnosis summary.\n\n"
        "Rules:\n"
        "- Choose only from the provided list.\n"
        "- Prefer the primary tumor tissue over metastatic sites.\n"
        "- Return a JSON object with one key, `value`.\n"
        "- Set `value` to the tissue name exactly as it appears in the list.\n"
        "- If there is no match, set `value` to: Unknown.\n\n"
        "- Prefer `Skin` for any melanoma diagnosis.\n\n"
        "LIST OF TISSUES:\n"
        + "\n".join(tissues)
    )


def create_oncotree_name_prompt():
    return (
        "You are an expert pathologist familiar with the OncoTree classification "
        "system.\n\n"
        "Pick the single OncoTree name from the provided list that best matches "
        "the primary cancer diagnosis summary.\n\n"
        "Rules:\n"
        "- The list of OncoTree names is delimited by the $ character.\n"
        "- Return a JSON object with one key, `value`.\n"
        "- Set `value` to the full OncoTree name exactly as it appears in the list.\n"
        "- Use the ENTIRE name to ensure an exact match - e.g. 'Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma' or 'High-Grade B-Cell Lymphoma, with MYC and BCL2 and/or BCL6 Rearrangements'\n"
        "- Only use names that appear in the provided list.\n"
        "- Focus on the **primary** tumor diagnosis.\n"
        "- Do not include explanations, reasoning, or extra keys.\n"
        "- If no appropriate match exists, set `value` to: Unknown.\n"
        "- Try to choose the most specific name possible based on the note details, but only if it is clearly supported by the text.\n"
        "- Often for breast cancer, Invasive Breast Carcinoma is the most specific name that can be chosen, but choose more specific names if appropriate and clearly supported.\n"
        "- If there are multiple primary tumors mentioned, choose the one that appears most prominently (e.g., in the summary or assessment) or that is most likely to be the indication for immunotherapy.\n"
    )


# LLM wrapper
def generate_oncotree_response(model, temperature, system_prompt, user_prompt):
    response = ollama.chat(
        model = model,
        format = OUTPUT_SCHEMA,
        options = {"temperature": temperature},
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],  

    )
    print_token_counts("OncoTree mapping", response)
    raw_content = response["message"]["content"]
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = raw_content

    return clean_response(parsed)


def classify_oncotree(note, model, temperature, tissue_list_path, data_base_path):
    default = {
        "oncotree_tissue": None,
        "oncotree_name": None,
        "oncotree_code": None,
    }

    try:
        diagnosis = extract_primary_diagnosis(
            note=note,
            model=model,
            temperature=temperature,
        )
        print(f"Extracted diagnosis: {diagnosis}")
        diagnosis_context = note if diagnosis.lower() == "unknown" else diagnosis
        tissue = predict_tissue_from_list(
            tissue_list_path=tissue_list_path,
            note=diagnosis_context,
            model=model,
            temperature=temperature,
        )
        tissues = parse_tissue_list(tissue_list_path)
    except FileNotFoundError:
        return default
    
    tissue = canonical_match(tissue, tissues)
    if tissue is None:
        return default

    try:
        oncotree_name = predict_oncotree_name_from_tissue(
            tissue_name=tissue,
            note=diagnosis_context,
            model=model,
            temperature=temperature,
            data_base_path=data_base_path,
        )
        name_to_code = load_oncotree_name_to_code(
            tissue_name=tissue,
            data_base_path=data_base_path,
        )
    except FileNotFoundError:
        return {
            "oncotree_tissue": tissue,
            "oncotree_name": None,
            "oncotree_code": None,
        }

    oncotree_name = canonical_match(oncotree_name, name_to_code.keys())
    if oncotree_name is None:
        tissue_name, tissue_code = tissue_level_oncotree(tissue, name_to_code)
        return {
            "oncotree_tissue": tissue,
            "oncotree_name": tissue_name,
            "oncotree_code": tissue_code,
        }

    return {
        "oncotree_tissue": tissue,
        "oncotree_name": oncotree_name,
        "oncotree_code": name_to_code[oncotree_name],
    }
