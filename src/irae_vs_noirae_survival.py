# Generate survival curve for irAE vs no irAE patients
# Only considering patients that were for sure treated with ICI
import json
import os
import re
import pandas as pd
from oncotree_mapping import (
    canonical_match,
    load_oncotree_name_to_code,
    parse_tissue_list,
    predict_oncotree_name_from_tissue,
    predict_tissue_from_list,
    tissue_level_oncotree,
)


def normalize_mrn(mrn):
    value = str(mrn).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(8) if value.isdigit() else value


def mrn_from_filename(filename):
    filename = os.path.basename(str(filename))
    return normalize_mrn(filename.split("_", 1)[1].rsplit(".", 1)[0])


def read_skipped_mrns(skipped_file_path="data/patient_events_skipped.jsonl"):
    if not skipped_file_path or not os.path.exists(skipped_file_path):
        return set()

    skipped_mrns = set()
    with open(skipped_file_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            source_file = None
            if "source_file" in record:
                source_file = record.get("source_file")
            elif record.get("source_files"):
                source_file = record.get("source_files")[0]
            if source_file:
                skipped_mrns.add(mrn_from_filename(source_file))
    return skipped_mrns


# Read in all MRN #s from filenames in file path
def read_mrn(file_path, skipped_file_path="data/patient_events_skipped.jsonl"):
    possible_iraes = []
    skipped_mrns = read_skipped_mrns(skipped_file_path)

    for filename in os.listdir(file_path):
        if filename.endswith(".txt"):
            mrn = mrn_from_filename(filename)
            if mrn in skipped_mrns:
                continue
            possible_iraes.append(mrn)
    return possible_iraes


# Read in excel file with survival and last follow up dates
# Add column for irAE vs no irAE based on MRN #s from above function
# Skip over any MRNs that were excluded during normaliztaion 
def read_survival_data(file_path, possible_iraes, skipped_file_path="data/patient_events_skipped.jsonl"):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    skipped_mrns = read_skipped_mrns(skipped_file_path)
    possible_iraes = {normalize_mrn(mrn) for mrn in possible_iraes}

    df = df[~df["MRN"].apply(lambda x: normalize_mrn(x) in skipped_mrns)].copy()
    df["irAE"] = df["MRN"].apply(lambda x: "irAE" if normalize_mrn(x) in possible_iraes else "no irAE")
    return df

def label_oncotree(df, normalized_data_path="data/patient_events_normalized.jsonl"):
    # Read in normalized data to get OncoTree tissue and name mapping
    oncotree_mapping = {}
    with open(normalized_data_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            mrn = mrn_from_filename(record.get("source_file"))
            oncotree_tissue = record.get("oncotree_tissue")
            oncotree_name = record.get("oncotree_name")
            oncotree_code = record.get("oncotree_code")
            if mrn and mrn not in oncotree_mapping:
                oncotree_mapping[mrn] = {
                    "oncotree_tissue": oncotree_tissue,
                    "oncotree_name": oncotree_name,
                    "oncotree_code": oncotree_code
                }
    # Create columns if missing
    for column in ["oncotree_tissue", "oncotree_name", "oncotree_code"]:
        if column not in df.columns:
            df[column] = None

    # Fill in missing cells with OncoTree tissue and name from normalized data
    for column in ["oncotree_tissue", "oncotree_name", "oncotree_code"]:
        mapped_values = df["MRN"].apply(lambda x: oncotree_mapping.get(normalize_mrn(x), {}).get(column))
        df[column] = df[column].where(~df[column].apply(is_missing), mapped_values)

    return df


def safe_column_name(value):
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    return value.strip("_")


def read_patient_irae_types(normalized_data_path):
    patient_irae_types = {}
    with open(normalized_data_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("condition_type") != "irae" or is_missing(record.get("irae_type")):
                continue

            mrn = mrn_from_filename(record.get("source_file"))
            patient_irae_types.setdefault(mrn, set()).add(record.get("irae_type"))

    return patient_irae_types


def label_irae_types(df, normalized_data_path="data/patient_events_normalized.jsonl"):
    patient_irae_types = read_patient_irae_types(normalized_data_path)
    all_irae_types = sorted(
        {irae_type for irae_types in patient_irae_types.values() for irae_type in irae_types}
    )

    df["irae_types"] = df["MRN"].apply(
        lambda x: "; ".join(sorted(patient_irae_types.get(normalize_mrn(x), [])))
    )

    for irae_type in all_irae_types:
        column = f"irae_type_{safe_column_name(irae_type)}"
        df[column] = df["MRN"].apply(
            lambda x, t=irae_type: int(t in patient_irae_types.get(normalize_mrn(x), set()))
        )

    return df


def is_missing(value):
    return pd.isna(value) or str(value).strip() == ""

def predict_oncotree_from_cancer_type(
    cancer_type,
    model,
    temperature=0,
    tissue_list_path="data/oncotree_tissues/tissue_types.txt",
    data_base_path="data/oncotree_tissues"):

    tissue = predict_tissue_from_list(
        tissue_list_path=tissue_list_path,
        note=cancer_type,
        model=model,
        temperature=temperature)
    
    tissue = canonical_match(tissue, parse_tissue_list(tissue_list_path))
    if tissue is None:
        return {"oncotree_tissue": None, "oncotree_name": None, "oncotree_code": None}

    name_to_code = load_oncotree_name_to_code(tissue, data_base_path)
    oncotree_name = predict_oncotree_name_from_tissue(
        tissue_name=tissue,
        note=cancer_type,
        model=model,
        temperature=temperature,
        data_base_path=data_base_path,
    )
    oncotree_name = canonical_match(oncotree_name, name_to_code.keys())
    if oncotree_name is None:
        oncotree_name, oncotree_code = tissue_level_oncotree(tissue, name_to_code)
    else:
        oncotree_code = name_to_code[oncotree_name]

    return {
        "oncotree_tissue": tissue,
        "oncotree_name": oncotree_name,
        "oncotree_code": oncotree_code,
    }

def predict_missing_oncotree(
    df,
    cancer_type_col="cancer type",
    model="gemma4:e4b",
    temperature=0,
    tissue_list_path="data/oncotree_tissues/tissue_types.txt",
    data_base_path="data/oncotree_tissues"):

    # Make cols if they dont already exist in excel sheet
    for column in ["oncotree_tissue", "oncotree_name", "oncotree_code"]:
        if column not in df.columns:
            df[column] = None

    # If oncotree_tissue or oncotree_name is missing, predict from cancer type column using LLM
    for index, row in df.iterrows():
        if not (is_missing(row["oncotree_tissue"]) or is_missing(row["oncotree_name"])):
            continue
        if is_missing(row[cancer_type_col]):
            continue
        print(f"Predicting OncoTree for row {index} with cancer type '{row[cancer_type_col]}'...")
        oncotree = predict_oncotree_from_cancer_type(
            cancer_type=str(row[cancer_type_col]),
            model=model,
            temperature=temperature,
            tissue_list_path=tissue_list_path,
            data_base_path=data_base_path,
        )
        df.at[index, "oncotree_tissue"] = oncotree["oncotree_tissue"]
        df.at[index, "oncotree_name"] = oncotree["oncotree_name"]
        df.at[index, "oncotree_code"] = oncotree["oncotree_code"]

    return df

def label_unmapped_oncotree_as_unknown(df):
    missing_any_oncotree = (
        df["oncotree_tissue"].apply(is_missing)
        | df["oncotree_name"].apply(is_missing)
        | df["oncotree_code"].apply(is_missing)
    )

    df.loc[missing_any_oncotree, "oncotree_tissue"] = df.loc[
        missing_any_oncotree, "oncotree_tissue"
    ].apply(lambda x: "Unknown" if is_missing(x) else x)

    df.loc[missing_any_oncotree, "oncotree_name"] = df.loc[
        missing_any_oncotree, "oncotree_name"
    ].apply(lambda x: "Unknown" if is_missing(x) else x)

    df.loc[missing_any_oncotree, "oncotree_code"] = df.loc[
        missing_any_oncotree, "oncotree_code"
    ].apply(lambda x: "Unknown" if is_missing(x) else x)

    return df

def save_survival_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    # Read in MRNs representing patients with irAEs, skipping excluded px
    possible_iraes = read_mrn("data/patient_notes", skipped_file_path="data/patient_events_skipped.jsonl")
    print(f"Found {len(possible_iraes)} possible irAE patients after skipping excluded MRNs.")

    # Read in survival data and label irAE vs no irAE
    survival_df = read_survival_data(
        #"data/result_2640_ici_OS_clean.xlsx",
        "data/result_2640_ici_OS_0626_2026_cleaned.xlsx",
        possible_iraes,
        skipped_file_path="data/patient_events_skipped.jsonl"
    )

    normalized_data_path = "data/events_060926_gemma4_e4b_normalized_rxnorm.jsonl"

    # Label OncoTree tissue and name
    survival_df = label_oncotree(survival_df, normalized_data_path=normalized_data_path)
    print(f"Labeled pre-existing oncotree data for {len(survival_df['oncotree_name'].dropna())} patients.")

    # Label patient irAE types and one-hot encoded irAE type columns
    survival_df = label_irae_types(survival_df, normalized_data_path=normalized_data_path)

    # Predict missing OncoTree tissue and name
    survival_df = predict_missing_oncotree(
        survival_df,
        cancer_type_col="cancer type",
        model="gemma4:e4b",
        temperature=0,
        tissue_list_path="data/oncotree_tissues/tissue_types.txt",
        data_base_path="data/oncotree_tissues"
    )

    # Label unmapped OncoTree tissue and name and code as "Unknown"
    survival_df = label_unmapped_oncotree_as_unknown(survival_df)

    # Save the updated survival data to a new CSV file
    save_survival_data(survival_df, "data/survival_data_labeled.csv")

if __name__ == "__main__":
    main()
