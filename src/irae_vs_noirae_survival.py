# Generate survival curve for irAE vs no irAE patients
# irAE means possible irAE - not validated by my + CNs pipeline yet
import os
import pandas as pd

# Read in all MRN #s from filenames in data/all_possible_irae
def read_mrn(file_path):
    possible_iraes = []
    for filename in os.listdir(file_path):
        if filename.endswith(".txt"):
            mrn = filename.split("_")[1]
            mrn = mrn.split(".")[0]
            possible_iraes.append(mrn)
    return possible_iraes


# Read in excel file with survival and last follow up dates
# Add column for irAE vs no irAE based on MRN #s from above function
def read_survival_data(file_path, possible_iraes):
    df = pd.read_excel(file_path)
    df["irAE"] = df["MRN"].apply(lambda x: "irAE" if str(x) in possible_iraes else "no irAE")
    return df



            



