import pandas as pd
from tqdm import tqdm
import numpy as np
import os

dir = os.getcwd()
# establish string keywords
T1w = "T1w"
T2w = "T2w"
flair = "FLAIR"
swi = "swi"

mri_csv_data = pd.read_csv(dir + "/../data/oasis_mri_data.csv")


# For Windows, use the line below, comment above
# mri_csv_path = "\\..\\data\\oasis_mri_data.csv"

def check_for_keyscans(data):
    for index, row in tqdm(data.iterrows()):
        # kick out empty Scan row
        if pd.isnull(row['Scans']):
            data.drop(index, axis=0, inplace=True)
        # check conditions
        elif T1w in row['Scans'] and T2w in row['Scans'] and flair in row['Scans'] and swi in row['Scans']:
            continue
        # if conditions fail, drop row entry from pd
        else:
            data.drop(index, axis=0, inplace=True)

    return data


def main():
    print("Oasis MRI CSV data pruning:\n")
    pruned_data = check_for_keyscans(mri_csv_data)
    print("Pruned.\n")

    print("Exporting pruned data as CSV.\n")
    pruned_data.to_csv(dir + "/../results/pruned_data.csv", index=False)

    print("Operation complete.\n")


if __name__ == '__main__':
    main()
