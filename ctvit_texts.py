import pandas as pd
import os
import json
def load_accession_text(xlsx_file):
    df = pd.read_excel(xlsx_file)
    accession_to_text = {}
    for index, row in df.iterrows():
        accession_to_text[row['AccessionNo']] = row['Impressions']
    return accession_to_text


def prepare_samples(data_folder):
    samples = []
    accession_to_text = load_accession_text("example_data/data_reports.xlsx")
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            print(file)
            if file.endswith(".nii.gz"):
                nii_gz_files=os.path.join(root, file)
            accession_number = nii_gz_files.split("/")[-2].split(".")[-1]
            print(accession_number)
            impression_text = accession_to_text[accession_number]

            img_name = nii_gz_files.split("/")[-1].split(".nii.gz")[0]+"_metadata.json"
            current_directory = os.getcwd()
            print("Current Directory:", current_directory)

            for i in range(2):
                try:
                    metadata_file = "example_data/ctvit-transformer/"+str(i+1)+"/"+str(accession_number)+"/"+img_name
                    print(metadata_file)
                    with open(str(metadata_file), 'r') as f:
                        metadata = json.load(f)
                        print(metadata)
                except:
                    continue
            print(metadata)
            # Extract required metadata
            try:
                age = metadata['PatientAge'][:-1].zfill(3)
                age = age[1:]
            except:
                age = "None"
            try:
                sex = metadata['PatientSex']
            except:
                sex="None"
            if sex.lower() == "m":
                sex="male"
            if sex.lower() =="f":
                sex="female"

            input_text = f'{age} years old {sex}: {impression_text}'
            file_txt = nii_gz_files.split(".nii.gz")[0]+".txt"
            with open(file_txt, 'w') as f:
                f.write(input_text)

prepare_samples("example_data/superres/ctvit_outputs")
