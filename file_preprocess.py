#!/usr/bin/env python
# coding=utf-8
"""
Author       : Bowen Zheng
Date         : 2024-02-22 16:55:29
LastEditors  : ibowennn shihun44@163.com
LastEditTime : 2024-02-27 16:19:02
Description  : 

"""
#!/usr/bin/env python
# coding=utf-8
"""
Author       : Bowen Zheng
Date         : 2024-02-22 16:55:29
LastEditors  : ibowennn shihun44@163.com
LastEditTime : 2024-02-23 18:34:40
Description  : 

"""
#!/usr/bin/env python
# coding=utf-8
"""
Author       : Bowen Zheng
Date         : 2024-02-22 16:55:29
LastEditors  : ibowennn shihun44@163.com
LastEditTime : 2024-02-22 19:24:38
Description  : 

"""
# %%
import glob
import os
import shutil

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

# %%
root_dir = "/home/data2/public/gzyy/images_rearranged"
dirs = os.listdir(root_dir)

# %%
files = glob.glob(os.path.join(root_dir, "*/*/*"))
print(f"Number of dirs: {len(dirs)};\nNumber of files: {len(files)}")


# %%
def get_information(List):
    metadata = {
        "patientID": [],
        "Name": [],
        "Age": [],
        "Sex": [],
        "BirthDate": [],
        "StudyDate": [],
        "StudyID": [],
        "StudyDescription": [],
        "SeriesID": [],
        "SeriesDescription": [],
        "DiffusionBValue": [],
        "Manufacturer": [],
        "path": [],
    }

    for i in tqdm(List):
        information = {}
        try:
            ds = pydicom.dcmread(i)
        except:
            print(i)

        information["patientID"] = ds.PatientID
        information["Name"] = i.split("/")[-3].split("_")[0]
        information["Age"] = ds.get("PatientAge", "NA")
        information["Sex"] = ds.get("PatientSex", "NA")
        information["BirthDate"] = ds.get("PatientBirthDate", "NA")
        information["StudyDate"] = ds.StudyDate
        information["StudyID"] = ds.StudyInstanceUID
        information["StudyDescription"] = ds.get("StudyDescription", "NA")
        information["SeriesID"] = ds.SeriesInstanceUID
        information["SeriesDescription"] = ds.get("SeriesDescription", "NA")
        information["DiffusionBValue"] = ds.get("DiffusionBValue", "NA")
        information["Manufacturer"] = ds.get("Manufacturer", "NA")
        information["path"] = i

        for key, value in information.items():
            metadata[key].append(value)

    metadata = pd.DataFrame(metadata)

    return metadata


# %%
metadata = get_information(files)
metadata.to_csv("../data/metadata.csv", index=False)
# %%
root_dir = "/home/data2/public/gzyy/images_rearranged"
for i in tqdm(range(len(metadata))):
    name = metadata.Name[i]
    id = metadata.patientID[i]
    studydate = metadata.StudyDate[i]
    studydes = metadata.StudyDescription[i]
    seriesdes = metadata.SeriesDescription[i]

    new_folder = os.path.join(
        root_dir,
        name,
        studydate,
        id,
        studydes,
        seriesdes.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "_"),
    )

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    else:
        pass

    shutil.copy2(metadata.path[i], new_folder)
# %%
metadata = pd.read_csv("../data/metadata.csv")


# %%
def rearrange_file(List, root_dir):
    for path in tqdm(List):
        ds = pydicom.dcmread(path)

        name = path.split("/")[-3].split("_")[0]
        id = ds.PatientID
        studydate = ds.StudyDate
        studydes = ds.get("StudyDescription", "NA")
        seriesdes = ds.get("SeriesDescription", "NA")

        new_folder = os.path.join(
            root_dir,
            name,
            studydate,
            id,
            studydes,
            seriesdes.replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "_"),
        )

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        else:
            pass

        shutil.copy2(path, new_folder)


root_dir = "/home/data2/public/gzyy/images_rearranged"
rearrange_file(files, root_dir)

# %%
