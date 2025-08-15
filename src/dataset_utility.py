import os
import numpy as np
import pandas as pd
import ast
from typing import Union

base_dir = os.path.dirname(__file__)


img_col_map = {
    "derm12345": 'image_id',
    "bcn20000": 'bcn_filename',
    "hiba": 'isic_id',
    "pad-ufes-20": 'img_id',
    "scin_clinical": 'image_path'
}

label_col_map = {
    "derm12345": 'label',
    "bcn20000": 'diagnosis',
    "hiba": 'diagnosis',
    "pad-ufes-20": 'diagnostic',
    "scin_clinical": 'dermatologist_skin_condition_on_label_name',
}

bcn20000_label_map = {
    "NV": "Nevus",
    "MEL": "Melanoma",
    "BCC": "Basal cell carcinoma",
    "BKL": "Benign keratosis",
    "AK": "Actinic keratosis",
    "SCC": "Squamous carcinoma",
    "DF": "Dermatofibroma",
    "VASC": "Vascular",
}

padufes20_label_map = {
    "ACK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "NEV": "Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "SEK": "Seborrheic Keratosis",
}


def get_paths(dataset:str, image_id:str):
    dataset_path = os.path.join(base_dir, f"../datasets/{dataset}")
    if dataset in {"derm12345", "bcn20000", "hiba"}:
        image_path = os.path.join(dataset_path, image_id + '.jpg')
    elif dataset in {"pad-ufes-20", "scin_clinical"}:
        image_path = os.path.join(dataset_path, image_id + '.png')
    else:
        raise ValueError("Invalid dataset name. Please provide a valid dataset name.")
    
    return dataset_path, image_path


def get_metadata_row(metadata:pd.DataFrame, dataset:str, image_id:str):
    if dataset in {"derm12345", "hiba"}:
        image_filename = image_id
    elif dataset == "bcn20000":
        image_filename = image_id + ".jpg"
    elif dataset in {"pad-ufes-20", "scin_clinical"}:
        image_filename = image_id + ".png"
    else:
        raise ValueError("Invalid dataset name. Please provide a valid dataset name.")

    img_col = img_col_map[dataset]
    return metadata[metadata[img_col] == image_filename].iloc[0]


def prepare_metadata_derm12345(row: pd.Series) -> str:
    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {row['super_class']}, {row['malignancy']}, {row['main_class_1']},"
        f" {row['main_class_2']}, {row['sub_class']}"
    )
    return metadata_context


def prepare_metadata_hiba(row: pd.Series) -> str:
    demo_str = (
        f"Age: {row['age_approx']}, "
        f"Sex: {row['sex']}, "
        f"Skin Type: {row['fitzpatrick_skin_type']}"
    )

    if row['personal_hx_mm'] and row['family_hx_mm']:
        hx_mm_str = "Patient and his/her family had melanoma."
    elif row['personal_hx_mm']:
        hx_mm_str = "Patient had melanoma."
    elif row['family_hx_mm']:
        hx_mm_str = "Family of patient had melanoma."
    else:
        hx_mm_str = "No melanoma history."

    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {row['diagnosis_1']}, {row['diagnosis_2']}, {row['diagnosis_3']}\n"
        f"- Patient Demographics: {demo_str}\n"
        f"- Melanoma History: {hx_mm_str}\n"
        f"- Anatomical Site: {row['anatom_site_general']}\n"
        f"- Biopsed: {row['concomitant_biopsy']}\n"
    )
    return metadata_context


def prepare_metadata_padufes20(row: pd.Series) -> str:
    """Prepare metadata string for the DERM12345 dataset"""
    label = padufes20_label_map[row['diagnostic']]

    demo_str = (
        f"Age: {row['age']}, "
        f"Gender: {row['gender']}, "
        f"Skin Type: {row['fitspatrick']}"
    )

    if row['skin_cancer_history']:
        history_str = "Patient Had Skin Cancer."
    elif row['cancer_history']:
        history_str = "Patient Had Cancer."
    else:
        history_str = "No Cancer History."

    diameter_str = (
        f"Horizontal: {row['diameter_1']} mm, "
        f"Vertical: {row['diameter_2']} mm"
    )

    symptoms_str = ""
    for symptom in ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation']:
        if row[symptom]: symptoms_str += symptom + ", "
    symptoms_str = symptoms_str[:-2] if symptoms_str else "None" # Remove trailing comma and space

    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {label}\n"
        f"- Patient Demographics: {demo_str}\n"
        f"- Medical History: {history_str}\n"
        f"- Affected Region: {row['region']}\n"
        f"- Size: {diameter_str}\n"
        f"- Symptoms: {symptoms_str}\n"
    )
    return metadata_context


def prepare_metadata_bcn20000(row: pd.Series) -> str:
    label = bcn20000_label_map[row['diagnosis']]
    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {label}\n"
        f"- Age: {row['age_approx']}\n"
        f"- Anatomical Site: {row['anatom_site_general']}\n"
        f"- Sex: {row['sex']}"
    )
    return metadata_context


def prepare_metadata_scin(row: pd.Series) -> str:
    """
    Prepare metadata string for the SCIN dataset.
    Explains the weighted label structure to the model.
    """
    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"The following shows the possible diagnoses with their confidence weights in parentheses.\n"
        f"The weights sum to 1.0, where higher weights indicate higher confidence in that diagnosis.\n"
        f"- Labels: {row['labels']}"
    )
    return metadata_context


def prepare_metadata_scin_clinical(row: pd.Series) -> str:
    '''
    # Format symptoms list
    symptoms_str = ', '.join(row['symptoms'])

    # Format body parts list
    body_parts_str = ', '.join(row['body_parts'])

    # Format textures list
    textures_str = ', '.join(row['textures'])

    # Format demographics
    demographics = row['demographics']
    demo_str = (
        f"Age Group: {demographics['age_group']}, "
        f"Sex: {demographics['sex_at_birth']}, "
        f"Skin Type: {demographics['fitzpatrick_skin_type']}, "
        f"Duration: {demographics['condition_duration']}"
    )
    '''
    demo_str = (
        f"Age Group: {row['age_group']}, "
        f"Sex: {row['sex_at_birth']}, "
        f"Skin Type: {row['fitzpatrick_skin_type']}, "
        f"Duration: {row['condition_duration']}"
    )

    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"Clinical Presentation:\n"
        '''
        f"- Reported Symptoms: {symptoms_str}\n"
        f"- Affected Areas: {body_parts_str}\n"
        f"- Physical Characteristics: {textures_str}\n"
        '''
        f"- Reported Symptoms: {row['symptoms']}\n"
        f"- Affected Areas: {row['body_parts']}\n"
        f"- Physical Characteristics: {row['textures']}\n"
        f"- Patient Demographics: {demo_str}\n\n"
        f"Diagnostic Assessment:\n"
        f"The following shows the possible diagnoses with their confidence weights in parentheses.\n"
        f"The weights sum to 1.0, where higher weights indicate higher confidence in that diagnosis.\n"
        '''
        f"- Diagnostic Considerations: {row['labels']}"
        '''
    )

    return metadata_context


def prepare_metadata(dataset:str, row:pd.Series):
        if dataset == "derm12345":
            return prepare_metadata_derm12345(row)
        elif dataset == "bcn20000":
            return prepare_metadata_bcn20000(row)
        elif dataset == "pad-ufes-20":
            return prepare_metadata_padufes20(row)
        elif dataset == "scin":
            return prepare_metadata_scin(row)
        elif dataset == "scin_clinical":
            return prepare_metadata_scin_clinical(row)
        elif dataset == "hiba":
            return prepare_metadata_hiba(row)
        else:
            raise ValueError("Invalid dataset name. Please provide a valid dataset name.")
        

def filter_one_condition_scin_clinical(metadata:pd.DataFrame):
    label_col = label_col_map["scin_clinical"]

    # Filter the images with only one diagnosis label
    mask = metadata[label_col].apply(lambda x: len(ast.literal_eval(x)) == 1)
    metadata = metadata.loc[mask].copy()  # Avoid SettingWithCopyWarning

    # Convert the label column from a list to a single value
    metadata[label_col] = metadata[label_col].apply(lambda x: ast.literal_eval(x)[0])

    return metadata


def validate_pred(dataset:str, pred:str, taxonomy:Union[dict, list], 
                  super_class_pred:str=None, main_class_pred:str=None, subclass_key:str="name"):
    if dataset == "derm12345":
        if super_class_pred ==  None:
            # Validate super class prediction
            class_labels = taxonomy.keys() 
        elif main_class_pred == None:
            # Validate main class prediction
            class_labels = taxonomy.get(super_class_pred).keys()
        else:
            # Validate subclass prediction
            class_labels = [item[subclass_key] for item in 
                            taxonomy.get(super_class_pred).get(main_class_pred)["subclasses"]]
    elif dataset  in {"hiba", "scin_clinical"}:
        class_labels = taxonomy
    elif dataset in {"bcn20000", "pad-ufes-20"}:
        class_labels = taxonomy.keys()

    if pred not in class_labels:
        pred = "malformed output" if pred != "unknown" else "unknown"
    return pred, pred in class_labels


def search_derm12345_by_label(taxonomy:dict, target_label:str):
    if target_label == "unknown":
        return "unknown", "unknown", "unknown"
    for super_class, main_classes in taxonomy.items():
        for main_class, data in main_classes.items():
            for subclass in data["subclasses"]:
                if subclass["label"] == target_label:
                    return super_class, main_class, subclass["name"]
    return "malformed output", "malformed output", "malformed output"


def get_main_classes_str(taxonomy:dict, target_super_classes:list):
    lines = []
    for super_class in target_super_classes:
        lines.append(f"{super_class}:")
        for main_class in taxonomy[super_class].keys():
            lines.append(f"  - {main_class}")
    return "\n".join(lines)


def get_subclasses_str(taxonomy:dict, target_main_classes:dict):
    lines = []
    for super_class, main_classes in target_main_classes.items():
        lines.append(f"{super_class}:")
        for main_class in main_classes.items():
            lines.append(f"  - {main_class}")
            for subclass in taxonomy[super_class][main_class]["subclasses"]:
                lines.append(f"    • {subclass['name']} (label: {subclass['label']})")
    return "\n".join(lines)


