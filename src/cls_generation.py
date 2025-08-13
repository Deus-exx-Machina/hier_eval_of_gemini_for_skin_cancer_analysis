import os
import re

import json
import pandas as pd
from PIL import Image
from typing import Union

from dataset_utility import label_col_map, get_paths, prepare_metadata
from gemini_api import GeminiAPIHandler

base_dir = os.path.dirname(__file__)


# Utility function for prompt preparation
def insert_relative_to_line(text:str, target_line:str, insert_text:str, position:str="after"):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == target_line.strip():
            if position == 'before':
                lines.insert(i, insert_text)
            elif position == 'after':
                lines.insert(i + 1, insert_text)
            break
    return '\n'.join(lines)


def prepare_prompt(dataset:str, taxonomy:Union[dict, list], metadata_row:pd.Series, 
                   isMultiStep:bool=False, super_class_pred:str=None, main_class_pred:str=None, use_context:bool=False):
    prompt_templates_file = os.path.join(base_dir, f"prompt_templates_for_cls.json")
    with open(prompt_templates_file, 'r') as ptf:
        prompt_templates = json.load(ptf)

    taxonomy_str = json.dumps(taxonomy, indent=4)

    if dataset == "derm12345":
        if not isMultiStep:
            prompt = '\n'.join(prompt_templates["Derm12345_single_step"]).format(taxonomy=taxonomy_str)
        elif super_class_pred == None:
            prompt = '\n'.join(prompt_templates["Other"] + ["<ClassName>"]).format(taxonomy='\n'.join(taxonomy.keys()))
        elif main_class_pred == None:
            prompt = '\n'.join(prompt_templates["Other"] + ["<ClassName>"]) \
                         .format(taxonomy='\n'.join(taxonomy.get(super_class_pred).keys()))
        else:
            subclasses_str = "\n".join(
                f'{i}. name: {entry["name"]}, label: {entry["label"]}'
                for i, entry in enumerate(taxonomy.get(super_class_pred).get(main_class_pred)["subclasses"], start=1)
            )
            prompt = '\n'.join(prompt_templates["Other"] + ["<ClassName> - <Label>"]) \
                         .format(taxonomy=subclasses_str)
        
        '''
        elif super_class_pred == None:
            prompt = '\n'.join(prompt_templates["Derm12345_super_class"] + prompt_templates["Derm12345_multi_step_rules"]) \
                         .format(super_classes='\n'.join(taxonomy.keys()), 
                                 response_format="<super-class>",
                                 granularity="SUPER-CLASSES")
        elif main_class_pred == None:
            prompt = '\n'.join(prompt_templates["Derm12345_main_class"] + prompt_templates["Derm12345_multi_step_rules"]) \
                         .format(super_class=super_class_pred,
                                 main_classes='\n'.join(taxonomy.get(super_class_pred).keys()),
                                 response_format="<main-class>",
                                 granularity="MAIN-CLASSES")
        else:
            subclasses_str = "\n".join(
                f'{i}. name: {entry["name"]}, label: {entry["label"]}'
                for i, entry in enumerate(taxonomy.get("melanocytic benign").get("banal compound")["subclasses"], start=1)
            )
            prompt = '\n'.join(prompt_templates["Derm12345_subclass"] + prompt_templates["Derm12345_multi_step_rules"]) \
                         .format(super_class=super_class_pred, main_class=main_class_pred,
                                 subclasses=subclasses_str,
                                 response_format="<subclass> - <label>",
                                 granularity="SUBCLASSES")
        prompt = '\n'.join(prompt_templates["Derm12345_system"] + [prompt])
        '''
    else:
        prompt = '\n'.join(prompt_templates["Other"]).format(taxonomy=taxonomy_str)
        prompt = append_context(dataset, prompt, metadata_row, use_context)
        prompt = supplement_prompt(dataset, prompt)
    return prompt


def append_context(dataset:str, prompt:str, metadata_row:pd.Series, use_context:bool=False):
    metadata_context = prepare_metadata(dataset, metadata_row)
    metadata_context = remove_diagnosis(dataset, metadata_context)
    metadata_str = (
        f"***** CONTEXT *****\n"
        f"{metadata_context}\n"
    )
    prompt = insert_relative_to_line(prompt, "***** GOAL *****", metadata_str, "before")

    if use_context:
        instruction_supplement = (
            f"Base your answer both on the visible characteristics of the lesion/skin disease "
            f"and the context provided below."
        )
    else:
        prompt = insert_relative_to_line(prompt, "***** CONTEXT *****", "(FOR INTERNAL REFERENCE ONLY)", "after")
        instruction_supplement = (
            f"Although some context is provided below, do NOT rely on it in your answer.\n"
            f"Base your answer **solely** on the visible characteristics of the lesion/skin disease in the image."
        )
    instruction_line_1 = "You are a dermatologist examining a lesion/skin disease image."
    prompt = insert_relative_to_line(prompt, instruction_line_1, instruction_supplement, "after")
        
    return prompt


def remove_diagnosis(dataset:str, metadata_context:str):
    if dataset == "scin_clinical": 
        return metadata_context.split("Diagnostic Assessment:")[0]
    else:
        # Removes any line that starts with "- Label:"
        return re.sub(r'^- Label:.*\n?', '', metadata_context, flags=re.MULTILINE)
    

def supplement_prompt(dataset:str, prompt:str):
    # Append reponse format and description of taxonomy
    if dataset  in {"hiba", "scin_clinical"}:
        prompt = insert_relative_to_line(prompt, "***** RESPONSE FORMAT *****", "<ClassName>", position="after")
    elif dataset in {"bcn20000", "pad-ufes-20"}:
        prompt = insert_relative_to_line(prompt, "***** RESPONSE FORMAT *****", 
                                         "<ClassName> - <Abbreviation>", position="after") 
        prompt = insert_relative_to_line(prompt, "***** TAXONOMY *****", 
                                         "The taxonomy is listed as: <Abbreviation>: <ClassName>", position="after")
    
    return prompt

####################################################################################################

def generate_raw_cls_response(api_handler:GeminiAPIHandler, dataset:str, taxonomy:Union[dict, list], 
                              image_path:str, metadata_row:pd.Series, 
                              isMultiStep:bool=False, super_class_pred:str=None, main_class_pred:str=None,
                              use_context:bool=False):
    prompt = prepare_prompt(dataset, taxonomy, metadata_row, 
                            isMultiStep, super_class_pred, main_class_pred, use_context)

    # Load the image
    with Image.open(image_path) as pil_image:
        # Generate response
        response = api_handler.generate_from_pil_image(pil_image, prompt)
    return response


def call_gemini_api_cls(api_handler:GeminiAPIHandler, dataset:str, 
                        image_id:str, metadata_row:pd.Series, 
                        isMultiStep:bool=False, use_context:bool=False) -> dict:
    """
    Call the Gemini API to perform classification on an image
    """
    dataset_path, image_path = get_paths(dataset, image_id)
    taxonomy_file = os.path.join(dataset_path, "taxonomy.json")
    with open(taxonomy_file, 'r') as tf:
        taxonomy = json.load(tf)

    result = {
        "image_id": image_id, 
        "image_path": image_path
    } # Initialize result dictionary
    
    if isMultiStep:
        multistep_cls(api_handler, dataset, taxonomy, image_id, image_path, metadata_row, result)
        supplement_derm12345_result(result, taxonomy, metadata_row)
    else:
        response = generate_raw_cls_response(api_handler, dataset, taxonomy, 
                                             image_path, metadata_row, use_context=use_context).rstrip('\n')
        result.update({"response": response})

        if dataset == "derm12345":
            supplement_derm12345_result(result, taxonomy, metadata_row, response)
        else:
            supplement_other_result(result, dataset, taxonomy, metadata_row, response)

    return result


def multistep_cls(api_handler:GeminiAPIHandler, dataset:str, taxonomy:Union[dict, list],
                  image_id:str, image_path:str, metadata_row:pd.Series, result:dict) -> dict:
    main_class_pred = subclass_response = subclass_pred = label_pred = "unknown"

    logger = api_handler.logger
    handler_index = api_handler.index

    super_class_pred = generate_raw_cls_response(api_handler, dataset, taxonomy, 
                                                 image_path, metadata_row, isMultiStep=True).rstrip('\n').lower()
    super_class_pred, isValid = validate_pred(dataset, super_class_pred, taxonomy)
    if not isValid:
        logger.warning(f"[Handler_{handler_index}] Failed to predict the super class of {image_id}.")
    else:
        logger.info(f"[Handler_{handler_index}] Successfully predicted the super class {super_class_pred} " +
                    f"of {image_id}. Proceed to main class prediction.")
        main_class_pred = generate_raw_cls_response(api_handler, dataset, taxonomy, image_path, metadata_row, 
                                                    isMultiStep=True, super_class_pred=super_class_pred).rstrip('\n').lower()
        main_class_pred, isValid = validate_pred(dataset, main_class_pred, taxonomy, super_class_pred=super_class_pred)
        if not isValid:
            logger.warning(f"[Handler_{handler_index}] Failed to predict the main class of {image_id}.")
        else:
            logger.info(f"[Handler_{handler_index}] Successfully predicted the main class {main_class_pred} " +
                        f"of {image_id}. Proceed to subclass prediction.")
            subclass_response = generate_raw_cls_response(api_handler, dataset, taxonomy, image_path, metadata_row, 
                                                          isMultiStep=True, super_class_pred=super_class_pred, 
                                                          main_class_pred=main_class_pred).rstrip('\n').lower()
            if len(subclass_response.split(' - ')) == 2:
                subclass_pred, label_pred = subclass_response.split(' - ')
                subclass_pred, _ = validate_pred(dataset, subclass_pred, taxonomy,
                                                 super_class_pred=super_class_pred, main_class_pred=main_class_pred, 
                                                 subclass_key="name")
                label_pred, _ = validate_pred(dataset, label_pred, taxonomy, 
                                              super_class_pred=super_class_pred, main_class_pred=main_class_pred,
                                              subclass_key="label")
            elif subclass_response != "unknown":
                subclass_pred = label_pred = "malformed output"
                
    if super_class_pred == "malformed output": 
        main_class_pred = subclass_pred = label_pred = "malformed super class output"
    if main_class_pred == "malformed output": 
        subclass_pred = label_pred = "malformed main class output"
    elif main_class_pred != "unknown": 
        main_class_pred = super_class_pred + ' - ' + main_class_pred

    result.update({
        "super_class_pred": super_class_pred,
        "main_class_pred": main_class_pred,
        "subclass_response": subclass_response,
        "subclass_pred": subclass_pred,
        "label_pred": label_pred,
    })

####################################################################################################

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


def supplement_derm12345_result(result:dict,  taxonomy:Union[dict, list], metadata_row:pd.Series, response:str=""):
    if response != "":
        parts = response.rstrip('\n').split(' - ')
        super_class_pred = main_class_pred = subclass_pred = label_pred = "malformed output"

        if len(parts) == 4:
            super_class_pred, isValid = validate_pred("derm12345", parts[0].lower(), taxonomy)
            if isValid:
                main_class_pred, isValid = validate_pred("derm12345", parts[1].lower(), taxonomy, 
                                                     super_class_pred=super_class_pred)
                if isValid:
                    subclass_pred, _ = validate_pred("derm12345", parts[2].lower(), taxonomy,
                                                     super_class_pred=super_class_pred, main_class_pred=main_class_pred, 
                                                     subclass_key="name")
                    label_pred, _ = validate_pred("derm12345", parts[3].lower(), taxonomy,
                                                  super_class_pred=super_class_pred, main_class_pred=main_class_pred,
                                                  subclass_key="label")
                    
        if main_class_pred not in ["malformed output", "unknown"]:        
            main_class_pred = super_class_pred + ' - ' + main_class_pred

        result.update({
            "super_class_pred": super_class_pred,
            "main_class_pred": main_class_pred,
            "subclass_pred": subclass_pred,
            "label_pred": label_pred
        })

    super_class_true = str(metadata_row['super_class']) + ' ' + str(metadata_row['malignancy'])
    if str(metadata_row['main_class_1']) == str(metadata_row['main_class_2']):
        main_class_true = super_class_true + ' - ' + str(metadata_row['main_class_1'])
    else:
        main_class_true = super_class_true + ' - ' + \
                        str(metadata_row['main_class_1']) + ' ' + str(metadata_row['main_class_2'])
    subclass_true = str(metadata_row['sub_class'])
    label_true = str(metadata_row['label'])

    result.update({
        "super_class_true": super_class_true,
        "main_class_true": main_class_true,
        "subclass_true": subclass_true,
        "label_true": label_true
    })


def supplement_other_result(result:dict, dataset: str, taxonomy:Union[dict, list],
                            metadata_row:pd.Series, response:str):
    if dataset == "bcn20000":
        metadata_info = metadata_row[['age_approx', 'anatom_site_general', 'lesion_id', 'sex']]
    elif dataset == "hiba":
        metadata_info = metadata_row[['age_approx', 'anatom_site_general', 'anatom_site_special', 'concomitant_biopsy', 
                                      'dermoscopic_type', 'diagnosis_confirm_type', 'fitzpatrick_skin_type', 
                                      'image_type', 'lesion_id', 'patient_id', 'sex']]
    elif dataset == "pad-ufes-20":
        metadata_info = metadata_row[['lesion_id', 'patient_id', 'age', 'pesticide', 'gender', 'fitspatrick', 'region',	
                                      'diameter_1', 'diameter_2', 'img_id', 'biopsed']]
    elif dataset == "scin_clinical":
        metadata_info = metadata_row[['case_id', 'age_group', 'sex_at_birth', 'fitzpatrick_skin_type', 'related_category', 
                                      'condition_duration', 'combined_race', 'dermatologist_skin_condition_on_label_name', 
                                      'dermatologist_skin_condition_confidence', 'textures', 'body_parts', 'symptoms']]
    
    diagnosis_true = metadata_row[label_col_map[dataset]]
    result.update(metadata_info.fillna('unknown').replace(r'^\s*$', "unknown", regex=True).to_dict())

    diagnosis_pred = "malformed output"
    if dataset == "hiba" or dataset == "scin_clinical":
        diagnosis_pred, _ = validate_pred(dataset, response.lower(), taxonomy)
    elif dataset == "bcn20000" or dataset == "pad-ufes-20":
        parts = response.rstrip('\n').split(' - ')
        if len(parts) == 2:
            diagnosis_pred, _ = validate_pred(dataset, parts[1].upper(), taxonomy)

    result.update({
        "diagnosis_pred": diagnosis_pred,
        "diagnosis_true": diagnosis_true
    })
                           