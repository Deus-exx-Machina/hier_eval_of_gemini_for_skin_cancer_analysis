import os
import json

import numpy as np
import pandas as pd
from IPython.display import display

from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

base_dir = os.path.dirname(__file__)
eval_format = 'xlsx' # Default format for saving evaluation files
#eval_format = 'csv'


def create_eval_dir(dataset_dirname:str):
    eval_dir = os.path.join(base_dir, f"api_cls/{dataset_dirname}/evaluation_metrics")
    os.makedirs(os.path.dirname(eval_dir), exist_ok=True)
    return eval_dir


def save_eval(eval_df:pd.DataFrame, eval_dir:str, filename:str, format:str='csv' or 'xlsx', index:bool=False):
    filepath = os.path.join(eval_dir, f"{filename}.{format}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if format == 'csv': 
        eval_df.to_csv(filepath, index=index)
    elif format == 'xlsx':
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            eval_df.to_excel(writer, sheet_name='Sheet1', index=index)
            worksheet = writer.sheets['Sheet1'] # Access the workbook and sheet
            # Adjust column widths based on maximum content length
            for column_cells in worksheet.columns:
                length = max(len(str(cell.value or "")) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2


def cal_specificity(labels: pd.Series, confusion_mat: np.ndarray, average:str=None):
    specificities = []

    for i, label in enumerate(labels):
        TP = confusion_mat[i, i]
        FN = confusion_mat[i, :].sum() - TP
        FP = confusion_mat[:, i].sum() - TP
        TN = confusion_mat.sum() - (TP + FN + FP)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(specificity)

    if average == None:
        return specificities
    elif average == "macro":
        return np.mean(specificities)
    else:
        raise ValueError(f"Invalid Argument: average={average}")


def cal_invalid_pred_rates(y_pred:pd.Series):
    total = len(y_pred)
    malformed_count = (y_pred == 'malformed output').sum()
    abstention_count = (y_pred == 'unknown').sum()
    return pd.Series({
        'Malformed Rate': malformed_count / total,
        'Abstention Rate': abstention_count / total
    })
    

def eval_overall(y_true:pd.Series, y_pred:pd.Series, isprocessing:bool=False, granularity:str=None, metadata_split:str=None):
    labels = sorted(set(y_true))  # all unique labels in ground truth
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sensitivity = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    specificity = cal_specificity(labels, cm, average="macro")
    invalid_rates = cal_invalid_pred_rates(y_pred)
    eval_metrics = {
        'Accuracy': f"{accuracy:.3%}",
        'Sensitivity (Macro)': f"{sensitivity:.3%}",
        'Specificity (Macro)': f"{specificity:.3%}",
        'Malformed Rate': f"{invalid_rates['Malformed Rate']:.3%}",
        'Abstention Rate': f"{invalid_rates['Abstention Rate']:.3%}"
    }
    message = f"Accuracy={accuracy:.3%}, Sensitivity={sensitivity:.3%}, Specificity={specificity:.3%}"

    if metadata_split is not None:
        eval_metrics = {'Metadata_Split': f'{metadata_split}', **eval_metrics}
        message = f"- {metadata_split}: " + message

    if granularity is not None: 
        eval_metrics = {'Granularity': f'{granularity}', **eval_metrics}
        message = f"• **{granularity} Level: " + message

    if isprocessing:
        tqdm.write(message) # Temporarily overwrite the progress bar
    else:
        print(message)
    return pd.DataFrame([eval_metrics])


def eval_per_class(y_true:pd.Series, y_pred:pd.Series, eval_dir:str, granularity:str=None, metadata_split:str=None):   
    labels = sorted(set(y_true))  # all unique labels in ground truth
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_filename = f"{granularity}_confusion_matrix" if granularity is not None else "confusion_matrix"
    save_eval(cm_df, eval_dir, cm_filename, format=eval_format, index=True)

    sensitivities = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    specificities = cal_specificity(labels, cm, average=None)
    
    # Calculate the invalid prediction rates by class
    y_df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    invalid_rates = y_df.groupby('true')['pred'].apply(cal_invalid_pred_rates).unstack()
    invalid_rates = invalid_rates.reindex(labels, fill_value=0.0) # reindex to match provided class label order

    eval_data = {
        'Class': labels,
        'Sensitivity': sensitivities,
        'Specificity': specificities,
        'Malformed Rate': invalid_rates['Malformed Rate'],
        'Abstention Rate': invalid_rates['Abstention Rate']
    }
    eval_df = pd.DataFrame(eval_data).reset_index(drop=True)
    display(eval_df)
    
    eval_filename = f"per_{granularity}_class" if granularity is not None else "per_class"

    if metadata_split is not None: eval_filename = f"{metadata_split}_" + eval_filename
    elif granularity is None: eval_filename = f"whole_dataset_" + eval_filename

    save_eval(eval_df, eval_dir, eval_filename, format=eval_format)
    return eval_df


def eval_derm12345(results:pd.DataFrame, isMultiStep:bool=False, isprocessing:bool=False):
    overall_eval = pd.DataFrame()
    dataset_dirname = "derm12345"
    if isMultiStep:
        dataset_dirname += "_multistep"
    eval_dir = create_eval_dir(dataset_dirname)

    super_overall_eval = eval_overall(results["super_class_true"], results["super_class_pred"], isprocessing, "SuperClass")
    main_overall_eval = eval_overall(results["main_class_true"], results["main_class_pred"], isprocessing, "MainClass")
    sub_overall_eval = eval_overall(results["label_true"], results["label_pred"], isprocessing, "SubClass")
    overall_eval = pd.concat([super_overall_eval, main_overall_eval, sub_overall_eval], ignore_index=True)
    
    if isprocessing == False:
        save_eval(overall_eval, eval_dir, "overall_evaluation", format=eval_format)

        per_super_class = eval_per_class(results["super_class_true"], results["super_class_pred"], eval_dir, "super")
        per_main_class = eval_per_class(results["main_class_true"], results["main_class_pred"], eval_dir, "main")
        per_sub_class = eval_per_class(results["label_true"], results["label_pred"], eval_dir, "sub")

        # Create hierarchical evaluation files
        taxonomy_file = os.path.join(base_dir, "../datasets/derm12345/taxonomy.json")
        with open(taxonomy_file, 'r') as dh:
            taxonomy_tree = json.load(dh)

        for super_class_key, super_class_dict in taxonomy_tree.items():
            one_super_eval = per_super_class[per_super_class['Class'] == super_class_key]

            for main_class_key, main_class_dict in super_class_dict.items():
                main_class_key = super_class_key + ' - ' + main_class_key
                one_main_eval = per_main_class[per_main_class['Class'] == main_class_key]
                one_super_eval = pd.concat([one_super_eval, one_main_eval], ignore_index=True)

                for subclass in main_class_dict['subclasses']:
                    one_sub_eval = per_sub_class[per_sub_class['Class'] == subclass.get('label')]
                    one_main_eval = pd.concat([one_main_eval, one_sub_eval], ignore_index=True)

                save_eval(one_main_eval, eval_dir, main_class_key, format=eval_format)
            save_eval(one_super_eval, eval_dir, super_class_key, format=eval_format)


def eval_bcn20000(results:pd.DataFrame, isprocessing:bool=False):
    eval_dir = create_eval_dir("bcn20000")

    overall_eval = eval_overall(results["diagnosis_true"], results["diagnosis_pred"], isprocessing)

    results["age_approx"] = pd.to_numeric(results["age_approx"], errors="coerce") # Convert age to numeric, coercing errors to NaN
    results["age_decade_floor"] = (results["age_approx"] // 10) * 10
    results["age_decade_floor"] = results["age_decade_floor"].astype('Int64') # Convert to nullable integer type

    for field in ["age_decade_floor", "anatom_site_general", "sex"]:
        non_empty_values = np.sort(results[field].dropna().unique())
        if field != 'age_decade_floor':
            non_empty_values = [x for x in non_empty_values if str(x).strip() != '']

        for value in non_empty_values:
            if field == 'age_decade_floor':
                metadata_split = f"age={value}-{value + 9}"
            else:
                metadata_split = f"{field}={value.replace(' ', '_').replace('/', '_or_')}"
                
            split_overall_eval = eval_overall(results["diagnosis_true"][results[field] == value], 
                                              results["diagnosis_pred"][results[field] == value],
                                              isprocessing, metadata_split=metadata_split)
            overall_eval = pd.concat([overall_eval, split_overall_eval], ignore_index=True)

            if isprocessing == False:
                eval_per_class(results["diagnosis_true"][results[field] == value], 
                               results["diagnosis_pred"][results[field] == value], 
                               eval_dir, metadata_split=metadata_split)
            
    if isprocessing == False:
        save_eval(overall_eval, eval_dir, "overall_evaluation", format=eval_format)
        print("***Per Class of the Whole Dataset***")
        eval_per_class(results["diagnosis_true"], results["diagnosis_pred"], eval_dir)


def eval_other_dataset(results:pd.DataFrame, dataset:str, isprocessing:bool=False):
    eval_dir = create_eval_dir(dataset)
    overall_eval = eval_overall(results["diagnosis_true"], results["diagnosis_pred"], isprocessing)
    if isprocessing == False:
        save_eval(overall_eval, eval_dir, "overall_evaluation", format=eval_format)
        eval_per_class(results["diagnosis_true"], results["diagnosis_pred"], eval_dir)


def eval_gemini_cls(results:pd.DataFrame, dataset:str, isMultiStep:bool=False, isprocessing:bool=False):
    if dataset == "derm12345":
        eval_derm12345(results, isMultiStep, isprocessing)
    elif dataset == "bcn20000":
        eval_bcn20000(results, isprocessing)
    elif dataset in ["hiba", "pad-ufes-20", "scin_clinical"]:
        eval_other_dataset(results, dataset, isprocessing)
    else:
        raise ValueError("Invalid dataset name. Please provide a valid dataset name.")