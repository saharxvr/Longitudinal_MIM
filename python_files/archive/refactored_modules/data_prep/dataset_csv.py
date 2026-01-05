"""
Dataset CSV Creation and Filtering Utilities.

Functions for creating filtered CSV files for various CXR datasets,
including filtering by view position, findings, and labels.
"""

import os
from typing import List, Set, Optional
import pandas as pd
import numpy as np
from ast import literal_eval

from config.paths import (
    CXR14_FOLDER,
    MIMIC_CXR_PATH,
    PADCHEST_FOLDER,
    VINDR_FOLDER,
)


def create_no_finding_csv(
    dataset: str,
    output_path: str,
    view_position: str = 'AP'
) -> pd.DataFrame:
    """
    Create CSV of cases with 'No Finding' label.
    
    Args:
        dataset: Dataset name ('cxr14', 'mimic', 'padchest', 'vindr')
        output_path: Path to save the CSV
        view_position: Filter by view position ('AP', 'PA', 'both')
    
    Returns:
        DataFrame with filtered cases
    
    Example:
        >>> df = create_no_finding_csv('cxr14', 'no_finding_AP.csv', view_position='AP')
    """
    if dataset.lower() == 'cxr14':
        return _create_cxr14_no_finding_csv(output_path, view_position)
    elif dataset.lower() == 'mimic':
        return _create_mimic_no_finding_csv(output_path, view_position)
    elif dataset.lower() == 'padchest':
        return _create_padchest_no_finding_csv(output_path)
    elif dataset.lower() == 'vindr':
        return _create_vindr_no_finding_csv(output_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _create_cxr14_no_finding_csv(output_path: str, view_position: str) -> pd.DataFrame:
    """Create No Finding CSV for CXR-14 dataset."""
    data_path = os.path.join(CXR14_FOLDER, 'sheets', 'Data_Entry_2017_v2020.csv')
    data_df = pd.read_csv(data_path)
    
    # Load labels
    labels_paths = [
        os.path.join(CXR14_FOLDER, 'sheets', f'miccai2023_nih-cxr-lt_labels_{split}.csv')
        for split in ['train', 'val', 'test']
    ]
    df = pd.concat([pd.read_csv(p) for p in labels_paths])
    
    # Filter No Finding
    no_finding_df = df.loc[df['No Finding'] == 1]
    
    # Filter by view position
    if view_position != 'both':
        vp_df = data_df[data_df["View Position"] == view_position]
        vp_df = vp_df.rename(columns={"Image Index": "id"})
        result_df = pd.merge(no_finding_df, vp_df, how='inner', on=["id"]).reset_index()[['id']]
    else:
        result_df = no_finding_df[['id']]
    
    # Convert to NIfTI naming
    for i in range(len(result_df)):
        cur_im = result_df.iloc[i]['id'].split(".")[0] + '.nii.gz'
        result_df.at[i, 'id'] = cur_im
    
    result_df.to_csv(output_path, index=False)
    return result_df


def _create_mimic_no_finding_csv(output_path: str, view_position: str) -> pd.DataFrame:
    """Create No Finding CSV for MIMIC-CXR dataset."""
    # Load metadata
    meta_path = os.path.join(MIMIC_CXR_PATH, 'mimic-cxr-2.0.0-metadata.csv.gz')
    meta_df = pd.read_csv(meta_path)
    
    # Filter AP views
    if view_position == 'AP':
        meta_df = meta_df.loc[
            (meta_df['ViewPosition'] == 'AP') | 
            (meta_df['PerformedProcedureStepDescription'] == 'CHEST (PORTABLE AP)') |
            (meta_df['ProcedureCodeSequence_CodeMeaning'] == 'CHEST (PORTABLE AP)') |
            (meta_df['ViewCodeSequence_CodeMeaning'] == 'antero-posterior')
        ]
    
    meta_df = meta_df[['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime']]
    
    # Load NegBio and CheXpert labels
    negbio_path = os.path.join(MIMIC_CXR_PATH, 'mimic-cxr-2.0.0-negbio.csv.gz')
    chexpert_path = os.path.join(MIMIC_CXR_PATH, 'mimic-cxr-2.0.0-chexpert.csv.gz')
    
    negbio_df = pd.read_csv(negbio_path)
    negbio_df = negbio_df.loc[negbio_df['No Finding'] == 1.0]
    
    chexpert_df = pd.read_csv(chexpert_path)
    chexpert_df = chexpert_df.loc[chexpert_df['No Finding'] == 1.0]
    
    # Merge for consensus
    combined_df = pd.merge(chexpert_df, negbio_df, how='inner', on=['subject_id', 'study_id'])
    combined_df = pd.merge(combined_df, meta_df, how='inner', on=['subject_id', 'study_id'])
    
    # Remove duplicates, keeping first by time
    combined_df = combined_df.sort_values(by=['StudyDate', 'StudyTime'])
    combined_df = combined_df.drop_duplicates(['subject_id', 'study_id'], keep='first')
    combined_df = combined_df.reset_index()[['subject_id', 'study_id', 'dicom_id']]
    
    # Convert to cropped NIfTI naming
    for i in range(len(combined_df)):
        cur_im = combined_df.iloc[i]['dicom_id'].split(".")[0] + '_cropped.nii.gz'
        combined_df.at[i, 'dicom_id'] = cur_im
    
    combined_df.to_csv(output_path, index=False)
    return combined_df


def _create_padchest_no_finding_csv(output_path: str) -> pd.DataFrame:
    """Create No Finding CSV for PadChest dataset."""
    csv_path = os.path.join(PADCHEST_FOLDER, 'relevant_images.csv')
    df = pd.read_csv(csv_path)
    
    # Filter 'normal' only
    df_normal = df[df["Labels"] == "[\'normal\']"]["ImageID"]
    df_normal.to_csv(output_path, index=False)
    
    return df_normal


def _create_vindr_no_finding_csv(output_path: str) -> pd.DataFrame:
    """Create No Finding CSV for VinDr-CXR dataset."""
    csv_path = os.path.join(VINDR_FOLDER, 'labels_train.csv')
    df = pd.read_csv(csv_path)
    
    # Average across radiologists and round
    all_df = df.groupby('image_id').mean().reset_index().round()
    all_df[all_df.columns.difference(['image_id'])] = \
        all_df[all_df.columns.difference(['image_id'])].astype(int)
    
    # Filter No Finding
    no_finding_df = all_df[all_df['No finding'] == 1]['image_id']
    no_finding_df.to_csv(output_path, index=False)
    
    return no_finding_df


def create_specific_abnormalities_csv(
    dataset: str,
    output_path: str,
    abnormalities: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Create CSV of cases with specific abnormalities.
    
    Args:
        dataset: Dataset name
        output_path: Path to save the CSV
        abnormalities: Set of abnormality names to include. If None, uses defaults.
    
    Returns:
        DataFrame with filtered cases
    """
    if abnormalities is None:
        abnormalities = {
            'pulmonary mass', 'pneumothorax', 'pneumonia', 'pulmonary edema',
            'consolidation', 'nodule', 'pleural effusion', 'infiltrates',
            'atelectasis', 'cardiomegaly'
        }
    
    if dataset.lower() == 'padchest':
        return _create_padchest_abnormalities_csv(output_path, abnormalities)
    else:
        raise NotImplementedError(f"Abnormality filtering for {dataset} not implemented")


def _create_padchest_abnormalities_csv(
    output_path: str, 
    abnormalities: Set[str]
) -> pd.DataFrame:
    """Create abnormalities CSV for PadChest."""
    csv_path = os.path.join(PADCHEST_FOLDER, 'relevant_images.csv')
    df = pd.read_csv(csv_path)
    
    # Parse labels
    df['Labels'] = df.Labels.apply(literal_eval)
    
    # Create binary columns for each abnormality
    for label in abnormalities:
        df[label] = df.Labels.apply(lambda x: label in set(x))
    
    # Filter to only cases with at least one abnormality
    df['has_abnormality'] = df.apply(
        lambda row: any(row[l] for l in abnormalities), axis=1
    )
    df = df[df['has_abnormality'] == True]
    
    # Select relevant columns
    df = df[['ImageID'] + list(abnormalities)]
    df[list(abnormalities)] = df[list(abnormalities)].astype(int)
    
    df.to_csv(output_path, index=False)
    return df


def filter_relevant_images(
    input_csv: str,
    output_csv: str,
    view_positions: Optional[List[str]] = None,
    method_label: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter dataset CSV to relevant images only.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to save filtered CSV
        view_positions: List of allowed view positions
        method_label: Filter by labeling method (e.g., 'Physician')
    
    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(input_csv)
    
    if method_label is not None:
        df = df[df["MethodLabel"] == method_label]
    
    if view_positions is not None:
        df = df[
            df["ViewPosition_DICOM"].isin(view_positions) | 
            df["Projection"].isin(view_positions)
        ]
    
    df.to_csv(output_csv, index=False)
    return df
