"""
Data Configuration
==================

Settings for data loading, preprocessing, augmentation, and label handling.

Sections:
---------
1. Label Definitions
2. Dataset-Specific Mappings
3. Label Group Settings
4. String Constants

Usage:
------
    from config.data_config import CUR_LABELS, CSVS_TO_LABEL_MAPPING
    from config.paths import VINDR_FOLDER, PADCHEST_FOLDER
"""

from config.paths import VINDR_FOLDER, PADCHEST_FOLDER

# =============================================================================
# LABEL DEFINITIONS
# =============================================================================

# Full set of possible labels across all datasets (CXR-14 based)
ALL_LABELS: list = [
    'Atelectasis',      # Lung collapse
    'Cardiomegaly',     # Enlarged heart
    'Consolidation',    # Lung opacity/infiltration
    'Edema',            # Pulmonary edema
    'Effusion',         # Pleural effusion
    'Emphysema',        # Lung hyperinflation
    'Fibrosis',         # Pulmonary fibrosis
    'Hernia',           # Hiatal hernia
    'Infiltration',     # Lung infiltrate
    'Mass',             # Pulmonary mass
    'Nodule',           # Pulmonary nodule
    'Pleural Thickening',  # Thickened pleura
    'Pneumonia',        # Lung infection
    'Pneumothorax'      # Air in pleural space
]
"""
Complete list of pathology labels from CXR-14 dataset.
Used as reference for label indexing across datasets.
14 labels total (hence "Chest X-ray 14").
Used by: datasets.py - projection matrix, label encoding
"""

ALL_LABELS_NUM: int = 14
"""
Total number of labels in ALL_LABELS.
Used for one-hot encoding dimensions.
"""

CUR_LABELS: list = ['Abnormal', 'Normal']
"""
Currently active labels for training/evaluation.
Can be changed based on task:
- Binary: ['Abnormal', 'Normal'] 
- Multi-label: subset of ALL_LABELS
- Grouped: ['Localized', 'Interstitial', 'Cardiomegaly']

Examples of other configurations (uncomment to use):
# CUR_LABELS = ['Consolidation', 'Edema', 'Infiltration', 'Mass', 'Nodule']
# CUR_LABELS = ['Pneumothorax', 'Atelectasis', 'Consolidation', 'Nodule_Mass', 'ILD', 'Fibrosis']
# CUR_LABELS = ['Localized', 'Interstitial', 'Cardiomegaly']

Used by: datasets.py, evaluation scripts
"""

CUR_LABEL_GROUPS: list = None
"""
Optional grouping of labels into super-categories.
Format: List of lists, each inner list contains indices into CUR_LABELS.
None = no grouping, use CUR_LABELS directly.

Example:
# CUR_LABEL_GROUPS = [[0, 2, 3, 4], [1]]  # Group labels 0,2,3,4 and label 1 separately

Used by: datasets.py - ContrastiveLearningDataset
"""

GET_WEIGHTS: list = None
"""
Class weights for handling imbalanced datasets.
None = uniform weights.
List of floats = manual weights per class.

Example:
# GET_WEIGHTS = [2.5, 1., 4.]  # Weight rare classes higher
# GET_WEIGHTS = [2., 1.5, 2.5, 2., 2., 3.]

Used by: datasets.py - weighted sampling, loss weighting
"""

LABEL_NO_FINDING: bool = False
"""
Whether to include 'No Finding' as explicit label.
True: Add extra class for healthy images
False: Healthy = no positive labels

Used by: datasets.py - label encoding
"""

NO_FINDING_PROB_FACTOR: float = 4.0
"""
Sampling weight factor for 'No Finding' class.
Higher = more frequent sampling of normal images.
Helps balance dataset with many abnormal cases.
Used by: datasets.py - ContrastiveLearningDataset sampling
"""

UNLABELED_PERC: float = 0.0
"""
Percentage of data to treat as unlabeled (for semi-supervised).
0.0 = fully supervised training.
0.0-1.0 = hide labels for this fraction of data.
Used by: semi-supervised training experiments
"""

# Derived constant: number of output classes
if not CUR_LABEL_GROUPS:
    CUR_LABELS_NUM: int = len(CUR_LABELS) + 1 if LABEL_NO_FINDING else len(CUR_LABELS)
else:
    CUR_LABELS_NUM: int = len(CUR_LABEL_GROUPS) + 1 if LABEL_NO_FINDING else len(CUR_LABEL_GROUPS)
"""
Total number of classes for classification.
Includes 'No Finding' if LABEL_NO_FINDING=True.
Used by: models.py - classification head output size
"""

# =============================================================================
# DATASET-SPECIFIC LABEL MAPPINGS
# =============================================================================

_VINDR_LABEL_MAPPING: dict = {
    'Localized': ['Consolidation', 'Nodule/Mass', 'Infiltration'],
    'Interstitial': ['Edema', 'ILD', 'Pulmonary fibrosis'],
    'Pneumothorax': ['Pneumothorax'],
    'Atelectasis': ['Atelectasis'],
    'Consolidation': ['Consolidation'],
    'Cardiomegaly': ['Cardiomegaly'],
    'Nodule_Mass': ['Nodule/Mass'],
    'ILD': ['ILD'],
    'Fibrosis': ['Pulmonary fibrosis'],
    'Normal': ['No finding']
}
"""
Maps project labels to VinDr-CXR dataset label names.
VinDr uses different naming conventions.
'Localized' and 'Interstitial' are grouped categories.
"""

CSVS_TO_LABEL_MAPPING: dict = {
    f'{VINDR_FOLDER}/labels_train.csv': _VINDR_LABEL_MAPPING,
    f'{VINDR_FOLDER}/labels_test.csv': _VINDR_LABEL_MAPPING,
    f'{PADCHEST_FOLDER}/specific_abnormalities_labels.csv': {
        'Localized': ['consolidation', 'pulmonary mass', 'nodule', 'infiltrates'],
        'Interstitial': ['pulmonary edema', 'pulmonary fibrosis', 
                        'reticulonodular interstitial pattern', 
                        'reticular interstitial pattern', 
                        'interstitial pattern'],
        'Pneumothorax': ['pneumothorax'],
        'Atelectasis': ['atelectasis'],
        'Cardiomegaly': ['cardiomegaly']
    }
}
"""
Maps CSV file paths to their respective label mapping dictionaries.
Each CSV uses different label column formats.
Used by: datasets.py - NormalAbnormalCXR14 for label loading
"""

CSVS_TO_IM_PATH_GETTERS: dict = {
    f'{VINDR_FOLDER}/labels_train.csv': (
        lambda row: f"{VINDR_FOLDER}/train/{row['image_id']}.nii.gz", 
        'image_id'
    ),
    f'{VINDR_FOLDER}/labels_test.csv': (
        lambda row: f"{VINDR_FOLDER}/test/{row['image_id']}.nii.gz", 
        'image_id'
    ),
    f'{PADCHEST_FOLDER}/specific_abnormalities_labels.csv': (
        lambda row: f"{PADCHEST_FOLDER}/images/{row['ImageID'].split('.')[0]}.nii.gz", 
        'ImageID'
    )
}
"""
Maps CSV paths to (image_path_function, id_column) tuples.
image_path_function: Takes row, returns full path to NIfTI file
id_column: Column name containing image identifier

Used by: datasets.py - constructing file paths from CSVs
"""

# =============================================================================
# STRING CONSTANTS
# =============================================================================

DOT: str = '.'
"""String constant for dot character. Used in file extension handling."""

EMPTY: str = ''
"""String constant for empty string. Used in string operations."""
