import numpy as np
import torch

# General
EFF_NET_BLOCK_IDXS = (0, 4, 11, 18, 28, 38, 51, 52)
IMG_SIZE = 512
FEATURE_SIZE = 32
FEATURE_CHANNELS = 640
INTER_CHANNELS = 704
# INTER_CHANNELS = 576
HIDDEN_CHANNELS = 768
# HIDDEN_CHANNELS = 512
PATCHES_IN_SPATIAL_DIM = 16
PATCHES_NUM = PATCHES_IN_SPATIAL_DIM ** 2
PATCH_SIZE = (FEATURE_SIZE ** 2) // PATCHES_NUM
BATCH_NORM_EPS = 1e-5
BATCH_NORM_MOMENTUM = 0.08
INIT_STD = 0.02
GROUP_NORM_GROUPS = 32
WEIGHT_DECAY = 1e-2
MAX_LR = 6e-4
USE_BN = False
INIT_WEIGHTS = True

cuda_dev_count = torch.cuda.device_count()
if not torch.cuda.is_available():
    DEVICE = 'cpu'
elif cuda_dev_count == 1:
    DEVICE = 'cuda'
else:
    DEVICE = f'cuda:{np.argmax([torch.cuda.mem_get_info(k)[0] for k in range(cuda_dev_count)]).item()}'
    torch.cuda.set_device(DEVICE)

print(f'Device = {DEVICE}')

# Masked-Image-Modeling
BATCH_SIZE = 4
UPDATE_EVERY_BATCHES = 12
MASK_PATCH_SIZE = 128
MASK_MODE = 'single'
INIT_MASK_PROB = 0.05
MAX_MASK_PROB = 0.7
END_MASK_PROB = 0.7
MASKED_RECONSTRUCTION_EPOCHS = 30
SIGMAS = []
MASKED_IN_CHANNELS = len(SIGMAS) + 1 if type(SIGMAS) == list else 1
USE_CANNY = type(SIGMAS) == list and len(SIGMAS) > 0
USE_L1 = True
USE_L2 = True
USE_FOURIER = False
USE_PERC_STYLE = False
USE_GAN = False
GAN_START_EPOCH = 13
USE_SSIM = False
USE_MASK_TOKEN = False
USE_POS_EMBED = False
LAMBDA_L1_ALL = 1.
LAMBDA_L1_MASKED = 1.
LAMBDA_L2 = 1.
LAMBDA_FOURIER = 0.05
LAMBDA_FOURIER_MASKED = 0.3
LAMBDA_GAN = 0.5
LAMBDA_P = 2.
LAMBDA_S = 2.
LAMBDA_SSIM = 1.5

# Longitudinal MIM specific
LONGITUDINAL_MIM_EPOCHS = 10
USE_PATCH_DEC = False

# Detection/Contrastive Learning
DETECTION_BATCH_SIZE = 8
DETECTION_UPDATE_EVERY_BATCHES = 5
CONTRASTIVE_BATCH_SIZE = 128
CONTRASTIVE_TRAINING_EPOCHS = 40
DETECTION_START_EPOCH = 1
CONTRASTIVE_START_EPOCH = 38
NORM_START_EPOCH = 55
DETECTION_IN_CHANNELS = 1
TEMPERATURE = 0.1
USE_NEWER_CONTRASTIVE_LOSS = True
OLD_LOSS_C = 0.5
USE_GLOBAL_POOLING = False
GLOBAL_POOLING_FEATURES = 4 * FEATURE_CHANNELS
NON_GLOBAL_POOLING_FEATURES = 2 * (FEATURE_SIZE ** 2)
LATENT_FEATURES = 128
USE_DEDISORDER = False
CONT_LAMBDAS = {'bce': 1., 'cont': 0.01, 'norm': 0.1}

NO_FINDING_PROB_FACTOR = 4.
LABEL_NO_FINDING = False
UNLABELED_PERC = 0.0
# ALL_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
#               'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax']
# ALL_LABELS_NUM = 14
# PADCHEST_ALL_LABELS = ['multiple nodules', 'pseudonodule', 'abnormal foreign body', 'external foreign body']
# CUR_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
#               'Infiltration', 'Mass', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax']
# CUR_LABELS = ['Consolidation', 'Edema', 'Infiltration', 'Mass', 'Nodule']
CUR_LABELS = ['Abnormal', 'Normal']
# CUR_LABELS = ['Localized', 'Interstitial', 'Cardiomegaly']
# CUR_LABELS = ['Pneumothorax', 'Atelectasis', 'Consolidation', "Nodule_Mass", 'ILD', 'Fibrosis']
CUR_LABEL_GROUPS = None
# CUR_LABEL_GROUPS = [[0, 2, 3, 4], [1]]
# GET_WEIGHTS = [2.5, 1., 4.]
# GET_WEIGHTS = [2., 1.5, 2.5, 2., 2., 3.]
GET_WEIGHTS = None
if not CUR_LABEL_GROUPS:
    CUR_LABELS_NUM = len(CUR_LABELS) + 1 if LABEL_NO_FINDING else len(CUR_LABELS)
else:
    CUR_LABELS_NUM = len(CUR_LABEL_GROUPS) + 1 if LABEL_NO_FINDING else len(CUR_LABEL_GROUPS)


# Paths
PROJECT_FOLDER = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/'
MIMIC_FOLDER = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org'
MIMIC_OTHER_AP = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/other_files/'
MIMIC_SUPINE = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/supine_pairs/'
CXR14_FOLDER = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14'
CXR14_CL_TRAIN = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/contrastive_learning/train'
CXR14_CL_VAL = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/contrastive_learning/val'
CXR14_CL_TEST = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/contrastive_learning/test'
PNEUMONIA_FOLDER = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal'
PNEUMONIA_DS_NORMAL = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal/normal_nibs'
PADCHEST_FOLDER = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest'
VINDR_FOLDER = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR'

# LOAD_PATH = PROJECT_FOLDER + f'saved_models/Base/InitializedEncoderBottleneck_{MASKED_IN_CHANNELS}Channels_PretrainedEffViT_16_224_{"BN" if USE_BN else "GN"}.pt'
# LOAD_PATH = ''
LOAD_PATH = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/MIM/Checkpoint_id0_MaskGrad_Sig5_Expert_Perc_1Channel_single128_Sched_NoFinding_Decoder5_Augs_Eff_ViT_Epoch30_MaskToken_MS-SSIM_L1_PosEmb_GN.pt'

DETECTION_LOAD_PATH = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Base/Detection_Dropout_InitializedEncoderBottleneck_{DETECTION_IN_CHANNELS}Channels_PretrainedEffViT_16_224_GN.pt'

LONGITUDINAL_LOAD_PATH = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Base/Longitudinal_InitializedEnc_PretrainedEff_16_224_GN.pt'
# LONGITUDINAL_LOAD_PATH = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id44_Epoch10_Longitudinal_AllEntities_DEVICES_FT_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
# LONGITUDINAL_LOAD_PATH = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id9_Epoch2_Longitudinal_FullImg_SmallNet_DiffEncs_DiffGT_1Channel_single128_Sched_Decoder6_Eff_ViT_MaskToken_L1L2_GN.pt.pt'
# LONGITUDINAL_LOAD_PATH = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id16_FTid9_Epoch3_Longitudinal_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
# LONGITUDINAL_LOAD_PATH = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id35_Epoch10_Longitudinal_MoreFT_MassesRotationInvariance_TrainSet_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'

EFFICIENTNET_B7_PRETRAINED_PATH = "google/efficientnet-b7"
VIT_PRETRAINED_PATH_16_224 = 'google/vit-base-patch16-224'
VIT_PRETRAINED_PATH_32_384 = 'google/vit-base-patch32-384'

# -------------------------------------- #

_VINDR_LABEL_MAPPING = {'Localized': ['Consolidation', 'Nodule/Mass', 'Infiltration'], 'Interstitial': ['Edema', 'ILD', 'Pulmonary fibrosis'],
                        'Pneumothorax': ['Pneumothorax'], 'Atelectasis': ['Atelectasis'], 'Consolidation': ['Consolidation'], 'Cardiomegaly': ['Cardiomegaly'],
                        'Nodule_Mass': ['Nodule/Mass'], 'ILD': ['ILD'], 'Fibrosis': ['Pulmonary fibrosis'], 'Normal': ['No finding']}
CSVS_TO_LABEL_MAPPING = {
    '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_train.csv': _VINDR_LABEL_MAPPING,
    '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_test.csv': _VINDR_LABEL_MAPPING,
    '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/specific_abnormalities_labels.csv':
        {'Localized': ['consolidation', 'pulmonary mass', 'nodule', 'infiltrates'],
         'Interstitial': ['pulmonary edema', 'pulmonary fibrosis', 'reticulonodular interstitial pattern', 'reticular interstitial pattern', 'interstitial pattern'],
         'Pneumothorax': ['pneumothorax'],
         'Atelectasis': ['atelectasis'],
         'Cardiomegaly': ['cardiomegaly']}
}

CSVS_TO_IM_PATH_GETTERS = {
    '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_train.csv': (lambda row: f"{VINDR_FOLDER}/train/{row['image_id']}.nii.gz", 'image_id'),
    '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_test.csv': (lambda row: f"{VINDR_FOLDER}/test/{row['image_id']}.nii.gz", 'image_id'),
    '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/specific_abnormalities_labels.csv': (lambda row: f"{PADCHEST_FOLDER}/images/{row['ImageID'].split('.')[0]}.nii.gz", 'ImageID')
}

# ------------------------------------- #
DOT = '.'
EMPTY = ''

# ------------------------------------- #

