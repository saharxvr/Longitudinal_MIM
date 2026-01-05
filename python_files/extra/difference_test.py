import matplotlib.pyplot as plt
import torch

from models import MaskedReconstructionModel, freeze_and_unfreeze
from constants import *
from preprocessing import BatchPreprocessingImagewise, generate_masked_variations, combine_variations_predictions, dedisorder_batch
import nibabel as nib


if __name__ == '__main__':
    save_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/MIM/Difference_test/expert/C1_new_test.png'
    # save_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/MIM/Difference_test/masks.png'
    # train_dataset = ContrastiveLearningDataset(CXR14_FOLDER + '/AP_images',
    #                                            CXR14_FOLDER + '/train_labels.csv',
    #                                            CUR_LABELS,
    #                                            groups=CUR_LABEL_GROUPS,
    #                                            do_shuffle=True,
    #                                            get_weights=GET_WEIGHTS,
    #                                            label_no_finding=LABEL_NO_FINDING,
    #                                            train=False)
    patch_size = 128

    preprocess = BatchPreprocessingImagewise(use_fourier=False, use_canny=False, clip_limit=2.2, clahe_prob=1., rand_crop_prob=0.0, sigmas=[])
    mim_model = MaskedReconstructionModel(use_mask_token=USE_MASK_TOKEN, dec=5, in_channels=1, use_pos_embed=USE_POS_EMBED, patch_size=patch_size).to(DEVICE)

    # model_path_end = 'Checkpoint_Epoch5_BatchSize_208_NewerLoss_Conv1x1Reduce_Labels_[Consolidation|Infiltration|Mass|Nodule][Edema]_MIM_.pt'
    # model_path = PROJECT_FOLDER + f'saved_models/CL/DETECTION_Checkpoint_Decoder1_Augs_00075_Eff_ViT_Epoch5_Decoder2_L1L2Fourier_GN/{model_path_end}'
    mim_model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/MIM/Checkpoint_id0_MaskGrad_Sig5_Expert_Perc_1Channel_single128_Sched_NoFinding_Decoder5_Augs_Eff_ViT_Epoch30_MaskToken_MS-SSIM_L1_PosEmb_GN.pt'
    mim_checkpoint_dict = torch.load(mim_model_path)
    mim_model.load_state_dict(mim_checkpoint_dict['model_dict'])
    freeze_and_unfreeze([mim_model], [])
    mim_model.eval()

    # im_path = CXR14_FOLDER + '/AP_images/00000061_025.nii.gz'
    # im_path = CXR14_FOLDER + '/AP_images/00000017_001.nii.gz'
    # im_path = PROJECT_FOLDER + '/pneumonia_normal/normal_nibs/IM-0006-0001_jpeg.rf.4f9ab865d41a4b2940ffad02884d26ae.nii.gz'
    # im_path = PROJECT_FOLDER + '/pneumonia_normal/pneumonia_nibs/person1005_virus_1688_jpeg.rf.587c11c5cfd6fa29a49132557a12a446.nii.gz'

    # Mass/Nodule
    # im_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test/665c4a6d2693dc0286d65ab479c9b169.nii.gz'

    # No Finding
    # im_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test/5c55f871429730b84a8ee275839b8aae.nii.gz'

    # Ab VinDr Test
    im_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test/bdd762b3c87ea8261290c6f2825ddcf0.nii.gz'

    im_data = torch.tensor(nib.load(im_path).get_fdata()[None, None, ...])
    im_data, _, cannys = preprocess(im_data)

    # im_mean = torch.mean(im_data, dim=(-1, -2))
    # im_std = torch.std(im_data, dim=(-1, -2))
    # im_data = ((im_data - im_mean) / im_std) * 0.23 + 0.5

    patch_size = 128
    mask_type = None
    mask_vars = generate_masked_variations(im_data, patch_size=patch_size, mask_type=mask_type, canny_im=cannys, mask_token=mim_model.get_mask_token())[0]
    out_cat = []
    for chunk in torch.split(mask_vars, 4):
        out_chunk = mim_model(chunk)
        out_cat.append(out_chunk)
    mask_vars = torch.cat(out_cat, dim=0)

    # d = 512 // patch_size
    # fig, ax = plt.subplots(d, d)
    # for i in range(d):
    #     for j in range(d):
    #         ax[i][j].imshow(mask_vars[d * i + j][0].cpu().numpy().T, cmap='gray')
    # plt.savefig(save_path)

    # fig, ax = plt.subplots(2, 2)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im_data[0][0].cpu().numpy().T, cmap='gray')
    # ax[0,1].imshow(cannys[0][0].cpu().numpy().T, cmap='gray')
    # print(cannys[0][0].cpu().numpy().T)
    combined = combine_variations_predictions(mask_vars, mask_type=mask_type, patch_size=patch_size)
    # combined = ((combined - torch.mean(combined, dim=(-1, -2))) / torch.std(combined, dim=(-1, -2))) * 0.23 + 0.5
    # combined = kornia.enhance.equalize_clahe(combined, 1.8, (8, 8))
    ax[1].imshow(combined[0][0].cpu().numpy().T, cmap='gray')
    dif_im = torch.abs(im_data - combined)
    ax[2].imshow(dif_im[0][0].cpu().numpy().T, cmap='gray')
    # ax[1,0].imshow(mask_vars[0][0].cpu().numpy().T, cmap='gray')
    # ax[1,1].imshow(mask_vars[1][0].cpu().numpy().T, cmap='gray')
    plt.savefig(save_path)
