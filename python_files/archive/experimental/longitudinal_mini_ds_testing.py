import os
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import *
from datasets import LongitudinalMIMDataset
from constants import *
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.random as np_rd
from scipy.stats import wasserstein_distance
from scipy.ndimage import label
from augmentations import *
import pandas as pd
from PIL import Image


def compute_emd(im1, im2):
    return wasserstein_distance(im1.flatten().cpu(), im2.flatten().cpu())


def compute_histogram(image_tensor, max_abs_val=255):
    image_np = image_tensor.cpu().numpy()
    hist, _ = np.histogram(image_np, bins=max_abs_val * 2 + 1, range=(-max_abs_val, max_abs_val))
    return hist


def dice_coefficient(y_true: torch.tensor, y_pred: torch.tensor):
    y_true_sum = y_true.sum()
    smooth = max(200. - y_true_sum, 1.)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return ((2. * intersection + smooth) / (y_true_sum + y_pred_f.sum() + smooth)).item()


def remove_small_ccs(im, min_count=16):
    ccs, ccs_num = label(im.cpu().squeeze(), struct)
    ccs = torch.from_numpy(ccs).to(DEVICE)
    unique_vals, unique_counts = torch.unique(ccs, return_counts=True)
    unique_vals = unique_vals[unique_counts > min_count]
    ccs = ccs[None, None, ...]
    ccs_indic = torch.isin(ccs, unique_vals)
    new_im = im * ccs_indic
    new_ccs = ccs * ccs_indic
    ccs_num = torch.numel(unique_vals) - 1
    return new_im, new_ccs, ccs_num


def calculate_detection_measures(gt_ccs: torch.tensor, out_ccs: torch.tensor, gt_ccs_num: int, out_ccs_num: int, low_conf: torch.tensor = None):
    if low_conf is not None:
        orig_gt_labels, orig_gt_labels_counts = torch.unique(gt_ccs, return_counts=True)
        orig_out_labels, orig_out_labels_counts = torch.unique(out_ccs, return_counts=True)

        gt_high_conf = gt_ccs * (~low_conf)
        out_high_conf = out_ccs * (~low_conf)

        high_conf_gt_labels, high_conf_gt_labels_counts = torch.unique(gt_high_conf, return_counts=True)
        high_conf_out_labels, high_conf_out_labels_counts = torch.unique(out_high_conf, return_counts=True)

        low_conf_gt_after_filter = torch.isin(orig_gt_labels, high_conf_gt_labels)
        low_conf_out_after_filter = torch.isin(orig_out_labels, high_conf_out_labels)
        full_low_conf_gt_labels_filter = ~low_conf_gt_after_filter
        full_low_conf_out_labels_filter = ~low_conf_out_after_filter

        full_low_conf_gt_labels = orig_gt_labels[full_low_conf_gt_labels_filter]
        full_low_conf_out_labels = orig_out_labels[full_low_conf_out_labels_filter]

        orig_gt_labels_after_counts = orig_gt_labels_counts[low_conf_gt_after_filter]
        orig_out_labels_after_counts = orig_out_labels_counts[low_conf_out_after_filter]

        gt_labels_after_ratio = high_conf_gt_labels_counts / orig_gt_labels_after_counts
        out_labels_after_ratio = high_conf_out_labels_counts / orig_out_labels_after_counts

        gt_low_ratio_labels = high_conf_gt_labels[gt_labels_after_ratio < 0.3]
        out_low_ratio_labels = high_conf_out_labels[out_labels_after_ratio < 0.05]

        gt_labels_to_remove = torch.cat([full_low_conf_gt_labels, gt_low_ratio_labels])
        out_labels_to_remove = torch.cat([full_low_conf_out_labels, out_low_ratio_labels])

        gt_ccs[torch.isin(gt_ccs, gt_labels_to_remove)] = 0.
        out_ccs[torch.isin(out_ccs, out_labels_to_remove)] = 0.

        gt_ccs_num = len(torch.unique(gt_ccs)) - 1
        out_ccs_num = len(torch.unique(out_ccs)) - 1

    out_intersection = out_ccs * (gt_ccs != 0.)
    # if low_conf is not None:
    #     out_intersection = out_intersection * (~low_conf)
    _, out_intersection_counts = torch.unique(out_intersection, return_counts=True)
    TP_prec = (out_intersection_counts > 5).sum() - 1
    FP = out_ccs_num - TP_prec

    gt_intersection = gt_ccs * (out_ccs != 0.)
    # if low_conf is not None:
    #     gt_intersection = gt_intersection * (~low_conf)
    gt_intersection_labels, gt_intersection_counts = torch.unique(gt_intersection, return_counts=True)
    TP_rec = (gt_intersection_counts > 5).sum() - 1

    # if low_conf is not None:
    #     gt_labels, gt_labels_counts = torch.unique(gt_ccs, return_counts=True)
    #     gt_labels = gt_labels[1:]
    #     gt_labels_counts = gt_labels_counts[1:]
    #     filter_non_intersect = ~torch.isin(gt_labels, gt_intersection_labels)
    #     gt_non_intersect_labels = gt_labels[filter_non_intersect]
    #     gt_non_intersect_labels_counts = gt_labels_counts[filter_non_intersect]
    #
    #     gt_low_conf_intersection = gt_ccs * low_conf
    #     gt_low_conf_labels, gt_low_conf_labels_counts = torch.unique(gt_low_conf_intersection, return_counts=True)
    #     gt_low_conf_labels = gt_low_conf_labels[1:]
    #     gt_low_conf_labels_counts = gt_low_conf_labels_counts[1:]
    #
    #     filter_low_conf_non_intersect = torch.isin(gt_low_conf_labels, gt_non_intersect_labels)
    #     counts_after = gt_low_conf_labels_counts[filter_low_conf_non_intersect]
    #     filter_non_intersect_low_conf = torch.isin(gt_non_intersect_labels, gt_low_conf_labels)
    #     counts_orig = gt_non_intersect_labels_counts[filter_non_intersect_low_conf]
    #
    #     counts_ratios = counts_after / counts_orig
    #     low_conf_undetected_ccs = torch.sum(counts_ratios > 0.5).item()
    #
    #     FN = gt_ccs_num - TP_rec - low_conf_undetected_ccs
    # else:
    FN = gt_ccs_num - TP_rec

    precision = torch.nan_to_num(TP_prec / (TP_prec + FP), nan=1., posinf=1., neginf=1.)
    recall = torch.nan_to_num(TP_rec / (TP_rec + FN), nan=1., posinf=1., neginf=1.)

    if precision and recall:
        # f1_score = torch.nan_to_num((TP_prec + TP_rec) / (TP_prec + TP_rec + FP + FN), nan=1., posinf=1., neginf=1.)
        f1_score = 2 / ((1. / precision) + (1. / recall))
    elif precision:
        f1_score = precision * 0.5
    else:
        f1_score = recall * 0.5
    return precision, recall, f1_score, gt_ccs_num, out_ccs_num, TP_rec, FN


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    np_rd.seed(42)

    plots_folder = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/mini_ds_tests'
    ds_folder = plots_folder + '/ds'

    os.makedirs(plots_folder, exist_ok=True)

    test_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/images',
                                           '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images'])
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1)

    # model = LongitudinalMIMModel(use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=USE_PATCH_DEC, use_pos_embed=USE_POS_EMBED).to(DEVICE)
    model = LongitudinalMIMModelBig(use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=USE_PATCH_DEC, use_pos_embed=USE_POS_EMBED).to(DEVICE)

    model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id16_FTid9_Epoch3_Longitudinal_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
    if 'Checkpoint' in model_path:
        checkpoint_dict = torch.load(model_path)
        print("Loading model weights")
        model.load_state_dict(checkpoint_dict['model_dict'])
    elif model_path:
        # model.enc.load_state_dict(torch.load(LONGITUDINAL_LOAD_PATH))
        model.encoder.load_state_dict(torch.load(model_path))

    model.eval()

    random_perspective_tf = RandomAffineWithMaskTransform()
    random_bspline_tf = RandomBsplineAndSimilarityWithMaskTransform()
    random_geometric_tf = v2.RandomChoice([random_perspective_tf, random_bspline_tf], p=[0.25, 0.75])
    random_intensity_tf = RandomIntensityTransform(clahe_p=0.5, clahe_clip_limit=(0.75, 2.5), blur_p=0., jitter_p=0.5)
    rescale_tf = RescaleValuesTransform()
    mask_crop_tf = CropResizeWithMaskTransform()

    losses_dict = {}
    l1_loss = nn.L1Loss()
    ths = ((0.02, torch.inf), (0.07, torch.inf), (0.12, torch.inf))

    types_list = [
        "",
        "_uncertainty"
    ]

    losses_dict['bl_size_rat'] = []
    losses_dict['fu_size_rat'] = []
    losses_dict['gt_size_rat'] = []
    names = []

    for type_str in types_list:
        losses_dict[f'l1{type_str}'] = []
        losses_dict[f'l1_pos_gt{type_str}'] = []
        losses_dict[f'l1_pos_out{type_str}'] = []
        losses_dict[f'l1_neg_gt{type_str}'] = []
        losses_dict[f'l1_neg_out{type_str}'] = []
        losses_dict[f'l1_no_dif{type_str}'] = []

        # l2_loss = nn.MSELoss()
        # losses_dict['l2'] = []

        losses_dict[f'histogram_emd{type_str}'] = []

        # losses_dict['mislabled'] = [] #TODO


        for i in range(len(ths)):
            losses_dict[f'dice_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'dice_neg{type_str}_th_{i + 1}'] = []
            losses_dict[f'detection_precision_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'detection_precision_neg{type_str}_th_{i + 1}'] = []
            losses_dict[f'detection_recall_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'detection_recall_neg{type_str}_th_{i + 1}'] = []
            losses_dict[f'detection_F1_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'detection_F1_neg{type_str}_th_{i + 1}'] = []

            losses_dict[f'ccs_num_diff_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'ccs_num_diff_neg{type_str}_th_{i + 1}'] = []
            losses_dict[f'gt_ccs_num_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'out_ccs_num_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'gt_ccs_num_neg{type_str}_th_{i + 1}'] = []
            losses_dict[f'out_ccs_num_neg{type_str}_th_{i + 1}'] = []

            losses_dict[f'TP_rec_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'TP_rec_neg{type_str}_th_{i + 1}'] = []
            losses_dict[f'FN_pos{type_str}_th_{i + 1}'] = []
            losses_dict[f'FN_neg{type_str}_th_{i + 1}'] = []

    struct = torch.tensor([[1,1,1], [1,1,1], [1,1,1]])

    torch.cuda.empty_cache()

    # with torch.no_grad():
    #     for n in tqdm(os.listdir(ds_folder)):
    #         print(n)
    #         cur_f = ds_folder + '/' + n
    #         files = sorted(os.listdir(cur_f))
    #         bl_n = cur_f + '/' + files[0]
    #         fu_n = cur_f + '/' + files[1]
    #         gt_n = cur_f + '/' + files[2]
    #         out_n = cur_f + '/' + files[3]
    #
    #         bl = plt.imread(bl_n, format='gray')
    #         print(type(bl))
    #         print(bl.shape)
    #         plt.imshow(bl, cmap='gray')
    #         plt.savefig('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/figtest.png')
    #         exit()

    cases_num = 750
    c_ab = 'all_abs_with_sum_png'

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            torch.cuda.empty_cache()

            bls, fus, gts, fus_mask, name, bls_mask = batch

            c_dir = f'{ds_folder}/{name[0].split(".")[0]}'
            os.makedirs(c_dir, exist_ok=True)
            for n in os.listdir(c_dir):
                if n.startswith('output_') or n.startswith('gt_') or n.startswith('gts_') or n.startswith('outs_'):
                    os.remove(f'{c_dir}/{n}')

            bls = bls.to(DEVICE)
            fus = fus.to(DEVICE)
            gts = gts.to(DEVICE)
            fus_mask = fus_mask.to(DEVICE).bool()
            bls_mask = bls_mask.to(DEVICE).bool()

            # outputs = model(bls, fus)
            outputs, uncertainty = generate_pred_with_uncertainty(bls, bls_mask, fus, fus_mask, model, random_geometric_tf, rescale_tf, random_intensity_tf, mask_crop_tf)
            low_confidence = uncertainty > 1.
            # plt.imshow(gts.cpu().squeeze())
            # plt.savefig(plots_folder + '/gt.png')
            # plt.close()
            # plt.imshow(outputs.cpu().squeeze())
            # plt.savefig(plots_folder + '/mean_outputs.png')
            # plt.close()
            # plt.imshow(uncertainty.cpu().squeeze(), cmap='seismic')
            # plt.savefig(plots_folder + '/uncertainty.png')
            # exit()

            #######################
            # General processing

            gts[torch.logical_or(gts > 0.9, gts < -0.9)] = 0.
            # gts[torch.logical_and(gts < 0.02, gts > -0.02)] = 0.
            gts[torch.logical_and(gts < 0.002, gts > -0.002)] = 0.
            gts = gts * fus_mask

            outputs[torch.logical_or(outputs > 0.9, outputs < -0.9)] = 0.
            # outputs[torch.logical_and(outputs < 0.02, outputs > -0.02)] = 0.
            outputs[torch.logical_and(outputs < 0.002, outputs > -0.002)] = 0.
            outputs[..., :30, -30:] = 0.
            outputs = outputs * fus_mask

            low_confidence = low_confidence * fus_mask

            #######################
            # Calculating parameters and variations

            out_max_val = outputs.max().item()
            out_min_val = outputs.min().item()

            gt_max_val = gts.max().item()
            gt_min_val = gts.min().item()

            rounded_gts = torch.round(gts * 255.).int()
            rounded_outputs = torch.round(outputs * 255.).int()

            pos_gts_bool = gts > 0.
            # pos_gts = gts * pos_gts_bool
            neg_gts_bool = gts < 0.
            # neg_gts = gts * neg_gts_bool
            pos_outputs_bool = outputs > 0.
            # pos_outputs = outputs * pos_outputs_bool
            neg_outputs_bool = outputs < 0.
            # neg_outputs = outputs * neg_outputs_bool

            high_conf_gts = gts * ~low_confidence
            high_conf_outputs = outputs * ~low_confidence

            inputs_list = [
                (gts, outputs, ""),
                (high_conf_gts, high_conf_outputs, "_uncertainty")
            ]
            assert types_list == [t[2] for t in inputs_list]

            #######################
            # Saving raw images and tensors

            plt.imsave(f'{c_dir}/bl.png', (bls * bls_mask).cpu().squeeze(), cmap='gray', format='png')
            plt.imsave(f'{c_dir}/bl.jpeg', (bls * bls_mask).cpu().squeeze(), cmap='gray', format='jpeg')
            plt.imsave(f'{c_dir}/fu.png', (fus * fus_mask).cpu().squeeze(), cmap='gray', format='png')
            plt.imsave(f'{c_dir}/fu.jpeg', (fus * fus_mask).cpu().squeeze(), cmap='gray', format='jpeg')
            plt.imsave(f'{c_dir}/gt.png', gts.cpu().squeeze(), format='png')
            plt.imsave(f'{c_dir}/gt.jpeg', gts.cpu().squeeze(), format='jpeg')
            # plt.imsave(f'{c_dir}/gt_{gt_min_val: .3f}_{gt_max_val: .3f}.png', gts.cpu().squeeze())
            # plt.imsave(f'{c_dir}/output_{out_min_val: .3f}_{out_max_val: .3f}.png', outputs.cpu().squeeze())
            plt.imsave(f'{c_dir}/output.png', outputs.cpu().squeeze(), format='png')
            # plt.imsave(f'{c_dir}/uncertainty.png', uncertainty.cpu().squeeze(), cmap='seismic')

            # torch.save(bls.cpu().squeeze(), f'{c_dir}/bl.pt')
            # torch.save(fus.cpu().squeeze(), f'{c_dir}/fu.pt')
            # torch.save(gts.cpu().squeeze(), f'{c_dir}/gt.pt')
            # torch.save(outputs.cpu().squeeze(), f'{c_dir}/output.pt')
            # torch.save(uncertainty.cpu().squeeze(), f'{c_dir}/uncertainty.pt')

            ######################
            # Calculating measures

            orig_bl_size = os.stat(f'{c_dir}/bl.png').st_size
            comp_bl_size = os.stat(f'{c_dir}/bl.jpeg').st_size
            orig_fu_size = os.stat(f'{c_dir}/fu.png').st_size
            comp_fu_size = os.stat(f'{c_dir}/fu.jpeg').st_size
            orig_gt_size = os.stat(f'{c_dir}/gt.png').st_size
            comp_gt_size = os.stat(f'{c_dir}/gt.jpeg').st_size

            bl_size_rat = torch.sum(bls_mask).item() / orig_bl_size
            fu_size_rat = torch.sum(fus_mask).item() / orig_fu_size
            gt_size_rat = torch.sum(fus_mask).item() / orig_gt_size

            losses_dict['bl_size_rat'].append(bl_size_rat)
            losses_dict['fu_size_rat'].append(fu_size_rat)
            losses_dict['gt_size_rat'].append(gt_size_rat)
            names.append(name)

            for gt, output, type_str in inputs_list:

                #######################
                # l1

                l1 = l1_loss(output, gt)
                losses_dict[f'l1{type_str}'].append(l1.item())
                if torch.any(pos_gts_bool):
                    l1_pos_gt = l1_loss(output[pos_gts_bool], gt[pos_gts_bool])
                    losses_dict[f'l1_pos_gt{type_str}'].append(l1_pos_gt.item())
                if torch.any(neg_gts_bool):
                    l1_neg_gt = l1_loss(output[neg_gts_bool], gt[neg_gts_bool])
                    losses_dict[f'l1_neg_gt{type_str}'].append(l1_neg_gt.item())
                if torch.any(pos_outputs_bool):
                    l1_pos_out = l1_loss(output[pos_outputs_bool], gt[pos_outputs_bool])
                    losses_dict[f'l1_pos_out{type_str}'].append(l1_pos_out.item())
                if torch.any(neg_outputs_bool):
                    l1_neg_out = l1_loss(output[neg_outputs_bool], gt[neg_outputs_bool])
                    losses_dict[f'l1_neg_out{type_str}'].append(l1_neg_out.item())

                if torch.all(gt == 0.):
                    losses_dict[f'l1_no_dif{type_str}'].append(l1.item())

                #######################
                # l2

                # l2 = l2_loss(outputs, gts)
                # losses_dict['l2'].append(l2.item())

                # abs_gt = gts.abs()
                # abs_output = outputs.abs()

                for j, (th_low, th_high) in enumerate(ths):
                    gts_pos_th = torch.logical_and(gt > th_low, gt < th_high)
                    outs_pos_th = torch.logical_and(output > th_low, output < th_high)

                    gts_neg_th = torch.logical_and(gt < -th_low, gt > -th_high)
                    outs_neg_th = torch.logical_and(output < -th_low, output > -th_high)

                    # gts_pos_ccs, gts_pos_ccs_num = label(gts_pos_th.cpu().squeeze(), struct)
                    # gts_pos_ccs = torch.from_numpy(gts_pos_ccs).cuda()
                    # gts_pos_unique_vals, gts_pos_unique_counts = torch.unique(gts_pos_ccs, return_counts=True)
                    # gts_pos_unique_vals = gts_pos_unique_vals[gts_pos_unique_counts > 10]
                    # gts_pos_ccs = gts_pos_ccs[None, None, ...]
                    # gts_pos_th = gts_pos_th * torch.isin(gts_pos_ccs, gts_pos_unique_vals)

                    gts_pos_th, gts_pos_ccs, gts_pos_ccs_num = remove_small_ccs(gts_pos_th, min_count=16)
                    outs_pos_th, outs_pos_ccs, outs_pos_ccs_num = remove_small_ccs(outs_pos_th, min_count=16)
                    gts_neg_th, gts_neg_ccs, gts_neg_ccs_num = remove_small_ccs(gts_neg_th, min_count=16)
                    outs_neg_th, outs_neg_ccs, outs_neg_ccs_num = remove_small_ccs(outs_neg_th, min_count=16)

                    dice_pos = dice_coefficient(gts_pos_th, outs_pos_th)
                    dice_neg = dice_coefficient(gts_neg_th, outs_neg_th)

                    if type_str == "_uncertainty":
                        c_low_conf = low_confidence

                        gts_pos_th = torch.logical_and(gts > th_low, gts < th_high)
                        outs_pos_th = torch.logical_and(outputs > th_low, outputs < th_high)
                        gts_neg_th = torch.logical_and(gts < -th_low, gts > -th_high)
                        outs_neg_th = torch.logical_and(outputs < -th_low, outputs > -th_high)
                        gts_pos_th, gts_pos_ccs, gts_pos_ccs_num = remove_small_ccs(gts_pos_th, min_count=16)
                        outs_pos_th, outs_pos_ccs, outs_pos_ccs_num = remove_small_ccs(outs_pos_th, min_count=16)
                        gts_neg_th, gts_neg_ccs, gts_neg_ccs_num = remove_small_ccs(gts_neg_th, min_count=16)
                        outs_neg_th, outs_neg_ccs, outs_neg_ccs_num = remove_small_ccs(outs_neg_th, min_count=16)
                    else:
                        c_low_conf = None

                    pos_prec, pos_rec, pos_F1, gts_pos_ccs_num, outs_pos_ccs_num, pos_TP_rec, pos_FN = calculate_detection_measures(gts_pos_ccs, outs_pos_ccs, gts_pos_ccs_num, outs_pos_ccs_num, low_conf=c_low_conf)
                    neg_prec, neg_rec, neg_F1, gts_neg_ccs_num, outs_neg_ccs_num, neg_TP_rec, neg_FN = calculate_detection_measures(gts_neg_ccs, outs_neg_ccs, gts_neg_ccs_num, outs_neg_ccs_num, low_conf=c_low_conf)

                    losses_dict[f'dice_pos{type_str}_th_{j + 1}'].append(dice_pos)
                    losses_dict[f'dice_neg{type_str}_th_{j + 1}'].append(dice_neg)

                    losses_dict[f'detection_precision_pos{type_str}_th_{j + 1}'].append(pos_prec)
                    losses_dict[f'detection_recall_pos{type_str}_th_{j + 1}'].append(pos_rec)
                    losses_dict[f'detection_F1_pos{type_str}_th_{j + 1}'].append(pos_F1)
                    losses_dict[f'detection_precision_neg{type_str}_th_{j + 1}'].append(neg_prec)
                    losses_dict[f'detection_recall_neg{type_str}_th_{j + 1}'].append(neg_rec)
                    losses_dict[f'detection_F1_neg{type_str}_th_{j + 1}'].append(neg_F1)

                    losses_dict[f'ccs_num_diff_pos{type_str}_th_{j + 1}'].append(gts_pos_ccs_num - outs_pos_ccs_num)
                    losses_dict[f'ccs_num_diff_neg{type_str}_th_{j + 1}'].append(gts_neg_ccs_num - outs_neg_ccs_num)
                    losses_dict[f'gt_ccs_num_pos{type_str}_th_{j + 1}'].append(gts_pos_ccs_num)
                    losses_dict[f'out_ccs_num_pos{type_str}_th_{j + 1}'].append(outs_pos_ccs_num)
                    losses_dict[f'gt_ccs_num_neg{type_str}_th_{j + 1}'].append(gts_neg_ccs_num)
                    losses_dict[f'out_ccs_num_neg{type_str}_th_{j + 1}'].append(outs_neg_ccs_num)

                    losses_dict[f'TP_rec_pos{type_str}_th_{j + 1}'].append(pos_TP_rec)
                    losses_dict[f'TP_rec_neg{type_str}_th_{j + 1}'].append(neg_TP_rec)
                    losses_dict[f'FN_pos{type_str}_th_{j + 1}'].append(pos_FN)
                    losses_dict[f'FN_neg{type_str}_th_{j + 1}'].append(neg_FN)

                    # plt.imsave(f'{c_dir}/gt_pos{type_str}_th_{j + 1}.png', gts_pos_th.cpu().squeeze())
                    # plt.imsave(f'{c_dir}/out_pos{type_str}_th_{j + 1}.png', outs_pos_th.cpu().squeeze())
                    #
                    # plt.imsave(f'{c_dir}/gt_neg{type_str}_th_{j + 1}.png', gts_neg_th.cpu().squeeze())
                    # plt.imsave(f'{c_dir}/out_neg{type_str}_th_{j + 1}.png', outs_neg_th.cpu().squeeze())

                    to_check = torch.tensor([pos_prec, pos_rec, pos_F1, neg_prec, neg_rec, neg_F1, dice_pos, dice_neg])
                    if torch.any(torch.logical_or(to_check < 0., to_check > 1.)):
                        print("ERROR: Invalid value found")
                        print(f'Name = {name}')

                        print(f'pos, th{j + 1}')
                        print(f"dice={dice_pos}")
                        print(f'ccs num diff, {gts_pos_ccs_num - outs_pos_ccs_num}')
                        print(f'precision={pos_prec: .3f}, recall={pos_rec: .3f}, F1={pos_F1: .3f}')

                        print(f'neg, th{j + 1}')
                        print(f"dice={dice_neg}")
                        print(f'ccs num diff, {gts_neg_ccs_num - outs_neg_ccs_num}')
                        print(f'precision={neg_prec: .3f}, recall={neg_rec: .3f}, F1={neg_F1: .3f}')

                        exit()

                max_abs = max(rounded_gts.abs().max().item(), rounded_outputs.abs().max().item())
                x_bar = [i for i in range(-max_abs, max_abs + 1)]

                emd = compute_emd(rounded_outputs, rounded_gts)
                losses_dict[f'histogram_emd{type_str}'].append(emd)

                # hist1 = compute_histogram(rounded_gts, max_abs)
                # hist2 = compute_histogram(rounded_outputs, max_abs)
                # fig, ax = plt.subplots(2, 1, sharex=True)
                # ax[0].bar(x_bar, hist1)
                # ax[0].set_title("Ground-truth")
                # ax[0].set_yscale('log')
                # # ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                # ax[1].bar(x_bar, hist2, color='orange')
                # ax[1].set_title("Model output")
                # ax[1].set_yscale('log')
                # plt.suptitle(F'EMD = {emd}')
                # plt.tight_layout()
                # plt.savefig(f'{c_dir}/histograms{type_str}.png')
                # ax[0].clear()
                # ax[1].clear()
                # plt.clf()
                # plt.gcf()
                # plt.close()

            # print(name)
            # print(max_abs)
            # print(np.unique(rounded_gts.cpu().numpy(), return_counts=True))
            # print(np.unique(rounded_outputs.cpu().numpy(), return_counts=True))
            # exit()

            if i == cases_num - 1:
                out_name = 'kolmogorov_complexity_stats'

                sorted_names = [(n[0], rat, n_rec.item(), p_rec.item()) for rat, n, n_rec, p_rec in sorted(zip(losses_dict['gt_size_rat'], names, losses_dict[f'detection_recall_neg_th_1'], losses_dict[f'detection_recall_pos_th_1']))]
                print(*sorted_names[:20], sep='\n##### Next ######\n')

                losses_dict = {l: torch.tensor(v, dtype=torch.float) for l, v in losses_dict.items()}

                os.makedirs(f'{plots_folder}/correlation_tests/{c_ab}', exist_ok=True)

                def calc_cor_and_generate_scatter(rat_type, measure):
                    cor = torch.corrcoef(torch.stack([losses_dict[rat_type], losses_dict[measure]]))
                    print(f'Correlation of {rat_type} and {measure} is: {cor[0][1]: .3f}')
                    plt.scatter(losses_dict[rat_type], losses_dict[measure])
                    plt.title(f"{rat_type} to {measure}. Cor = {cor[0][1]: .3f}")
                    plt.xlabel(f'{rat_type}')
                    plt.ylabel(f'{measure}')
                    plt.savefig(f'{plots_folder}/correlation_tests/{c_ab}/{rat_type}_{measure}')
                    plt.clf()
                    plt.close()

                calc_cor_and_generate_scatter('bl_size_rat', "detection_precision_neg_th_1")
                calc_cor_and_generate_scatter('bl_size_rat', "detection_precision_pos_th_1")
                calc_cor_and_generate_scatter('bl_size_rat', "detection_recall_neg_th_1")
                calc_cor_and_generate_scatter('bl_size_rat', "detection_recall_pos_th_1")
                calc_cor_and_generate_scatter('bl_size_rat', "dice_neg_th_1")
                calc_cor_and_generate_scatter('bl_size_rat', "dice_pos_th_1")

                calc_cor_and_generate_scatter('fu_size_rat', "detection_precision_neg_th_1")
                calc_cor_and_generate_scatter('fu_size_rat', "detection_precision_pos_th_1")
                calc_cor_and_generate_scatter('fu_size_rat', "detection_recall_neg_th_1")
                calc_cor_and_generate_scatter('fu_size_rat', "detection_recall_pos_th_1")
                calc_cor_and_generate_scatter('fu_size_rat', "dice_neg_th_1")
                calc_cor_and_generate_scatter('fu_size_rat', "dice_pos_th_1")

                calc_cor_and_generate_scatter('gt_size_rat', "detection_precision_neg_th_1")
                calc_cor_and_generate_scatter('gt_size_rat', "detection_precision_pos_th_1")
                calc_cor_and_generate_scatter('gt_size_rat', "detection_recall_neg_th_1")
                calc_cor_and_generate_scatter('gt_size_rat', "detection_recall_pos_th_1")
                calc_cor_and_generate_scatter('gt_size_rat', "dice_neg_th_1")
                calc_cor_and_generate_scatter('gt_size_rat', "dice_pos_th_1")

                # plt.plot(losses_dict['bl_size_rat'], losses_dict[f'detection_precision_pos_th_1'])
                exit()
                losses_dict = {l: {"Mean": v.mean().item(), "STD": v.std().item()} for l, v in losses_dict.items()}
                # path = f'{plots_folder}/{out_name}.txt'
                # with open(path, 'w') as f:
                #     for k, v in losses_dict.items():
                #         print(k)
                #         print(v)
                #         print()
                #         f.write(f'{k}\nMean: {v["Mean"]: .3f}\nSTD: {v["STD"]: .3f}\n\n')

                df_data = {l: [f'{v["Mean"]: .3f}\n± {v["STD"]:.3f}'] for l, v in losses_dict.items()}
                df = pd.DataFrame(data=df_data)
                df = df.reindex(sorted(df.columns), axis=1)
                df.to_csv(f'{plots_folder}/{out_name}.csv')
                exit()
