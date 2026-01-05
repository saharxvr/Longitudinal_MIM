import os
import nibabel as nib
import torch
from scipy.ndimage import label
from constants import DEVICE, USE_MASK_TOKEN, USE_PATCH_DEC, USE_POS_EMBED
import numpy as np
import numpy.random as np_rd
import random
from datasets import LongitudinalMIMDataset
from models import LongitudinalMIMModelBig
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from tqdm import tqdm
from torchvision.transforms.v2.functional import adjust_sharpness


struct = torch.tensor([[1,1,1], [1,1,1], [1,1,1]])


def generate_alpha_map(x: torch.Tensor):
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.03)
    alphas_map = x_abs / max_val

    return alphas_map


def plot_diff_on_current(diff_map, current, out_p):
    diff_map = diff_map.squeeze()
    current = current.squeeze()

    alphas = generate_alpha_map(diff_map)
    divnorm = colors.TwoSlopeNorm(vmin=min(torch.min(diff_map).item(), -0.01), vcenter=0., vmax=max(torch.max(diff_map).item(), 0.01))
    fig, ax = plt.subplots()
    ax.imshow(current.squeeze().cpu(), cmap='gray')
    imm1 = ax.imshow(diff_map.squeeze().cpu(), alpha=alphas.cpu(), cmap=differential_grad, norm=divnorm)
    cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()

    plt.savefig(out_p)

    # ax[0].clear()
    plt.cla()
    plt.clf()
    plt.close()


def dice_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor, y_true_ccs: torch.Tensor, y_pred_ccs: torch.Tensor):
    y_true_sum = y_true.sum()
    y_pred_sum = y_pred.sum()
    # smooth = max(200. - y_true_sum, 1.)
    smooth = 10
    intersection = (y_true * y_pred).sum()
    dice = ((2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)).item()
    # return dice
    
    # Dice with/out FP & FN
    
    pred_intersection = y_pred_ccs * (y_true_ccs != 0.)
    true_intersection = y_true_ccs * (y_pred_ccs != 0.)

    pred_intersection_vals, pred_intersection_counts = torch.unique(pred_intersection, return_counts=True)
    true_intersection_vals, true_intersection_counts = torch.unique(true_intersection, return_counts=True)

    pred_intersection_vals = pred_intersection_vals[1:]
    true_intersection_vals = true_intersection_vals[1:]
    pred_intersection_counts = pred_intersection_counts[1:]
    true_intersection_counts = true_intersection_counts[1:]

    pred_intersection_vals = pred_intersection_vals[pred_intersection_counts > 10]
    true_intersection_vals = true_intersection_vals[true_intersection_counts > 10]

    y_pred_no_FP = torch.isin(y_pred_ccs, pred_intersection_vals)
    y_true_no_FN = torch.isin(y_true_ccs, true_intersection_vals)
    
    y_pred_no_FP_sum = y_pred_no_FP.sum()
    y_true_no_FN_sum = y_true_no_FN.sum()
    
    intersection_no_FP = (y_pred_no_FP * y_true).sum()
    intersection_no_FN = (y_pred * y_true_no_FN).sum()
    intersection_no_FP_no_FN = (y_pred_no_FP * y_true_no_FN).sum()
    
    dice_no_FP = ((2. * intersection_no_FP + smooth) / (y_true_sum + y_pred_no_FP_sum + smooth)).item()
    dice_no_FN = ((2. * intersection_no_FN + smooth) / (y_true_no_FN_sum + y_pred_sum + smooth)).item()
    dice_no_FP_no_FN = ((2. * intersection_no_FP_no_FN + smooth) / (y_true_no_FN_sum + y_pred_no_FP_sum + smooth)).item()
    
    dices_per_mass = []
    for val in true_intersection_vals:
        true_cc = y_true_ccs == val
        true_cc_area = torch.sum(true_cc)
        
        cc_inter = y_pred_ccs * true_cc
        pred_vals = torch.unique(cc_inter, return_counts=False)[1:]
        pred_cc = torch.isin(y_pred_ccs, pred_vals)
        pred_cc_area = torch.sum(pred_cc)
        
        inter_cc_area = torch.sum(cc_inter != 0)
        
        cc_dice = ((2. * inter_cc_area) / (true_cc_area + pred_cc_area)).item()
        
        dices_per_mass.append(cc_dice)

    return dice, dice_no_FP, dice_no_FN, dice_no_FP_no_FN, dices_per_mass


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


def calculate_detection_measures(gt_ccs: torch.tensor, out_ccs: torch.tensor, gt_ccs_num: int, out_ccs_num: int):
    out_intersection = out_ccs * (gt_ccs != 0.)

    out_intersection_labels, out_intersection_counts = torch.unique(out_intersection, return_counts=True)
    TP_prec = (out_intersection_counts > 5).sum() - 1
    FP = out_ccs_num - TP_prec

    gt_intersection = gt_ccs * (out_ccs != 0.)

    gt_intersection_labels, gt_intersection_counts = torch.unique(gt_intersection, return_counts=True)
    TP_rec = (gt_intersection_counts > 5).sum() - 1

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

    hit_labels = out_intersection_labels[out_intersection_counts > 5]
    out_FP = (~torch.isin(out_ccs, hit_labels)).float() * 0.01

    return precision, recall, f1_score, TP_rec, FN, TP_prec, FP, out_FP


def create_violin_plot(pos_arrs, neg_arrs, title, y_label, save_path):
    ticks = [m + 1 for m in range(len(pos_arrs))]
    maxes, mins = [], []

    pos_parts = plt.violinplot(pos_arrs, showmeans=True, showextrema=True, showmedians=False, side='low')
    neg_parts = plt.violinplot(neg_arrs, showmeans=True, showextrema=True, showmedians=False, side='high')

    print(f'Creating violin plot for {title} for {save_path}')

    for parts, arrs, col, std_offset in [(pos_parts, pos_arrs, 'red', -0.12), (neg_parts, neg_arrs, 'green', 0.12)]:
        for k, v in parts.items():
            if k == 'bodies':
                for i, pc in enumerate(v):
                    pc.set_facecolor(col)
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.4)

                    c_mean = np.mean(arrs[i])
                    c_std = np.std(arrs[i])
                    print(f'Cur mean: {c_mean}')
                    print(f'Cur std: {c_std}')
                    print()
                    maxes.append(c_mean + c_std + 0.05)
                    mins.append(c_mean - c_std - 0.05)
                    plt.hlines(c_mean - c_std, i + 1 + std_offset, i + 1, color='blue', linewidth=1.6, linestyles='dashed')
                    plt.hlines(c_mean + c_std, i + 1 + std_offset, i + 1, color='blue', linewidth=1.6, linestyles='dashed')
            else:
                v.set_color('black')

    max_val = max(max(maxes), 1.)
    min_val = min(min(mins), 0.)
    plt.ylim(min_val, max_val)

    plt.title(title)
    plt.xticks(ticks=ticks, labels=['None\n(no mass change)', 'None', 'Low', 'Normal', 'High'], rotation=12, fontsize=10)
    plt.xlabel('Orientation changes', fontsize=11)
    plt.ylabel(y_label, fontsize=11)

    legend_patches = [
        Patch(facecolor='red', edgecolor='black', label='Positive'),
        Patch(facecolor='green', edgecolor='black', label='Negative')
    ]
    plt.legend(handles=legend_patches, loc='best')

    pos_means = [sum(a) / len(a) for a in pos_arrs]
    neg_means = [sum(a) / len(a) for a in neg_arrs]
    plt.plot(ticks, pos_means, linestyle=(2.5, (5, 10)), linewidth=1.2, alpha=0.85, color='red')
    plt.plot(ticks, neg_means, linestyle=(7.5, (5, 10)), linewidth=1.2, alpha=0.85, color='green')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    with torch.no_grad():
        differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
            # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
            (0.000, (0.235, 1.000, 0.239)),
            (0.400, (0.000, 1.000, 0.702)),
            (0.500, (1.000, 0.988, 0.988)),
            (0.600, (1.000, 0.604, 0.000)),
            (1.000, (0.682, 0.000, 0.000))))

        torch.manual_seed(42)
        random.seed(42)
        np_rd.seed(42)

        # plots_folder = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/CT_scans/tests/new'
        # DRR_pairs_folder = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_test/new'
        # model_path = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id32_Epoch8_Longitudinal_MassesRotationInvariance_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        plots_folder = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/results/sharpen_ft_model'
        DRR_pairs_folder = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test'
        model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id36_Epoch6_Longitudinal_MoreFT_DiverseTrainSet_MassesRotationInvariance_TrainSet_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id34_Epoch8_Longitudinal_MassesRotationInvariance_TrainSet_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'

        model = LongitudinalMIMModelBig(use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=USE_PATCH_DEC, use_pos_embed=USE_POS_EMBED, use_technical_bottleneck=False).to(DEVICE)
        checkpoint_dict = torch.load(model_path)
        model.load_state_dict(checkpoint_dict['model_dict'], strict=True)
        model.eval()

        # rot_angles_lst = [(0, 0, 0), (3, 6, 6), (6, 12, 12), (10, 20, 20), (12, 24, 24), (15, 30, 30), (18, 36, 36), (22, 44, 44)]
        # rot_angles_lst = [(0, 0, 0), (0, 0, 0), (5, 10, 10), (15, 25, 25), (25, 40, 40)]
        rot_angles_lst = [(0, 0, 0), (10, 10, 10), (20, 20, 20)]

        sharpen = True

        avg_dices_pos, avg_precs_pos, avg_recs_pos = [], [], []
        avg_dices_neg, avg_precs_neg, avg_recs_neg = [], [], []

        avg_dices_no_FP_pos, avg_dices_no_FN_pos, avg_dices_no_FP_no_FN_pos = [], [], []
        avg_dices_no_FP_neg, avg_dices_no_FN_neg, avg_dices_no_FP_no_FN_neg = [], [], []
        
        avg_gt_ccs_num_pos, avg_out_ccs_num_pos, avg_TP_prec_pos, avg_FP_pos = [], [], [], []
        avg_gt_ccs_num_neg, avg_out_ccs_num_neg, avg_TP_prec_neg, avg_FP_neg = [], [], [], []

        avg_areas_pos = []
        avg_areas_neg = []
        
        all_dices_pos, all_precs_pos, all_recs_pos = [], [], []
        all_dices_neg, all_precs_neg, all_recs_neg = [], [], []

        all_dices_no_FP_pos, all_dices_no_FN_pos, all_dices_no_FP_no_FN_pos = [], [], []
        all_dices_no_FP_neg, all_dices_no_FN_neg, all_dices_no_FP_no_FN_neg = [], [], []

        all_dices_per_mass_pos, avg_precs_per_mass_pos, avg_recs_per_mass_pos = [], [], []
        all_dices_per_mass_neg, avg_precs_per_mass_neg, avg_recs_per_mass_neg = [], [], []

        pair_names_arrs = [[] for _ in range(len(rot_angles_lst))]
        str_rot_angles_lst = [str(l) for l in rot_angles_lst]

        for idx, rot_angles in enumerate(rot_angles_lst):
            # if idx == 0:
            #     continue
            # print(f'Working on rot angles {rot_angles}{" no masses" if idx == 0 else ""}')
            print(f'Working on rot angles {rot_angles}')

            avg_dices_no_FP_no_FN_pos.append([])
            avg_dices_no_FP_no_FN_neg.append([])
            avg_precs_pos.append([])
            avg_precs_neg.append([])
            avg_recs_pos.append([])
            avg_recs_neg.append([])

            all_types = os.listdir(DRR_pairs_folder)
            for c_type in all_types:
                print(f'Working on type {c_type}')

                c_pairs_path = f'{DRR_pairs_folder}/{c_type}'
                c_plots_path = f'{plots_folder}/{c_type}'

                # if idx < 2:
                #     dataset = LongitudinalMIMDataset(entity_dirs=[], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_test/angles_0_0_0'])
                # else:
                #     dataset = LongitudinalMIMDataset(entity_dirs=[], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[f'{DRR_pairs_folder}/angles_{rot_angles[0]}_{rot_angles[1]}_{rot_angles[2]}'])

                # dataset = LongitudinalMIMDataset(entity_dirs=[], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[f'{DRR_pairs_folder}/angles_{rot_angles[0]}_{rot_angles[1]}_{rot_angles[2]}'])
                dataset = LongitudinalMIMDataset(entity_dirs=[], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[f'{c_pairs_path}/angles_{rot_angles[0]}_{rot_angles[1]}_{rot_angles[2]}'])

                # if idx == 0:
                #     c_plots_folder = f'{plots_folder}/masses_rotation_invariance_angles_0_0_0_no_masses'
                # else:
                #     c_plots_folder = f'{plots_folder}/masses_rotation_invariance_angles_{rot_angles[0]}_{rot_angles[1]}_{rot_angles[2]}'
                c_plots_folder = f'{c_plots_path}/masses_rotation_invariance_angles_{rot_angles[0]}_{rot_angles[1]}_{rot_angles[2]}'

                ds_len = len(dataset)
                j = 0

                cur_dices_pos, cur_precs_pos, cur_recs_pos = [], [], []
                cur_dices_neg, cur_precs_neg, cur_recs_neg = [], [], []

                cur_dices_no_FP_pos, cur_dices_no_FN_pos, cur_dices_no_FP_no_FN_pos = [], [], []
                cur_dices_no_FP_neg, cur_dices_no_FN_neg, cur_dices_no_FP_no_FN_neg = [], [], []

                cur_gt_ccs_num_pos, cur_out_ccs_num_pos, cur_TP_prec_pos, cur_FP_pos = [], [], [], []
                cur_gt_ccs_num_neg, cur_out_ccs_num_neg, cur_TP_prec_neg, cur_FP_neg = [], [], [], []

                cur_areas_pos = []
                cur_areas_neg = []

                cur_dices_per_mass_pos, cur_precs_per_mass_pos, cur_recs_per_mass_pos = [], [], []
                cur_dices_per_mass_neg, cur_precs_per_mass_neg, cur_recs_per_mass_neg = [], [], []

                for i in tqdm(range(0, ds_len, 1)):
                    bl, fu, gt, _ = dataset[i]

                    # if idx == 0:
                    #     flag = random.random() > 0.5
                    #     if flag:
                    #         bl = fu.clone()
                    #     else:
                    #         fu = bl.clone()
                    #     gt = torch.zeros_like(fu)

                    if sharpen:
                        bl = adjust_sharpness(bl, sharpness_factor=8.)
                        fu = adjust_sharpness(fu, sharpness_factor=8.)

                    output = model(bl.to(DEVICE).float().unsqueeze(0), fu.to(DEVICE).float().unsqueeze(0)).squeeze()
                    gt = gt.to(DEVICE).squeeze()

                    output[output.abs() > 0.6] = 0
                    gt[gt.abs() > 0.6] = 0

                    output[:, :15] = 0.
                    output[:, -15:] = 0.
                    output[:15, :] = 0.
                    output[-15:, :] = 0.

                    if rot_angles[0] == 0:
                        outs_pos_th = output > 0.05
                        gts_pos_th = gt > 0.02
                    else:
                        outs_pos_th = output > 0.03
                        gts_pos_th = gt > 0.02

                    if rot_angles[0] == 0:
                        outs_neg_th = output < -0.05
                        gts_neg_th = gt < -0.02
                    else:
                        outs_neg_th = output < -0.03
                        gts_neg_th = gt < -0.02

                    outs_pos_th, _, __ = remove_small_ccs(outs_pos_th, min_count=30)
                    outs_neg_th, _, __ = remove_small_ccs(outs_neg_th, min_count=30)

                    outs_pos_th = (-torch.nn.functional.max_pool2d(-(outs_pos_th.squeeze().float()[None, None, ...]), kernel_size=9, stride=1, padding=4))
                    outs_pos_th = torch.nn.functional.max_pool2d(outs_pos_th, kernel_size=19, stride=1, padding=9)
                    outs_pos_th = (-torch.nn.functional.max_pool2d(-outs_pos_th, kernel_size=9, stride=1, padding=4)).squeeze().bool()

                    # gts_pos_th = torch.nn.functional.max_pool2d(gts_pos_th, kernel_size=5, stride=1, padding=2)
                    # gts_pos_th = -torch.nn.functional.max_pool2d(-gts_pos_th, kernel_size=5, stride=1, padding=2)

                    outs_neg_th = (-torch.nn.functional.max_pool2d(-(outs_neg_th.squeeze().float()[None, None, ...]), kernel_size=9, stride=1, padding=4))
                    outs_neg_th = torch.nn.functional.max_pool2d(outs_neg_th, kernel_size=19, stride=1, padding=9)
                    outs_neg_th = (-torch.nn.functional.max_pool2d(-outs_neg_th, kernel_size=9, stride=1, padding=4)).squeeze().bool()

                    # gts_neg_th = torch.nn.functional.max_pool2d(gts_neg_th, kernel_size=5, stride=1, padding=2)
                    # gts_neg_th = -torch.nn.functional.max_pool2d(-gts_neg_th, kernel_size=5, stride=1, padding=2)

                    gts_pos_th, gts_pos_ccs, gts_pos_ccs_num = remove_small_ccs(gts_pos_th, min_count=180)
                    outs_pos_th, outs_pos_ccs, outs_pos_ccs_num = remove_small_ccs(outs_pos_th, min_count=180)
                    gts_neg_th, gts_neg_ccs, gts_neg_ccs_num = remove_small_ccs(gts_neg_th, min_count=180)
                    outs_neg_th, outs_neg_ccs, outs_neg_ccs_num = remove_small_ccs(outs_neg_th, min_count=180)

                    dice_pos, dice_no_FP_pos, dice_no_FN_pos, dice_no_FP_no_FN_pos, dices_per_mass_pos = dice_coefficient(gts_pos_th, outs_pos_th, gts_pos_ccs, outs_pos_ccs)
                    dice_neg, dice_no_FP_neg, dice_no_FN_neg, dice_no_FP_no_FN_neg, dices_per_mass_neg = dice_coefficient(gts_neg_th, outs_neg_th, gts_neg_ccs, outs_neg_ccs)

                    pos_prec, pos_rec, pos_F1, pos_TP_rec, pos_FN, pos_TP_prec, pos_FP, out_FP_pos = calculate_detection_measures(gts_pos_ccs, outs_pos_ccs, gts_pos_ccs_num, outs_pos_ccs_num)
                    neg_prec, neg_rec, neg_F1, neg_TP_rec, neg_FN, neg_TP_prec, neg_FP, out_FP_neg = calculate_detection_measures(gts_neg_ccs, outs_neg_ccs, gts_neg_ccs_num, outs_neg_ccs_num)

                    # print(dice_pos)
                    # print(dice_neg)
                    # print(pos_prec)
                    # print(neg_prec)
                    # print(pos_rec)
                    # print(neg_rec)

                    cur_dices_pos.append(dice_pos)
                    cur_dices_neg.append(dice_neg)

                    cur_precs_pos.append(pos_prec.item())
                    cur_precs_neg.append(neg_prec.item())

                    cur_recs_pos.append(pos_rec.item())
                    cur_recs_neg.append(neg_rec.item())

                    cur_dices_no_FP_pos.append(dice_no_FP_pos)
                    cur_dices_no_FN_pos.append(dice_no_FN_pos)
                    cur_dices_no_FP_no_FN_pos.append(dice_no_FP_no_FN_pos)

                    cur_dices_no_FP_neg.append(dice_no_FP_neg)
                    cur_dices_no_FN_neg.append(dice_no_FN_neg)
                    cur_dices_no_FP_no_FN_neg.append(dice_no_FP_no_FN_neg)

                    cur_gt_ccs_num_pos.append(gts_pos_ccs_num)
                    cur_out_ccs_num_pos.append(outs_pos_ccs_num)
                    cur_TP_prec_pos.append(pos_TP_prec)
                    cur_FP_pos.append(pos_FP)

                    cur_gt_ccs_num_neg.append(gts_neg_ccs_num)
                    cur_out_ccs_num_neg.append(outs_neg_ccs_num)
                    cur_TP_prec_neg.append(neg_TP_prec)
                    cur_FP_neg.append(neg_FP)

                    cur_area_pos = torch.sum(outs_pos_th).item()
                    cur_area_neg = torch.sum(outs_neg_th).item()

                    cur_areas_pos.append(cur_area_pos)
                    cur_areas_neg.append(cur_area_neg)

                    pos_prec_item = pos_prec.item()
                    pos_rec_item = pos_rec.item()
                    neg_prec_item = neg_prec.item()
                    neg_rec_item = neg_rec.item()
                    cur_precs_per_mass_pos.extend([pos_prec_item for _ in range(outs_pos_ccs_num)])
                    cur_recs_per_mass_pos.extend([pos_rec_item for _ in range(gts_pos_ccs_num)])
                    cur_precs_per_mass_neg.extend([neg_prec_item for _ in range(outs_neg_ccs_num)])
                    cur_recs_per_mass_neg.extend([neg_rec_item for _ in range(gts_neg_ccs_num)])

                    cur_dices_per_mass_pos.extend(dices_per_mass_pos)
                    cur_dices_per_mass_neg.extend(dices_per_mass_neg)

                    pair_names_arrs[idx].append(f'pair{j}')

                    # Plot results:

                    out_dir = f'{c_plots_folder}/pair{j}'
                    os.makedirs(out_dir, exist_ok=True)
                    j += 1

                    plt.imshow(bl.cpu().squeeze(), cmap='gray')
                    plt.savefig(f'{out_dir}/bl.png')
                    plt.clf()
                    plt.close()

                    plt.imshow(fu.cpu().squeeze(), cmap='gray')
                    plt.savefig(f'{out_dir}/fu.png')
                    plt.clf()
                    plt.close()

                    plot_diff_on_current(gt, fu, out_p=f'{out_dir}/gt.png')
                    plot_diff_on_current(output, fu, out_p=f'{out_dir}/out.png')
                    plot_diff_on_current(out_FP_pos, fu, out_p=f'{out_dir}/FPs_pos.png')
                    plot_diff_on_current(out_FP_neg, fu, out_p=f'{out_dir}/FPs_neg.png')

                    pos_map_bin_out = outs_pos_th.float()
                    neg_map_bin_out = -outs_neg_th.float()
                    plot_diff_on_current(pos_map_bin_out + neg_map_bin_out, fu, out_p=f'{out_dir}/out_binary.png')

                    pos_map_bin_gt = gts_pos_th.float()
                    neg_map_bin_gt = -gts_neg_th.float()
                    plot_diff_on_current(pos_map_bin_gt + neg_map_bin_gt, fu, out_p=f'{out_dir}/gt_binary.png')

                # all_dices_pos.append(cur_dices_pos)
                # all_dices_neg.append(cur_dices_neg)
                #
                # all_precs_pos.append(cur_precs_pos)
                # all_precs_neg.append(cur_precs_neg)
                #
                # all_recs_pos.append(cur_recs_pos)
                # all_recs_neg.append(cur_recs_neg)
                #
                # all_dices_no_FN_pos.append(cur_dices_no_FN_pos)
                # all_dices_no_FN_neg.append(cur_dices_no_FN_neg)
                #
                # all_dices_no_FP_pos.append(cur_dices_no_FP_pos)
                # all_dices_no_FP_neg.append(cur_dices_no_FP_neg)
                #
                # all_dices_no_FP_no_FN_pos.append(cur_dices_no_FP_no_FN_pos)
                # all_dices_no_FP_no_FN_neg.append(cur_dices_no_FP_no_FN_neg)

                # avg_dice_pos = sum(cur_dices_pos) / len(cur_dices_pos)
                avg_prec_pos = sum(cur_precs_pos) / len(cur_precs_pos)
                avg_rec_pos = sum(cur_recs_pos) / len(cur_recs_pos)
                # avg_dice_neg = sum(cur_dices_neg) / len(cur_dices_neg)
                avg_prec_neg = sum(cur_precs_neg) / len(cur_precs_neg)
                avg_rec_neg = sum(cur_recs_neg) / len(cur_recs_neg)

                # avg_dices_pos.append(avg_dice_pos)
                avg_precs_pos[-1].append(avg_prec_pos)
                avg_recs_pos[-1].append(avg_rec_pos)

                # avg_dices_neg.append(avg_dice_neg)
                avg_precs_neg[-1].append(avg_prec_neg)
                avg_recs_neg[-1].append(avg_rec_neg)

                # print(f'avg_dice_pos:\n{avg_dice_pos}\n')
                print(f'avg_prec_pos:\n{avg_prec_pos}\n')
                print(f'avg_rec_pos:\n{avg_rec_pos}\n')

                # print(f'avg_dice_neg:\n{avg_dice_neg}\n')
                print(f'avg_prec_neg:\n{avg_prec_neg}\n')
                print(f'avg_rec_neg:\n{avg_rec_neg}\n')

                print("###############################")

                # avg_dice_no_FP_pos = sum(cur_dices_no_FP_pos) / len(cur_dices_no_FP_pos)
                # avg_dice_no_FN_pos = sum(cur_dices_no_FN_pos) / len(cur_dices_no_FN_pos)
                avg_dice_no_FP_no_FN_pos = sum(cur_dices_no_FP_no_FN_pos) / len(cur_dices_no_FP_no_FN_pos)
                # avg_dice_no_FP_neg = sum(cur_dices_no_FP_neg) / len(cur_dices_no_FP_neg)
                # avg_dice_no_FN_neg = sum(cur_dices_no_FN_neg) / len(cur_dices_no_FN_neg)
                avg_dice_no_FP_no_FN_neg = sum(cur_dices_no_FP_no_FN_neg) / len(cur_dices_no_FP_no_FN_neg)

                # avg_dices_no_FP_pos.append(avg_dice_no_FP_pos)
                # avg_dices_no_FN_pos.append(avg_dice_no_FN_pos)
                avg_dices_no_FP_no_FN_pos[-1].append(avg_dice_no_FP_no_FN_pos)

                # avg_dices_no_FP_neg.append(avg_dice_no_FP_neg)
                # avg_dices_no_FN_neg.append(avg_dice_no_FN_neg)
                avg_dices_no_FP_no_FN_neg[-1].append(avg_dice_no_FP_no_FN_neg)

                # print(f'avg_dice_no_FP_pos:\n{avg_dice_no_FP_pos}\n')
                # print(f'avg_dice_no_FN_pos:\n{avg_dice_no_FN_pos}\n')
                print(f'avg_dice_no_FP_no_FN_pos:\n{avg_dice_no_FP_no_FN_pos}\n')

                # print(f'avg_dice_no_FP_neg:\n{avg_dice_no_FP_neg}\n')
                # print(f'avg_dice_no_FN_neg:\n{avg_dice_no_FN_neg}\n')
                print(f'avg_dice_no_FP_no_FN_neg:\n{avg_dice_no_FP_no_FN_neg}\n')

                # print("###############################")
                #
                # c_avg_gt_ccs_num_pos = sum(cur_gt_ccs_num_pos) / len(cur_gt_ccs_num_pos)
                # c_avg_out_ccs_num_pos = sum(cur_out_ccs_num_pos) / len(cur_out_ccs_num_pos)
                # c_avg_TP_prec_pos = sum(cur_TP_prec_pos) / len(cur_TP_prec_pos)
                # c_avg_FP_pos = sum(cur_FP_pos) / len(cur_FP_pos)
                #
                # c_avg_gt_ccs_num_neg = sum(cur_gt_ccs_num_neg) / len(cur_gt_ccs_num_neg)
                # c_avg_out_ccs_num_neg = sum(cur_out_ccs_num_neg) / len(cur_out_ccs_num_neg)
                # c_avg_TP_prec_neg = sum(cur_TP_prec_neg) / len(cur_TP_prec_neg)
                # c_avg_FP_neg = sum(cur_FP_neg) / len(cur_FP_neg)
                #
                # avg_gt_ccs_num_pos.append(c_avg_gt_ccs_num_pos)
                # avg_out_ccs_num_pos.append(c_avg_out_ccs_num_pos)
                # avg_TP_prec_pos.append(c_avg_TP_prec_pos)
                # avg_FP_pos.append(c_avg_FP_pos)
                #
                # avg_gt_ccs_num_neg.append(c_avg_gt_ccs_num_neg)
                # avg_out_ccs_num_neg.append(c_avg_out_ccs_num_neg)
                # avg_TP_prec_neg.append(c_avg_TP_prec_neg)
                # avg_FP_neg.append(c_avg_FP_neg)
                #
                # print(f'c_avg_gt_ccs_num_pos:\n{c_avg_gt_ccs_num_pos}')
                # print(f'c_avg_out_ccs_num_pos:\n{c_avg_out_ccs_num_pos}')
                # print(f'c_avg_TP_prec_pos:\n{c_avg_TP_prec_pos}')
                # print(f'c_avg_FP_pos:\n{c_avg_FP_pos}')
                #
                # print(f'c_avg_gt_ccs_num_neg:\n{c_avg_gt_ccs_num_neg}')
                # print(f'c_avg_out_ccs_num_neg:\n{c_avg_out_ccs_num_neg}')
                # print(f'c_avg_TP_prec_neg:\n{c_avg_TP_prec_neg}')
                # print(f'c_avg_FP_neg:\n{c_avg_FP_neg}')
                #
                # avg_area_pos = sum(cur_areas_pos) / len(cur_areas_pos)
                # avg_area_neg = sum(cur_areas_neg) / len(cur_areas_neg)
                #
                # avg_areas_pos.append(avg_area_pos)
                # avg_areas_neg.append(avg_area_neg)
                #
                # print(f'avg_area_pos:\n{avg_area_pos}')
                # print(f'avg_area_neg:\n{avg_area_neg}')

                # if idx != 0:
                #     avg_precs_per_mass_pos.append(sum(cur_precs_per_mass_pos) / len(cur_precs_per_mass_pos))
                #     avg_recs_per_mass_pos.append(sum(cur_recs_per_mass_pos) / len(cur_recs_per_mass_pos))
                #     avg_precs_per_mass_neg.append(sum(cur_precs_per_mass_neg) / len(cur_precs_per_mass_neg))
                #     avg_recs_per_mass_neg.append(sum(cur_recs_per_mass_neg) / len(cur_recs_per_mass_neg))
                #
                #     all_dices_per_mass_pos.append(cur_dices_per_mass_pos)
                #     all_dices_per_mass_neg.append(cur_dices_per_mass_neg)
                # else:
                #     avg_precs_per_mass_pos.append(1.)
                #     avg_recs_per_mass_pos.append(1.)
                #     avg_precs_per_mass_neg.append(1.)
                #     avg_recs_per_mass_neg.append(1.)
                #
                #     all_dices_per_mass_pos.append([1.])
                #     all_dices_per_mass_neg.append([1.])

        print('avg_rec_pos:')
        print(avg_rec_pos)

        avg_recs_pos = np.array(avg_recs_pos).T
        avg_precs_pos = np.array(avg_precs_pos).T
        avg_dices_no_FP_no_FN_pos = np.array(avg_dices_no_FP_no_FN_pos).T
        avg_recs_neg = np.array(avg_recs_neg).T
        avg_precs_neg = np.array(avg_precs_neg).T
        avg_dices_no_FP_no_FN_neg = np.array(avg_dices_no_FP_no_FN_neg).T

        for arr, t in zip(avg_recs_pos, all_types):
            plt.plot(str_rot_angles_lst, arr, label=t, marker='o')
            print(f'Recall (pos) for type {t}:\n{arr}')
        plt.ylim(0., 1.)
        plt.title('Manufacturers - Recall (pos)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles')
        plt.ylabel('Recall')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_folder}/aaa_recall_pos.png')
        plt.clf()
        plt.close()
        
        for arr, t in zip(avg_precs_pos, all_types):
            plt.plot(str_rot_angles_lst, arr, label=t, marker='o')
            print(f'Precision (pos) for type {t}:\n{arr}')
        plt.ylim(0., 1.)
        plt.title('Manufacturers - Precision (pos)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles')
        plt.ylabel('Precision')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_folder}/aaa_precision_pos.png')
        plt.clf()
        plt.close()
        
        for arr, t in zip(avg_dices_no_FP_no_FN_pos, all_types):
            plt.plot(str_rot_angles_lst, arr, label=t, marker='o')
            print(f'Dice (pos) for type {t}:\n{arr}')
        plt.ylim(0., 1.)
        plt.title('Manufacturers - Dice (pos)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles')
        plt.ylabel('Dice (TPs)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_folder}/aaa_dice_pos.png')
        plt.clf()
        plt.close()

        for arr, t in zip(avg_recs_neg, all_types):
            plt.plot(str_rot_angles_lst, arr, label=t, marker='o')
            print(f'Recall (neg) for type {t}:\n{arr}')
        plt.ylim(0., 1.)
        plt.title('Manufacturers - Recall (neg)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles')
        plt.ylabel('Recall')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_folder}/aaa_recall_neg.png')
        plt.clf()
        plt.close()

        for arr, t in zip(avg_precs_neg, all_types):
            plt.plot(str_rot_angles_lst, arr, label=t, marker='o')
            print(f'Precision (neg) for type {t}:\n{arr}')
        plt.ylim(0., 1.)
        plt.title('Manufacturers - Precision (neg)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles')
        plt.ylabel('Precision')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_folder}/aaa_precision_neg.png')
        plt.clf()
        plt.close()

        for arr, t in zip(avg_dices_no_FP_no_FN_neg, all_types):
            plt.plot(str_rot_angles_lst, arr, label=t, marker='o')
            print(f'Dice (neg) for type {t}:\n{arr}')
        plt.ylim(0., 1.)
        plt.title('Manufacturers - Dice (neg)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles')
        plt.ylabel('Dice (TPs)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_folder}/aaa_dice_neg.png')
        plt.clf()
        plt.close()

        # add = ''
        # create_violin_plot(all_precs_pos, all_precs_neg, title='Precision (per image)', y_label='Precision', save_path=f'{plots_folder}/aaa_{add}violin_prec.png')
        # create_violin_plot(all_recs_pos, all_recs_neg, title='Recall (per image)', y_label='Recall', save_path=f'{plots_folder}/aaa_{add}violin_rec.png')
        # create_violin_plot(all_dices_pos, all_dices_neg, title='Dice (FP & FN) (per image)', y_label='Dice', save_path=f'{plots_folder}/aaa_{add}violin_dice_FP_FN.png')
        # create_violin_plot(all_dices_no_FP_pos, all_dices_no_FP_neg, title='Dice (no FP) (per image)', y_label='Dice', save_path=f'{plots_folder}/aaa_{add}violin_dice_no_FP_FN.png')
        # create_violin_plot(all_dices_no_FN_pos, all_dices_no_FN_neg, title='Dice (no FN) (per image)', y_label='Dice', save_path=f'{plots_folder}/aaa_{add}violin_dice_FP_no_FN.png')
        # create_violin_plot(all_dices_no_FP_no_FN_pos, all_dices_no_FP_no_FN_neg, title='Dice (no FP & no FN) (per image)', y_label='Dice', save_path=f'{plots_folder}/aaa_{add}violin_dice_no_FP_no_FN.png')
        # create_violin_plot(all_dices_per_mass_pos, all_dices_per_mass_neg, title='Dice (TP only) (per mass)', y_label='Dice', save_path=f'{plots_folder}/aaa_{add}violin_dice_TP_per_mass.png')
        #
        # ticks = [m + 1 for m in range(len(avg_precs_per_mass_pos))]
        #
        # plt.title("Average precision (per mass)")
        # plt.plot(ticks, avg_precs_per_mass_pos, label='pos', color='red', marker='o')
        # plt.plot(ticks, avg_precs_per_mass_neg, label='neg', color='green', marker='o')
        # plt.ylim(0., 1.)
        # plt.xticks(ticks=ticks, labels=['None\n(no mass change)', 'None', 'Low', 'Normal', 'High'], rotation=12, fontsize=10)
        # plt.xlabel('Orientation changes', fontsize=11)
        # plt.ylabel('Precision', fontsize=11)
        # plt.legend(loc='best')
        # plt.savefig(f'{plots_folder}/aaa_precision_per_mass_pos.png')
        # plt.clf()
        # plt.close()
        #
        # plt.title("Average recall (per mass)")
        # plt.plot(ticks, avg_recs_per_mass_pos, label='pos', color='red', marker='o')
        # plt.plot(ticks, avg_recs_per_mass_neg, label='neg', color='green', marker='o')
        # plt.ylim(0., 1.)
        # plt.xticks(ticks=ticks, labels=['None\n(no mass change)', 'None', 'Low', 'Normal', 'High'], rotation=12, fontsize=10)
        # plt.xlabel('Orientation changes', fontsize=11)
        # plt.ylabel('Recall', fontsize=11)
        # plt.legend(loc='best')
        # plt.savefig(f'{plots_folder}/aaa_recall_per_mass_pos.png')
        # plt.clf()
        # plt.close()
        #
        # idx_dict = {0: "angles (0, 0, 0) no masses", 1: "angles (0, 0, 0)", 2: "angles (5, 10, 10)", 3: "angles (15, 25, 25)", 4: "angles (25, 40, 40)"}
        # for c_idx, (pair_arr, prec_pos_arr, prec_neg_arr, rec_pos_arr, rec_neg_arr, dice_pos_arr, dice_neg_arr) in enumerate(zip(pair_names_arrs, all_precs_pos, all_precs_neg, all_recs_pos, all_recs_neg, all_dices_no_FP_no_FN_pos, all_dices_no_FP_no_FN_neg)):
        #     for precs, recs, dices, sign in ((prec_pos_arr, rec_pos_arr, dice_pos_arr, 'pos'), (prec_neg_arr, rec_neg_arr, dice_neg_arr, 'neg')):
        #         for arr, measure_name in ((precs, 'Precision'), (recs, 'Recall'), (dices, 'Dice')):
        #             idxs = np.argsort(arr)
        #             outlier_names = np.array(pair_arr)[idxs][:10]
        #             print(f'{idx_dict[c_idx]}\n{measure_name}\n{sign}\n{outlier_names}\n')

        exit()

        print(f'avg_dices_pos:\n{avg_dices_pos}\n')
        print(f'avg_precs_pos:\n{avg_precs_pos}\n')
        print(f'avg_recs_pos:\n{avg_recs_pos}\n')

        print(f'avg_dices_neg:\n{avg_dices_neg}\n')
        print(f'avg_precs_neg:\n{avg_precs_neg}\n')
        print(f'avg_recs_neg:\n{avg_recs_neg}\n')

        plt.plot(str_rot_angles_lst, avg_dices_pos)
        plt.ylim(0.2, 1.)
        plt.title('Dice pos (FP & FN)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_pos.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_precs_pos)
        plt.ylim(0.2, 1.)
        plt.title('Precision pos')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Precision')
        plt.savefig(f'{plots_folder}/aaa_prec_pos.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_recs_pos)
        plt.ylim(0.2, 1.)
        plt.title('Recall pos')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Recall')
        plt.savefig(f'{plots_folder}/aaa_rec_pos.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_dices_neg)
        plt.ylim(0.2, 1.)
        plt.title('Dice neg (FP & FN)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_neg.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_precs_neg)
        plt.ylim(0.2, 1.)
        plt.title('Precision neg')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Precision')
        plt.savefig(f'{plots_folder}/aaa_prec_neg.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_recs_neg)
        plt.ylim(0.2, 1.)
        plt.title('Recall neg')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Recall')
        plt.savefig(f'{plots_folder}/aaa_rec_neg.png')
        plt.clf()
        plt.close()
        
        ##################
        
        plt.plot(str_rot_angles_lst, avg_dices_no_FP_pos)
        plt.ylim(0.2, 1.)
        plt.title('Dice pos (no FP)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_no_FP_pos.png')
        plt.clf()
        plt.close()
        
        plt.plot(str_rot_angles_lst, avg_dices_no_FN_pos)
        plt.ylim(0.2, 1.)
        plt.title('Dice pos (no FN)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_no_FN_pos.png')
        plt.clf()
        plt.close()
        
        plt.plot(str_rot_angles_lst, avg_dices_no_FP_no_FN_pos)
        plt.ylim(0.2, 1.)
        plt.title('Dice pos (no FP & no FN)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_no_FP_no_FN_pos.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_dices_no_FP_neg)
        plt.ylim(0.2, 1.)
        plt.title('Dice neg (no FP)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_no_FP_neg.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_dices_no_FN_neg)
        plt.ylim(0.2, 1.)
        plt.title('Dice neg (no FN)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_no_FN_neg.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_dices_no_FP_no_FN_neg)
        plt.ylim(0.2, 1.)
        plt.title('Dice neg (no FP & no FN)')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('Dice coef')
        plt.savefig(f'{plots_folder}/aaa_dice_no_FP_no_FN_neg.png')
        plt.clf()
        plt.close()
        
        #######################################
        exit()
        
        plt.plot(str_rot_angles_lst, avg_gt_ccs_num_pos)
        plt.title('GT CCs num pos')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('CCs num')
        plt.savefig(f'{plots_folder}/aaa_gt_ccs_num_pos.png')
        plt.clf()
        plt.close()
        
        plt.plot(str_rot_angles_lst, avg_out_ccs_num_pos)
        plt.title('Output CCs num pos')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('CCs num')
        plt.savefig(f'{plots_folder}/aaa_out_ccs_num_pos.png')
        plt.clf()
        plt.close()
        
        plt.plot(str_rot_angles_lst, avg_TP_prec_pos)
        plt.title('TPs precision pos')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('TPs num')
        plt.savefig(f'{plots_folder}/aaa_TP_prec_pos.png')
        plt.clf()
        plt.close()
        
        plt.plot(str_rot_angles_lst, avg_FP_pos)
        plt.title('FPs pos')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('FPs num')
        plt.savefig(f'{plots_folder}/aaa_FP_pos.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_gt_ccs_num_neg)
        plt.title('GT CCs num neg')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('CCs num')
        plt.savefig(f'{plots_folder}/aaa_gt_ccs_num_neg.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_out_ccs_num_neg)
        plt.title('Output CCs num neg')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('CCs num')
        plt.savefig(f'{plots_folder}/aaa_out_ccs_num_neg.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_TP_prec_neg)
        plt.title('TPs precision neg')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('TPs num')
        plt.savefig(f'{plots_folder}/aaa_TP_prec_neg.png')
        plt.clf()
        plt.close()

        plt.plot(str_rot_angles_lst, avg_FP_neg)
        plt.title('FPs neg')
        plt.xticks(rotation=25, fontsize=9)
        plt.xlabel('Rotation angles\' max')
        plt.ylabel('FPs num')
        plt.savefig(f'{plots_folder}/aaa_FP_neg.png')
        plt.clf()
        plt.close()



