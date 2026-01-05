import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

import augmentations
from models import LongitudinalMIMModelBig, freeze_and_unfreeze
from plot_animations import rotate_ax
import numpy as np
from datasets import LongitudinalMIMDataset
import os
from constants import DEVICE
from torch.utils.data import DataLoader
import numpy.random as np_rd
import random
import matplotlib.pyplot as plt
import nibabel as nib


def load_cases(p, bl_names, fu_names):
    seg_p = p + '_segs'

    priors = []
    currents = []
    pair_names = []

    for bl_name, fu_name in zip(bl_names, fu_names):
        pair_name = f"{bl_name.split('.')[0]}_{fu_name.split('.')[0]}"
        pair_names.append(pair_name)
        prior_path = p + '/' + bl_name
        prior_seg_path = seg_p + '/' + bl_name.split('.')[0] + '_seg.nii.gz'
        current_path = p + '/' + fu_name
        current_seg_path = seg_p + '/' + fu_name.split('.')[0] + '_seg.nii.gz'

        prior, current = pair_prep(prior_path, prior_seg_path, current_path, current_seg_path)
        priors.append(prior)
        currents.append(current)

    return priors, currents, pair_names


def load_test_set():
    cases_dirs = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_sigal/images', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_sigal/images2', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_sigal/images3',
                  '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_nitzan/images']
    bl_names_arrs = [['p1_1.nii.gz', 'p1_2.nii.gz', 'p2_1.nii.gz', 'p2_2.nii.gz', 'p2_3.nii.gz', 'p3_1.nii.gz', 'p4_1.nii.gz', 'p4_2.nii.gz'],
                     [f'{i}A.nii.gz' for i in range(1, 25)],
                     [f'{i}A.nii.gz' for i in range(25, 38)] + ['26B.nii.gz', '36B.nii.gz', '36C.nii.gz'],
                     ['A1.nii.gz', 'A2.nii.gz', 'A3.nii.gz', 'A4.nii.gz', 'A5.nii.gz', 'A6.nii.gz', 'B1.nii.gz', 'B2.nii.gz', 'B3.nii.gz',
                      'C1.nii.gz', 'C2.nii.gz', 'C3.nii.gz', 'D1.nii.gz', 'D2.nii.gz', 'D3.nii.gz', 'D4.nii.gz', 'E1.nii.gz', 'E2.nii.gz',
                      'F1.nii.gz', 'F2.nii.gz', 'F3.nii.gz', 'F4.nii.gz', 'G1.nii.gz', 'H1.nii.gz', 'H2.nii.gz', 'I1.nii.gz', 'I2.nii.gz',
                      'I3.nii.gz', 'J1.nii.gz', 'J2.nii.gz', 'J3.nii.gz', 'J4.nii.gz'] + ['AA1.nii.gz', 'AA2.nii.gz', 'BB1.nii.gz', 'CC1.nii.gz', 'DD1.nii.gz', 'DD2.nii.gz']]

    fu_names_arrs = [['p1_2.nii.gz', 'p1_3.nii.gz', 'p2_2.nii.gz', 'p2_3.nii.gz', 'p2_4.nii.gz', 'p3_2.nii.gz', 'p4_2.nii.gz', 'p4_3.nii.gz'],
                     [f'{i}B.nii.gz' for i in range(1, 25)],
                     [f'{i}B.nii.gz' for i in range(25, 38)] + ['26C.nii.gz', '36C.nii.gz', '36D.nii.gz'],
                     ['A2.nii.gz', 'A3.nii.gz', 'A4.nii.gz', 'A5.nii.gz', 'A6.nii.gz', 'A7.nii.gz', 'B2.nii.gz', 'B3.nii.gz', 'B4.nii.gz',
                      'C2.nii.gz', 'C3.nii.gz', 'C4.nii.gz', 'D2.nii.gz', 'D3.nii.gz', 'D4.nii.gz', 'D5.nii.gz', 'E2.nii.gz', 'E3.nii.gz',
                      'F2.nii.gz', 'F3.nii.gz', 'F4.nii.gz', 'F5.nii.gz', 'G2.nii.gz', 'H2.nii.gz', 'H3.nii.gz', 'I2.nii.gz', 'I3.nii.gz',
                      'I4.nii.gz', 'J2.nii.gz', 'J3.nii.gz', 'J4.nii.gz', 'J5.nii.gz'] + ['AA2.nii.gz', 'AA3.nii.gz', 'BB2.nii.gz', 'CC2.nii.gz', 'DD2.nii.gz', 'DD3.nii.gz']]

    priors = []
    currents = []
    pair_names = []

    for p, bl_names, fu_names in zip(cases_dirs, bl_names_arrs, fu_names_arrs):
        c_priors, c_currents, c_pair_names = load_cases(p, bl_names, fu_names)
        priors.extend(c_priors)
        currents.extend(c_currents)
        pair_names.extend(c_pair_names)

    priors = torch.stack(priors, dim=0)
    currents = torch.stack(currents, dim=0)

    return priors, currents, pair_names


def embed_test_set():
    priors, currents, pair_names = load_test_set()

    c_preds = []

    split_priors = torch.split(priors, 4)
    split_currents = torch.split(currents, 4)

    for p, c in zip(split_priors, split_currents):
        c_preds.append(model(p.to(DEVICE), c.to(DEVICE)).to('cpu'))
    c_preds = torch.cat(c_preds)

    return c_preds, pair_names


def plot_PCA_projected(data_arrs, labels, n_components, path):
    data = torch.cat(data_arrs, dim=0)

    mean = torch.mean(data, dim=0)
    centered_data = data - mean

    pca = PCA(n_components=n_components)
    # pca = PCA(.95)

    pca.fit(centered_data)

    fig = plt.figure()
    if n_components == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    # proj0 = pca.transform(data_arrs[0])
    # proj1 = pca.transform(data_arrs[1])
    # cent0 = np.mean(proj0, axis=0)
    # cent1 = np.mean(proj1, axis=0)
    # cents = [cent0, cent1]

    projected_data_arrs = []

    for i, (arr, label) in enumerate(zip(data_arrs, labels)):
        print(f'working on label {label}')

        label = str(label)

        projected_data = torch.tensor(pca.transform(arr))
        projected_data = (projected_data[torch.argwhere(torch.amax(projected_data.abs().flatten(1, -1), dim=1) < 1500)]).squeeze().numpy()

        projected_data_arrs.append(projected_data)

        # Plot the original data with eigenvectors
        if n_components == 3:
            ax.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], label=label, alpha=0.7)
        elif n_components == 2:
            ax.scatter(projected_data[:, 0], projected_data[:, 1], label=label, alpha=0.7)
            #
            # if path_arrs:
            #     paths = np.array(path_arrs[i], dtype=np.str_)
            #     other_cent = cents[(label_num + 1) % 2]
            #     print(other_cent)
            #     cent_dists = np.linalg.norm(projected_data - other_cent, ord=None, axis=1)
            #     furthest_idxs = np.argsort(cent_dists)
            #     furthest_paths = np.take_along_axis(paths, furthest_idxs, axis=0)[:10]
            #     print(furthest_paths)
            #     furthest_dists = np.take_along_axis(projected_data, furthest_idxs[:, np.newaxis], axis=0)[:10]
            #     print(furthest_dists)

    # plt.axis('equal')

    plt.legend()
    # plt.show()
    # exit()
    if n_components == 3:
        rotate_ax(fig, ax, path + '.gif')
    else:
        plt.savefig(path + '.png')

    return projected_data_arrs


def get_diff_nodiff_labeled_latents(c_preds, c_diff_maps):
    max_diff_vals = torch.amax(c_diff_maps.abs().flatten(1, -1), dim=1)
    diff_idxs = torch.argwhere(max_diff_vals > 0.02)
    no_diff_idxs = torch.argwhere(max_diff_vals <= 0.02)

    diff_preds = c_preds[diff_idxs].squeeze(1)
    no_diff_preds = c_preds[no_diff_idxs].squeeze(1)

    return [diff_preds, no_diff_preds], ['diff', 'no_diff']


def get_inpaint_labeled_latents(c_preds, c_diff_labels):
    assert len(inpaint_dirs) != 0
    inpainted_abnormal_disordered = c_preds[torch.argwhere(c_diff_labels == 0)].squeeze(1)
    inpainted_abnormal_lower = c_preds[torch.argwhere(c_diff_labels == 1)].squeeze(1)
    inpainted_healthy = c_preds[torch.argwhere(c_diff_labels == 2)].squeeze(1)
    inpainted_healthy_abnormal_disordered = c_preds[torch.argwhere(c_diff_labels == 3)].squeeze(1)
    inpainted_healthy_abnormal_lower = c_preds[torch.argwhere(c_diff_labels == 4)].squeeze(1)
    # return [inpainted_abnormal_disordered, inpainted_abnormal_lower, inpainted_healthy, inpainted_healthy_abnormal_disordered, inpainted_healthy_abnormal_lower],\
    #     ['abnormal_disordered', 'abnormal_lower', 'healthy', 'healthy_abnormal_disordered', 'healthy_abnormal_lower']
    return [inpainted_abnormal_disordered, inpainted_abnormal_lower, inpainted_healthy], \
        ['abnormal_disordered', 'abnormal_lower', 'healthy']


def get_batch_from_dataloader(dl, filt=False):
    priors, currents, c_diff_maps, c_labels, c_diff_labels = next(iter(dl))

    split_priors = torch.split(priors, 4)
    split_currents = torch.split(currents, 4)

    c_preds = []

    for p, c in zip(split_priors, split_currents):
        c_preds.append(model(p.to(DEVICE), c.to(DEVICE)).to('cpu'))
    c_preds = torch.cat(c_preds)

    if filt:
        max_diff_vals = torch.amax(c_diff_maps.abs().flatten(1, -1), dim=1)
        diff_idxs = torch.argwhere(max_diff_vals > 0.04)
        c_preds = c_preds[diff_idxs].squeeze(1)

    return c_preds, c_diff_maps, c_labels, c_diff_labels


def get_batch_with_abnormalization_modifications(ds: LongitudinalMIMDataset):
    abnor_tf = ds.random_abnormalization_tf
    abnor_lists = [[abnor_tf.masses], [abnor_tf.lesions], [abnor_tf.general_opacity], [abnor_tf.disordered_opacity], [abnor_tf.contour_deformation], [abnor_tf.devices]]
    # abnor_lists = [[abnor_tf.general_opacity]]

    ds.random_bl_fu_flip_tf.p = 0.5
    ds.random_channels_flip_tf.p = 0.5
    ds.overlay_diff_p = 1.
    ds.abnor_both_p = 0.5

    f_preds = []

    for abnors in abnor_lists:
        abnor_tf.abnormalities_general = abnors

        dl = DataLoader(ds, shuffle=True, batch_size=120)
        c_preds, _, __, ___ = get_batch_from_dataloader(dl, filt=True)

        f_preds.append(c_preds)

    # ds.random_channels_flip_tf.p = 2.
    # ds.overlay_diff_p = 0.

    # for abnors in abnor_lists:
    #     abnor_tf.abnormalities_general = abnors
    #
    #     dl = DataLoader(ds, shuffle=True, batch_size=100)
    #     c_preds, _, __, ___ = get_batch_from_dataloader(dl)
    #
    #     f_preds.append(c_preds)

    # abnor_tf.abnormalities_general = []
    # ds.overlay_diff_p = 0.
    #
    # dl = DataLoader(ds, shuffle=True, batch_size=50)
    # c_preds, _, __, ___ = get_batch_from_dataloader(dl)
    #
    # f_preds.append(c_preds)

    f_labels = [
        'masses', 'nodules', 'general_opacity', 'disordered_opacity', 'contour_deformation', 'devices',
        # 'masses_neg', 'nodules_neg', 'general_opacity_neg', 'disordered_opacity_neg', 'contour_deformation_neg', 'devices_neg'
    ]
    # f_labels = ['contour_deformation_pos', 'contour_deformation_neg']
    # f_labels = ['general_opacity']
    return f_preds, f_labels


def get_batch_with_overlay_parameter_scanning(ds: LongitudinalMIMDataset):
    seed = random.randint(1, 10000000)

    ds.shuffle()

    abnor_tf = ds.random_abnormalization_tf
    abnor_tf.abnormalities_general = [abnor_tf.masses]
    ds.random_bl_fu_flip_tf.p = 0
    ds.random_channels_flip_tf.p = 0

    # abnor_tf.ab_power = 24
    frequency_scale = 48
    abnor_tf.noise = (torch.rand(1, int(512 / frequency_scale), int(512 / frequency_scale)) * 1.2) ** 5
    abnor_tf.inv_size = 0.65
    abnor_tf.blur_fac = 9

    power_vals = [4,8,12,16,20,24,28,32,36,40]
    # blur_vals = [1,3,5,7,9,11,13,15,17,19]
    # inv_sizes = [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]

    f_preds = []
    f_labels = []

    for k, v_i in enumerate(power_vals):
        f_labels.append(f'masses_intensity_{k}')
        abnor_tf.ab_power = v_i

        torch.manual_seed(seed)
        random.seed(seed)
        np_rd.seed(seed)

        dl = DataLoader(ds, shuffle=False, batch_size=1)
        c_preds, _, __, ___ = get_batch_from_dataloader(dl)
        f_preds.append(c_preds)

    return f_preds, f_labels


if __name__ == '__main__':
    mask_crop_tf = augmentations.CropResizeWithMaskTransform()
    rescale_tf = augmentations.RescaleValuesTransform()
    im_and_seg_getter = lambda im_path, seg_path: (torch.tensor(nib.load(im_path).get_fdata().T[None, ...]), torch.tensor(nib.load(seg_path).get_fdata().T[None, ...]))
    prep_with_seg = lambda im, seg: rescale_tf(mask_crop_tf(torch.cat([im, seg], dim=0))[0])
    im_prep = lambda im_path, seg_path: prep_with_seg(*im_and_seg_getter(im_path, seg_path))
    pair_prep = lambda prior_path, prior_seg_path, current_path, current_seg_path: (im_prep(prior_path, prior_seg_path), im_prep(current_path, current_seg_path))

    entity_dirs = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/images', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images']
    # entity_dirs = []

    # inpaint_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs'
    # inpaint_dirs = [f'{inpaint_dir}/{d}' for d in os.listdir(inpaint_dir)]
    # inpaint_dirs.remove('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/unrelated_healthy')
    # inpaint_dirs.remove('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/inpainted_healthy_abnormal_disordered')
    # inpaint_dirs.remove('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/inpainted_healthy_abnormal_lower')
    inpaint_dirs = []

    # DRR_dirs = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/DRRs']
    DRR_dirs = []

    # model = LongitudinalMIMModelBig(use_technical_bottleneck=True).to(DEVICE)
    model = LongitudinalMIMModelBig(use_technical_bottleneck=False).to(DEVICE)
    # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id30_Epoch1__FT29_TechnicalBottleneck128_Longitudinal_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
    # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id29_Epoch3_Longitudinal_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
    # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id31_Epoch1_Longitudinal_DeviceInvariant_DRRs_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
    model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id18_FTid16_Epoch2_Longitudinal_Devices2_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
    checkpoint_dict = torch.load(model_path)
    model.load_state_dict(checkpoint_dict['model_dict'], strict=True)
    model.eval()
    freeze_and_unfreeze([model], [])
    model.return_latent = True

    dataset = LongitudinalMIMDataset(entity_dirs=entity_dirs, inpaint_dirs=inpaint_dirs, DRR_dirs=DRR_dirs, invariance=None,
                                     overlay_diff_p=0.5, abnor_both_p=0.5)
    dataset.return_label = True

    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=300)
    preds, diff_maps, labels, diff_labels = get_batch_from_dataloader(train_dataloader)

    # latents, plot_labels = get_inpaint_labeled_latents(preds, diff_labels)

    # latents, plot_labels = get_diff_nodiff_labeled_latents(preds, diff_maps)

    # latents, plot_labels = get_batch_with_abnormalization_modifications(dataset)

    # latents, plot_labels = get_batch_with_overlay_parameter_scanning(dataset)

    latents1, plot_labels1 = get_diff_nodiff_labeled_latents(preds, diff_maps)
    latents2, plot_labels2 = get_batch_with_abnormalization_modifications(dataset)

    test_latents, ICU_pair_names = embed_test_set()

    # latents = latents1 + [test_latents]
    # plot_labels = plot_labels1 + ['ICU Pairs']

    latents = [latents1[1]] + latents2 + [test_latents]
    plot_labels = [plot_labels1[1]] + plot_labels2 + ['ICU Pairs']

    # latents = latents1 + latents2
    # plot_labels = plot_labels1 + plot_labels2

    # latents = latents1
    # plot_labels = plot_labels1

    plot_PCA_projected(latents, labels=plot_labels, n_components=2,
                       path='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/latent_PCA/test_set_diff_no_diff')
    projected_arrs = plot_PCA_projected(latents, labels=plot_labels, n_components=3,
                                        path='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/latent_PCA/test_set_diff_no_diff')

    from sklearn.neighbors import KNeighborsClassifier

    ICU_pair_names = np.array(ICU_pair_names)
    projected_icu = projected_arrs[-1]
    projected_no_diff = projected_arrs[0]
    projected_masses = projected_arrs[1]
    projected_nodules = projected_arrs[2]
    projected_general_opacity = projected_arrs[3]
    projected_disordered_opacity = projected_arrs[4]
    projected_contour_deformation = projected_arrs[5]
    projected_devices = projected_arrs[6]

    all_projections = [projected_no_diff, projected_masses, projected_nodules, projected_general_opacity, projected_disordered_opacity, projected_devices]
    data = np.concatenate(all_projections, axis=0)
    classes = []
    for i in range(6):
        classes.extend([i for _ in range(all_projections[i].shape[0])])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data, classes)

    label_dict = {0: 'No_diff', 1: 'Masses', 2: 'Nodules', 3: 'General_opacity', 4: 'Disordered_opacity', 5: 'Devices'}
    for j, icu_pair in enumerate(projected_icu):
        print(f'Pair name = {ICU_pair_names[j]}')
        label_raw = knn.predict([icu_pair])[0]
        print(f'Label raw = {label_raw}')
        print(f'Label = {label_dict[label_raw]}')

    # get_10_closest_to_cent = lambda projected1, projected2: ICU_pair_names[np.argsort(np.linalg.norm(projected1 - np.mean(projected2, axis=0), axis=1))[:10]]
    #
    # no_diff_mean = np.mean(projected_no_diff, axis=0)
    # icu_dist = np.linalg.norm(projected_icu - no_diff_mean, axis=1)
    # max_no_diff_ICU = ICU_pair_names[np.argsort(icu_dist)[:10]]
    # print('Closest to no diff')
    # print(max_no_diff_ICU)
    # min_no_diff_ICU = ICU_pair_names[np.argsort(-icu_dist)[:10]]
    # print('Furthest from no diff')
    # print(min_no_diff_ICU)
    #
    # max_masses = get_10_closest_to_cent(projected_icu, projected_masses)
    # max_nodules = get_10_closest_to_cent(projected_icu, projected_nodules)
    # max_general_opacity = get_10_closest_to_cent(projected_icu, projected_general_opacity)
    # max_disordered_opacity = get_10_closest_to_cent(projected_icu, projected_disordered_opacity)
    # max_contour_deformation = get_10_closest_to_cent(projected_icu, projected_contour_deformation)
    # max_devices = get_10_closest_to_cent(projected_icu, projected_devices)
    #
    # print('Closest to Masses')
    # print(max_masses)
    #
    # print('Closest to Nodules')
    # print(max_nodules)
    #
    # print('Closest to General Opacity')
    # print(max_general_opacity)
    #
    # print('Closest to Disordered Opacity')
    # print(max_disordered_opacity)
    #
    # print('Closest to Contour Deformation')
    # print(max_contour_deformation)
    #
    # print('Closest to Devices')
    # print(max_devices)



