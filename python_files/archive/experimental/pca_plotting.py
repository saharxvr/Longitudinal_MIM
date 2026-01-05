import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from datasets import ContrastiveLearningDataset, ContrastiveLearningSmallDataset, NormalAbnormalDataset, GeneralContrastiveLearningDataset
from models import MaskedReconstructionModel, freeze_and_unfreeze, DetectionContrastiveModel#, ContrastiveLearningBottleneck
from plot_animations import rotate_ax
from constants import *
from preprocessing import BatchPreprocessingImagewise, dedisorder_batch
import numpy as np


def plot_PCA_projected(data_arrs, labels, label_nums, path, n_components, path_arrs):
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

    proj0 = pca.transform(data_arrs[0])
    proj1 = pca.transform(data_arrs[1])
    cent0 = np.mean(proj0, axis=0)
    cent1 = np.mean(proj1, axis=0)
    cents = [cent0, cent1]

    for i, (arr, label, label_num) in enumerate(zip(data_arrs, labels, label_nums)):
        print(f'working on label {label}')

        label = str(label)

        projected_data = pca.transform(arr)

        # Plot the original data with eigenvectors
        if n_components == 3:
            ax.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], label=label, alpha=0.7)
        elif n_components == 2:
            ax.scatter(projected_data[:, 0], projected_data[:, 1], label=label, alpha=0.7)
            #
            if path_arrs:
                paths = np.array(path_arrs[i], dtype=np.str_)
                other_cent = cents[(label_num + 1) % 2]
                print(other_cent)
                cent_dists = np.linalg.norm(projected_data - other_cent, ord=None, axis=1)
                furthest_idxs = np.argsort(cent_dists)
                furthest_paths = np.take_along_axis(paths, furthest_idxs, axis=0)[:10]
                print(furthest_paths)
                furthest_dists = np.take_along_axis(projected_data, furthest_idxs[:, np.newaxis], axis=0)[:10]
                print(furthest_dists)

    # plt.axis('equal')

    plt.legend()
    # plt.show()
    # exit()
    if n_components == 3:
        rotate_ax(fig, ax, path + '.gif')
    else:
        plt.savefig(path + '.png')


if __name__ == '__main__':
    dims = 2
    save_path = PROJECT_FOLDER + f'/plots/CL/PCA/NormalAbnormal_LateProjNoNormalize_test_BasePathologies_{dims}d'
    # train_dataset = ContrastiveLearningDataset(CXR14_FOLDER + '/AP_images',
    #                                            CXR14_FOLDER + '/train_labels.csv',
    #                                            CUR_LABELS,
    #                                            groups=CUR_LABEL_GROUPS,
    #                                            do_shuffle=True,
    #                                            get_weights=GET_WEIGHTS,
    #                                            label_no_finding=LABEL_NO_FINDING,
    #                                            train=False)
    train_dataset = NormalAbnormalDataset(rot_chance=0.0, ver_flip_change=0.0, hor_flip_chance=0.0, mode='test', use_non_expert=True)
    # train_dataset = GeneralContrastiveLearningDataset(
    #     [
    #         '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_test.csv',
    #         '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_train.csv',
    #         '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/specific_abnormalities_labels.csv'
    #     ],
    #     ['Localized', 'Interstitial', 'Cardiomegaly'],
    #     mode='train', unlabeled_perc=0.1, rot_chance=0., hor_flip_chance=0., ver_flip_change=0.)
    preprocess = BatchPreprocessingImagewise(use_fourier=False, use_canny=USE_CANNY, clip_limit=2.2, clahe_prob=1., rand_crop_prob=0.0, blur_prob=0.0)
    model = DetectionContrastiveModel(in_channels=DETECTION_IN_CHANNELS).to(DEVICE)

    # model_path_end = 'Checkpoint_Epoch5_BatchSize_208_NewerLoss_Conv1x1Reduce_Labels_[Consolidation|Infiltration|Mass|Nodule][Edema]_MIM_.pt'
    # model_path = PROJECT_FOLDER + f'saved_models/CL/DETECTION_Checkpoint_Decoder1_Augs_00075_Eff_ViT_Epoch5_Decoder2_L1L2Fourier_GN/{model_path_end}'
    # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/CL/Checkpoint_id0_GeneralDS_01dropout_Epoch60_BatchSize_128_ContStartEpoch52_1Channels_LaterProjNoNormalize_ContLossLambda001_NormLossLambda01_Labels_Localized|Interstitial|Pneumothorax_NoDedisorder.pt'
    # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/CL/Checkpoint_New_Epoch20_BatchSize_176_ContStartEpoch15_1Channels_LaterProjNoNormalize_ContLossLambda001_NormLossLambda05_Labels_Abnormal|Normal_NoDedisorder.pt'
    # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/CL/Checkpoint_GeneralDS_01dropout_Epoch25_BatchSize_176_ContStartEpoch20_1Channels_IncludingUnlabeled_LaterProjNoNormalize_ContLossLambda001_NormLossLambda01_Labels_Localized|Interstitial|Pneumothorax|Normal_NoDedisorder.pt'
    model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/CL/Checkpoint_id6_GeneralDS_033EncDropout_Epoch25_BatchSize_128_ContStartEpoch33_1Channels_LaterProjNoNormalize_ContLossLambda001_NormLossLambda01_Labels_Abnormal|Normal_NoDedisorder.pt'
    checkpoint_dict = torch.load(model_path)
    if USE_DEDISORDER:
        mim_model = MaskedReconstructionModel(dec=5, in_channels=MASKED_IN_CHANNELS,
                                              use_mask_token=USE_MASK_TOKEN,
                                              use_pos_embed=USE_POS_EMBED).to(DEVICE)
        mim_path = checkpoint_dict['mim_model_path']
        mim_checkpoint = torch.load(mim_path)
        mim_model.load_state_dict(mim_checkpoint['model_dict'])
        mim_model.eval()
        freeze_and_unfreeze([mim_model], [])
    # if 'mim_model_dict' not in checkpoint_dict:
    #     mim_model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/MIM/ToTest/old/Checkpoint_Decoder1_Augs_00075_Eff_ViT_Epoch5_Decoder2_L1L2Fourier_GN.pt'
    #     mim_checkpoint_dict = torch.load(mim_model_path)
    #     checkpoint_dict['mim_model_dict'] = mim_checkpoint_dict['model_dict']

    # mim_model.load_state_dict(checkpoint_dict['mim_model_dict'])
    # mim_model = mim_model.encoder_bottleneck
    model.load_state_dict(checkpoint_dict['model_dict'])

    # freeze_and_unfreeze([mim_model, model], [])
    freeze_and_unfreeze([model], [])

    model.contrastive_detection_stage()
    model.eval()

    # pos_labels = [0, 1, [1, 1, 0], 2]
    labels_dict = train_dataset.labels_dict
    print(labels_dict)
    # labels_dict = {0: 'Localized', 1: 'Interstitial', 2: 'Pneumothorax', 3: 'No finding', 4: 'Loc + Inter',
    #                5: 'Loc + Pneumothorax', 6: 'Inter + Pneumothorax', 7: 'Loc + Inter + Pneumothorax', 8: 'None', (1, 1, 0, 0): 'Loc + Inter'}
    # labels_dict = {0: 'Pneumothorax', 1: 'Atelectasis', 2: 'Consolidation', 3: 'Nodule_Mass', 4: 'ILD', 5: 'Fibrosis'}
    # pos_labels = [0, 1, 2,3,4,5,6,7,8,9,10,11]
    pos_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    label_meanings = [labels_dict[l] if type(l) == int else '+'.join([labels_dict[i] for i, c_l in enumerate(l) if c_l != 0]) for l in pos_labels]
    mean_arrs = [[] for l in pos_labels]

    for it in range(1):
        print(f'Iteration {it + 1}')

        im_arrs = []
        name_arrs = []

        for c_label in pos_labels:
            # items, names = train_dataset.get_items_by_label_idx(c_label, count=200, get_names=True)
            items = train_dataset.get_items_by_label_idx(c_label, count=200, get_names=False)
            im_arrs.append(preprocess(items)[0])
            # name_arrs.append(names)

        # im_arrs = [torch.cat([model(mim_model(pre_im[None, ...])).to('cpu').detach() for pre_im in pre_ims], dim=0) for pre_ims in im_arrs]
        if not USE_DEDISORDER:
            model_outs_lists = [[model(pre_im[None, ...]) for pre_im in pre_ims] for pre_ims in im_arrs]
            proj_im_arrs = [torch.cat([out[1].to('cpu').detach() for out in out_list], dim=0) for out_list in model_outs_lists]
            pred_arrs = [torch.cat([out[0].to('cpu').detach() for out in out_list], dim=0) for out_list in model_outs_lists]
            for arr, i in zip(pred_arrs, range(len(pos_labels))):
                mean_v = torch.nn.functional.sigmoid(torch.mean(arr, dim=0)).tolist()
                print(f'{"/".join(label_meanings)} average probabilities for {label_meanings[i]}: {mean_v}')
                mean_arrs[i].append(mean_v)
            # print(f'label 0 acc: {torch.sum((torch.round(torch.nn.functional.sigmoid(pred_arrs[0])) - torch.tensor([1,0]).repeat((pred_arrs[0].shape[0], 1)))[:,0])}')
            # print(f'label 1 acc: {torch.sum((torch.round(torch.nn.functional.sigmoid(pred_arrs[1])) - torch.tensor([0,1]).repeat((pred_arrs[1].shape[0], 1)))[:,1])}')
        else:
            model_outs_lists = [[model(torch.cat([pre_im[None, ...], pre_im[None, ...] - dedisorder_batch(pre_im[None,...], cannys=None, mim_model=mim_model)], dim=1)) for pre_im in pre_ims] for pre_ims in im_arrs]
            proj_im_arrs = [torch.cat([out[1].to('cpu').detach() for out in outs_list], dim=0) for outs_list in model_outs_lists]
            pred_arrs = [torch.cat([out[0].to('cpu').detach() for out in out_list], dim=0) for out_list in model_outs_lists]
            for arr, i in zip(pred_arrs, range(len(pos_labels))):
                mean_v = torch.nn.functional.sigmoid(torch.mean(arr, dim=0)).tolist()
                print(f'{"/".join(label_meanings)} average probabilities for {label_meanings[i]}: {mean_v}')
                mean_arrs[i].append(mean_v)
        plot_PCA_projected(proj_im_arrs, label_meanings, pos_labels, save_path, n_components=dims, path_arrs=name_arrs)

    # mean_arrs = torch.tensor(mean_arrs)
    # for i in range(len(pos_labels)):
    #     print(f'Final average {"/".join(label_meanings)} average probabilities for {label_meanings[i]} are {torch.mean(mean_arrs[i], dim=0).tolist()}')



# a = [torch.rand((40, 128)), torch.rand((40, 128)), torch.rand((40, 128))]
# b = [torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([1, 1])]
# plot_PCA_projected(a, b)