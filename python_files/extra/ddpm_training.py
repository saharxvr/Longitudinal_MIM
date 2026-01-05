import random

import os
import shutil
from glob import glob, iglob
import pandas as pd
from tqdm import tqdm
import monai as mn
import bkh_pytorch_utils as bpu
import torch
import matplotlib.pyplot as plt
import yaml
from mediffusion import DiffusionModule, Trainer


# if not os.path.isdir("workshop_data"):
#     gdown.download(
#         "https://drive.google.com/uc?export=download&confirm=pbef&id=1r_lLkw8JQR2EUCPI6TLqsq5S_mV2SOQU",
#         "workshop_data.zip",
#         quiet=False,
#     )
#     os.mkdir("workshop_data")
#
#     with zipfile.ZipFile("workshop_data.zip", 'r') as zip_ref:
#         zip_ref.extractall("workshop_data")
#
#     os.remove("workshop_data.zip")
#
#     gdown.download(
#         "https://drive.google.com/uc?export=download&confirm=pbef&id=1L_gKWO87A4qCL1H95yfMA3TmEwXjhtr8",
#         "ddpm_weights.zip",
#         quiet=False,
#     )
#     with zipfile.ZipFile("ddpm_weights.zip", 'r') as zip_ref:
#         zip_ref.extractall("workshop_data")
#
#     os.remove("ddpm_weights.zip")


def create_Chexpert_df():
    def Chexpert_csv_to_path(c_row):
        c_path = c_row['Path']
        c_path_split = c_path.split('/')
        # split_type = c_path_split[1]
        img_path = '/'.join(c_path_split[2:]).split('.')[0] + '.nii.gz'
        c_file = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_new/{img_path}'
        # c_file = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/{split_type}/images/{img_path}'
        return c_file

    train_labels_csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/train.csv'
    valid_labels_csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/valid/valid.csv'

    c_path_col = 'Path'
    c_labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
              'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
              'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Support Devices']
    drop_labels = ['Pleural Other', 'Fracture']
    train_df = pd.read_csv(train_labels_csv_path)
    valid_df = pd.read_csv(valid_labels_csv_path)
    c_df = pd.concat([train_df, valid_df])
    c_df = c_df[c_df.apply(lambda r: r[c_path_col] not in {'CheXpert-v1.0/train/patient05271/study5/view2_frontal.jpg', 'CheXpert-v1.0/train/patient48043/study1/view2_frontal.jpg',
                                                   'CheXpert-v1.0/train/patient09797/study4/view1_frontal.jpg', 'CheXpert-v1.0/train/patient12362/study1/view1_frontal.jpg',
                                                   'CheXpert-v1.0/train/patient25979/study8/view1_frontal.jpg', 'CheXpert-v1.0/train/patient44163/study1/view1_frontal.jpg',
                                                   'CheXpert-v1.0/train/patient40255/study2/view1_frontal.jpg'}, axis=1)]

    c_df = c_df[c_df['Frontal/Lateral'] == 'Frontal']
    c_df = c_df.dropna(axis=0, subset=['Frontal/Lateral'])
    c_df = c_df.loc[~((c_df['Pleural Other'] == 1.) | (c_df['Fracture'] == 1.))]
    c_df.drop(columns=drop_labels, inplace=True)
    c_df[c_df[c_labels].isna()] = 0.
    c_df[c_labels] = c_df[c_labels].astype(float)
    c_df[c_df[c_labels] == -1.] = 0.5
    c_df[c_path_col] = c_df.apply(lambda r: Chexpert_csv_to_path(r), axis=1)
    return c_df


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['WANDB_API_KEY'] = "1106fae45f2bf456d9ae3c0014900989d20a2863"
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    TOTAL_IMAGE_SEEN = 1e6
    BATCH_SIZE = 3
    NUM_DEVICES = 1
    TRAIN_ITERATIONS = int(TOTAL_IMAGE_SEEN / (BATCH_SIZE * NUM_DEVICES))

    path_col = 'Path'
    labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
              'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
              'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Support Devices']
    df = create_Chexpert_df()

    data_dictionary = []
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        # file = Chexpert_csv_to_path(row)
        file = row[path_col]
        cls_labels = list(row[labels])
        data_dictionary.append({"img": file, "cls": cls_labels})

    train_data = data_dictionary[:-100]
    val_data = data_dictionary[-100:]
    #

    transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys="img"),
        bpu.EnsureGrayscaleD(keys=["img"]),
        mn.transforms.ResizeD(keys='img', size_mode="all", mode="bilinear", spatial_size=(512, 512), align_corners=False),
        mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=-1, b_max=1, clip=True),
        # mn.transforms.SpatialPadD(keys='img', spatial_size=(512, 512), mode="constant", constant_values=-1),
        # # mn.transforms.ToTensorD(keys=["cls"], dtype=torch.float),
        # # mn.transforms.AsDiscreteD(keys=["cls"], to_onehot=[3]),
        mn.transforms.SelectItemsD(keys=["img", "cls"]),
        mn.transforms.ToTensorD(keys=["img", "cls"], dtype=torch.float32, track_meta=False),
        # mn.transforms.SqueezeDimD(keys='img', dim=0, update_meta=False)
    ])

    if os.path.exists("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/.cache/train"):
        shutil.rmtree("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/.cache/train")
    if os.path.exists("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/.cache/val"):
        shutil.rmtree("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/.cache/val")
    os.makedirs("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/.cache", exist_ok=True)

    train_ds = mn.data.PersistentDataset(data=train_data, transform=transforms, cache_dir="/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/.cache/train")
    valid_ds = mn.data.PersistentDataset(data=val_data, transform=transforms, cache_dir="/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/.cache/val")

    # print(train_data[0])
    # print(transforms(train_data[0])['img'].shape)
    # exit()

    # for i, p in tqdm(enumerate(train_ds)):
    #     var = torch.var(p['img'])
    #     if var < 0.1:
    #         print(var)
    #         print(i)
    #         print("####")
    # for i, p in tqdm(enumerate(valid_ds)):
    #     var = torch.var(p['img'])
    #     if var < 0.1:
    #         print(var)
    #         print(i)
    #         print("####")
    # exit()

    train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=int(TOTAL_IMAGE_SEEN))

    # with open("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/config.yaml", "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    model = DiffusionModule(
        "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/config.yaml",
        train_ds=train_ds,
        val_ds=valid_ds,
        dl_workers=2,
        train_sampler=train_sampler,
        batch_size=BATCH_SIZE,               # train batch size
        val_batch_size=BATCH_SIZE//2         # validation batch size (recommended size is half of batch_size)
    )

    trainer = Trainer(
        max_steps=TRAIN_ITERATIONS,
        val_check_interval=5000,
        root_directory="/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs", # where to save the weights and logs
        precision="32",
        devices=-1,                 # use all the devices in CUDA_VISIBLE_DEVICES
        nodes=1,
        wandb_project="Mediffusion",
        logger_instance="Mediffusion_Chexpert_512",
    )

    # with torch.autocast(device_type='cuda', dtype=torch.float32):
    trainer.fit(model)

    ###########
    #
    # im = valid_ds[1]['img']
    # plt.imshow(im.T.squeeze().cpu().numpy(), cmap='gray')
    # plt.savefig("./diffusion_img.png")
    #
    # model = DiffusionModule("./workshop_data/config.yaml")
    # model.load_ckpt("./workshop_data/last.ckpt", ema=True)
    # model.eval().cuda().half()
    #
    # seed = 0
    # torch.manual_seed(seed)
    # import numpy as np
    # np.random.seed(seed)
    # random.seed(seed)
    #
    # # noise = model.diffusion.q_sample(im, t=torch.LongTensor([999])).unsqueeze(0)
    # # print(noise.shape)
    # noise = torch.randn(1, 1, 256, 256)
    # noise = torch.randn(1, 1, 256, 256)
    # noise = torch.randn(1, 1, 256, 256)
    # noise = torch.randn(1, 1, 256, 256)
    # cls_label = torch.tensor([2])       # 0: Normal, 1: Bacterial Pneumonia, 2: Viral Pneumonia
    #
    # model_kwargs = {"cls": torch.nn.functional.one_hot(cls_label, num_classes=3)}
    #
    # mask_p = 'test_seg_diff_old.nii.gz'
    # import nibabel as nib
    # mask_nif = nib.load(mask_p)
    # mask = torch.nn.functional.max_pool2d(torch.flip(torch.tensor(mask_nif.get_fdata()).permute(2,0,1), dims=(-1,)), kernel_size=3, stride=1, padding=1)[None, ...]
    # # plt.imshow(mask.squeeze().numpy(),cmap='gray')
    # # plt.savefig("mask_diff_test.png")
    # # exit()
    #
    # # mask = torch.zeros_like(noise)
    # # mask[0,0,40:110, 70:180]=1
    # # mask[0,0,160:225, 70:200]=1
    #
    # img = model.predict(
    #     noise,
    #     model_kwargs=model_kwargs,
    #     classifier_cond_scale=4.,
    #     inference_protocol="DDIM100",
    #     original_image=im.unsqueeze(0),
    #     mask=mask
    #     # start_denoise_step=200
    # )
    #
    # print(img[0].shape)
    #
    # plt.imshow(img[0].permute(2, 1, 0).cpu().numpy(), cmap="gray")
    # plt.savefig("./diffusion_img_new.png")
    #
    # diff_im = img[0] - im
    # print(img[0].shape)
    # print(im.shape)
    # print(diff_im.shape)
    # plt.imshow(diff_im.permute(2, 1, 0).cpu().numpy(), cmap="gray")
    # plt.savefig("./diffusion_difference_img.png")