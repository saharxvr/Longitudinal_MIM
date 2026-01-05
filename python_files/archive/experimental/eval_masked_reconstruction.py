import math
import torch
import torch.nn as nn
from constants import *
from datasets import PatchReconstructionDataset
from torch.utils.data import DataLoader
from preprocessing import BatchPreprocessing, generate_masked_images
from models import *
from tqdm import tqdm
import torch.fft as fft
from piqa.psnr import PSNR
from piqa.ssim import SSIM
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    print("Running eval_masked_reconstruction.py")
    # print(f"LOAD STATE FROM {LOAD_PATH}" if LOAD_PATH else "STATE NOT LOADED")
    test_folders = [f'{MIMIC_OTHER_AP}p18/', f'{MIMIC_OTHER_AP}p19/']
    test_dataset = PatchReconstructionDataset(test_folders)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    len_test_ds = len(test_dataset)
    test_steps_per_epoch = math.ceil(len(test_dataset) / BATCH_SIZE)

    print(f"Evaluation steps: {test_steps_per_epoch}")

    preprocess = BatchPreprocessing(use_fourier=USE_FOURIER)

    mask_probs = [0.0, 0.2, 0.4, 0.8, 0.9]

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    psnr = PSNR().to(DEVICE)
    ssim = SSIM(n_channels=1).to(DEVICE)

    path_prefix = PROJECT_FOLDER + 'saved_models/MIM/'
    model_paths = [
        'Checkpoint_Eff_ViT_Epoch5_Decoder2_L1L2Fourier_GN.pt',
        'Checkpoint_Eff_ViT_Epoch5_Decoder2_MaskToken_L1L2Fourier_GN.pt',
        'Checkpoint_Epoch5_Decoder2_L1L2Fourier_GN.pt',
        'Checkpoint_MaskProb05Eff_ViT_Epoch5_Decoder2_L1L2Fourier_GN.pt',
        'Checkpoint_MaskProbSlope05Eff_ViT_Epoch5_Decoder2_L1L2Fourier_GN.pt'
    ]
    all_losses = [[] for _ in range(len(model_paths))]
    all_psnrs = [[] for _ in range(len(model_paths))]
    all_ssims = [[] for _ in range(len(model_paths))]

    plots_folder = PROJECT_FOLDER + f"plots/MIM/Evaluation"

    for path in model_paths:
        print(f"Running for model {path}")
        LOAD_PATH = path_prefix + path
        USE_MASK_TOKEN = 'MaskToken' in LOAD_PATH
        model = MaskedReconstructionModel(use_mask_token=USE_MASK_TOKEN).to(DEVICE)
        if 'Checkpoint' in LOAD_PATH:
            checkpoint_dict = torch.load(LOAD_PATH)
            model.load_state_dict(checkpoint_dict['model_dict'])
        elif LOAD_PATH:
            model.encoder_bottleneck.load_state_dict(torch.load(LOAD_PATH))

        mask_token = model.get_mask_token()

        fig, axs = plt.subplots(3, 3)

        model_folder = PROJECT_FOLDER + f"plots/MIM/Evaluation_{'MaskProbSlope05_' if 'MaskProbSlope05' in LOAD_PATH else ''}{'MaskProb05_' if 'MaskProb05' in LOAD_PATH else ''}{'NoPretraining_' if 'Eff' not in LOAD_PATH else ''}{'MaskToken_' if USE_MASK_TOKEN else ''}L1{'L2' if USE_L2 else ''}{'Fourier' if USE_FOURIER else ''}_{'BN' if USE_BN else 'GN'}"

        os.makedirs(model_folder, exist_ok=True)

        with torch.no_grad():
            model.eval()

            for k, mask_prob in enumerate(mask_probs):
                print(f"Running with MaskProb={mask_prob}")
                mean_loss = 0
                mean_psnr = 0
                mean_ssim = 0
                for i, batch in tqdm(enumerate(test_dataloader)):
                    inputs, fouriers = preprocess(batch)
                    masked = generate_masked_images(inputs.to(DEVICE), mask_probability=mask_prob, mask_token=mask_token)

                    outputs = model(masked)

                    loss = l1_loss(outputs, inputs.to(DEVICE))

                    if USE_L2:
                        loss += 0.1 * l2_loss(outputs, inputs.to(DEVICE))

                    if USE_FOURIER:
                        loss += 0.1 * l1_loss(fft.rfft2(outputs), fouriers.to(DEVICE))

                    b_psnr = psnr(outputs, inputs.to(torch.float32).to(DEVICE))
                    b_ssim = ssim(outputs, inputs.to(torch.float32).to(DEVICE))

                    mean_loss += loss.item()
                    mean_psnr += b_psnr.item()
                    mean_ssim += b_ssim.item()

                    if i == 0:
                        for j in range(3):
                            axs[j, 0].imshow(inputs[j, 0, :, :].to('cpu').numpy().T, cmap='gray')
                            axs[j, 1].imshow(masked[j, 0, :, :].to('cpu').numpy().T, cmap='gray')
                            axs[j, 2].imshow(outputs[j, 0, :, :].to('cpu').numpy().T, cmap='gray')
                        plt.savefig(model_folder + f'/MaskProb{mask_prob}.png')
                        for j in range(3):
                            axs[j, 0].clear()
                            axs[j, 1].clear()
                            axs[j, 2].clear()

                mean_loss /= test_steps_per_epoch
                mean_psnr /= test_steps_per_epoch
                mean_ssim /= test_steps_per_epoch

                print(f'Model = {path}')
                print(f"Measures for mask prob = {mask_prob}")
                print(f'mean_loss = {mean_loss}\nmean_psnr = {mean_psnr}\nmean_ssim = {mean_ssim}')

                all_losses[k].append(mean_loss)
                all_psnrs[k].append(mean_psnr)
                all_ssims[k].append(mean_ssim)

    model_idcs = [i + 1 for i in range(len(model_paths))]
    plt.clf()
    for i, prob in enumerate(mask_probs):
        plt.bar(model_idcs, all_losses[i])
        plt.title(f"Loss. MaskProb={prob}")
        plt.savefig(plots_folder + f'/losses_maskprob={prob}.png')
        plt.clf()
        plt.bar(model_idcs, all_psnrs[i])
        plt.title(f"PSNR. MaskProb={prob}")
        plt.savefig(plots_folder + f'/psnrs_maskprob={prob}.png')
        plt.clf()
        plt.bar(model_idcs, all_ssims[i])
        plt.title(f"SSIM. MaskProb={prob}")
        plt.savefig(plots_folder + f'/ssims_maskprob={prob}.png')
        plt.clf()
