import copy
import math
import os
from signal import signal, SIGINT

import torch
# import torch
from tqdm import tqdm
import torch.optim as optim
# import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocessing import BatchPreprocessingImagewise
from datasets import PatchReconstructionDataset, NoFindingDataset, SmallNoFindingDataset, ExpertNoFindingDataset
from constants import *
from models import *
from utils import MaskProbScheduler, masked_patches_l1_loss, fourier, masked_patches_fourier_loss, masked_patches_SSIM_loss, masked_patches_GAN_loss
from preprocessing import generate_masked_images, generate_single_patch_masked_images, generate_checkerboard_masked_images
from torch.utils.checkpoint import checkpoint
from vgg_losses import VGGPerceptualLoss
from piqa import SSIM, MS_SSIM


def get_masked(inps, mask_pr, mask_tk, epo, t_epos, mode=MASK_MODE):
    if mode == 'reg':
        return generate_masked_images(inps.to(DEVICE), mask_probability=mask_pr, mask_token=mask_tk)
    elif mode == 'single':
        return generate_single_patch_masked_images(inps.to(DEVICE), mask_token=mask_tk)
    elif mode == 'checkerboard':
        if epo < t_epos - 1:
            return generate_masked_images(inps.to(DEVICE), mask_probability=mask_pr, mask_token=mask_tk)
        else:
            return generate_checkerboard_masked_images(inps.to(DEVICE), mask_token=mask_tk)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    def interrupt_handler(*args):
        inp = input("Keyboard interrupt detected. Do you want to save the current training state? yes/no\n")
        if inp.upper() == 'YES':
            print("Saving current state and existing")
            save_checkpoint = {
                'epoch': epoch,
                'model_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'lr_scheduler': scheduler,
                'mask_prob_scheduler_step': mask_prob_scheduler.get_step(),
                'best_val_loss': best_val_loss
            }
            torch.save(save_checkpoint, PROJECT_FOLDER + f'saved_models/MIM/CheckpointInterrupt_id0_MaskGrad_Sig5_Expert_{"Perc_" if USE_PERC_STYLE else ""}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_NoFinding_Decoder5_Augs_{"Eff_ViT_" if "Eff" in LOAD_PATH else ""}_Epoch{epoch}_{"MaskToken_" if USE_MASK_TOKEN else ""}{"MS-SSIM_" if USE_SSIM else ""}{"Adv2x_" if USE_GAN else ""}L1{"L2" if USE_L2 else ""}{"Fourier_masked" if USE_FOURIER else ""}{"_PosEmb" if USE_POS_EMBED else ""}_{"BN" if USE_BN else "GN"}.pt')
            exit()
        else:
            print("Exiting without saving")
            exit()

    signal(SIGINT, interrupt_handler)

    plots_folder = PROJECT_FOLDER + f"plots/MIM/id0_MaskGrad_Sig5_Expert_{'Perc_' if USE_PERC_STYLE else ''}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_NoFinding_Decoder5_Augs_{'NoPretraining_' if not LOAD_PATH else ''}{'MaskToken_' if USE_MASK_TOKEN else ''}{'MS-SSIM_' if USE_SSIM else ''}{'Adv2x_' if USE_GAN else ''}L1{'L2' if USE_L2 else ''}{'Fourier_masked' if USE_FOURIER else ''}{'_PosEmb' if USE_POS_EMBED else ''}_{'BN' if USE_BN else 'GN'}"
    os.makedirs(plots_folder, exist_ok=True)

    print(f'Losses used:\nL2: {USE_L2}\nFourier: {USE_FOURIER}\nPerceptual + Style: {USE_PERC_STYLE}\nGAN: {USE_GAN}\nSSIM: {USE_SSIM}')
    print(f'USE_BN={USE_BN}\nUSE_MASK_TOKEN={USE_MASK_TOKEN}\nUSE_POS_EMBED={USE_POS_EMBED}\nINIT_WEIGHTS={INIT_WEIGHTS}\nBATCH_SIZE={BATCH_SIZE}\nUPDATE_EVERY_BATCHES={UPDATE_EVERY_BATCHES}')
    print(f'WEIGHT_DECAY={WEIGHT_DECAY}\nMAX_LR={MAX_LR}\nPatch_Size used: {MASK_PATCH_SIZE}')
    print(f"LOAD STATE FROM {LOAD_PATH}" if LOAD_PATH else "STATE NOT LOADED")
    # train_folders = [f'{MIMIC_OTHER_AP}p1{i}/' for i in (0, 1, 2, 3, 4, 5, 6, 7)]
    # train_folders.extend([CXR14_CL_TRAIN])
    # train_folders.append(MIMIC_SUPINE)
    val_folders = [f'{MIMIC_OTHER_AP}p18/']
    val_folders.extend([CXR14_CL_VAL])
    test_folders = [f'{MIMIC_OTHER_AP}p19/']
    test_folders.extend([CXR14_CL_TEST])

    # train_dataset = PatchReconstructionDataset(train_folders)
    # train_dataset = NoFindingDataset(mimic_path='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/No_Finding.csv',
    #                                  cxr14_path='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/No_Finding.csv',
    #                                  pneumonia_path=PNEUMONIA_DS_NORMAL)
    # train_dataset = SmallNoFindingDataset()
    train_dataset = ExpertNoFindingDataset('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/no_finding_train.csv', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/no_finding_test.csv', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/no_finding_only_normal.csv')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    len_train_ds = len(train_dataset)

    val_dataset = PatchReconstructionDataset(val_folders)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)
    len_val_ds = len(val_dataset)

    test_dataset = PatchReconstructionDataset(test_folders)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    len_test_ds = len(test_dataset)

    preprocess = BatchPreprocessingImagewise(clip_limit=(0.5, 3.5), use_fourier=USE_FOURIER, use_canny=USE_CANNY, sigmas=SIGMAS)
    model = MaskedReconstructionModel(use_mask_token=USE_MASK_TOKEN, dec=5, in_channels=MASKED_IN_CHANNELS, use_pos_embed=USE_POS_EMBED).to(DEVICE)

    start_epoch = 1
    optimizer = optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY)
    epochs = MASKED_RECONSTRUCTION_EPOCHS
    train_steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    val_steps_per_epoch = math.ceil(len(val_dataset) / BATCH_SIZE)
    test_steps_per_epoch = math.ceil(len(test_dataset) / BATCH_SIZE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=epochs,
                                              steps_per_epoch=math.ceil(train_steps_per_epoch / UPDATE_EVERY_BATCHES),
                                              pct_start=0.2, anneal_strategy='cos')
    mask_prob_scheduler = MaskProbScheduler(epochs=epochs, steps_per_epoch=train_steps_per_epoch, init_val=INIT_MASK_PROB, max_val=MAX_MASK_PROB, end_val=END_MASK_PROB)

    best_epoch = 0
    best_val_loss = torch.inf

    if 'Checkpoint' in LOAD_PATH:
        checkpoint_dict = torch.load(LOAD_PATH)
        model.load_state_dict(checkpoint_dict['model_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_dict'])
        scheduler = checkpoint_dict['lr_scheduler']
        mask_prob_scheduler.set_step(checkpoint_dict['mask_prob_scheduler_step'])
        best_val_loss = checkpoint_dict['best_val_loss']
        start_epoch = checkpoint_dict['epoch'] if 'Interrupt' in LOAD_PATH else checkpoint_dict['epoch'] + 1
    elif LOAD_PATH:
        model.encoder_bottleneck.load_state_dict(torch.load(LOAD_PATH))

    mask_prob = mask_prob_scheduler.calc_cur_val()
    mask_token = model.get_mask_token()

    print(f'Len of train dataset: {len_train_ds}')
    print(f'Train steps per epoch: {train_steps_per_epoch}\n')
    print(f'Len of val dataset: {len_val_ds}')
    print(f'Val steps per epoch: {val_steps_per_epoch}\n')
    print(f'Len of test dataset: {len_test_ds}')
    print(f'Test steps per epoch: {test_steps_per_epoch}\n')

    l1_loss = nn.L1Loss(reduction='sum')

    t_losses_dict = {'l1': 0., 'l1_masked': 0.}
    losses_dict = {'l1': 0., 'l1_masked': 0.}

    if USE_FOURIER:
        t_losses_dict['fourier'] = 0.
        losses_dict['fourier'] = 0.

        t_losses_dict['fourier_masked'] = 0.
        losses_dict['fourier_masked'] = 0.

    if USE_L2:
        l2_loss = nn.MSELoss()
        t_losses_dict['l2'] = 0.
        losses_dict['l2'] = 0.

    if USE_PERC_STYLE:
        perc_style_loss = VGGPerceptualLoss().to(DEVICE)
        t_losses_dict['perc'] = 0.
        t_losses_dict['style'] = 0.
        losses_dict['perc'] = 0.
        losses_dict['style'] = 0.

    if USE_GAN:
        discriminator = Discriminator().to(DEVICE)
        disc_optimizer = optim.SGD(discriminator.parameters(), weight_decay=5e-8, lr=8e-4)
        # disc_scheduler = optim.lr_scheduler.OneCycleLR(disc_optimizer, max_lr=MAX_LR, epochs=epochs,
        #                                           steps_per_epoch=math.ceil(train_steps_per_epoch / UPDATE_EVERY_BATCHES),
        #                                           pct_start=0.1, anneal_strategy='cos')
        gan_loss = nn.BCEWithLogitsLoss()
        # disc_train_dataset = NoFindingDataset(
        #     mimic_path='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/No_Finding.csv',
        #     cxr14_path='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/No_Finding.csv')
        disc_train_dataloader = train_dataloader
        disc_turn = True
        t_losses_dict['adv'] = 0.
        losses_dict['adv'] = 0.

    if USE_SSIM:
        ssim = MS_SSIM(window_size=11, n_channels=1, k2=0.1).to(DEVICE)
        # ssim = SSIM(window_size=11, n_channels=1).to(DEVICE)

        t_losses_dict['ssim'] = 0.
        # t_losses_dict['ssim2'] = 0.
        losses_dict['ssim'] = 0.
        # losses_dict['ssim2'] = 0.

    train_losses = []
    val_losses = []
    test_losses = []

    fig, axs = plt.subplots(2, 3)

    torch.cuda.empty_cache()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        cur_train_loss = 0

        if USE_GAN:
            discriminator.train()
            disc_optimizer.zero_grad()
            disc_data_iter = iter(disc_train_dataloader)

        print(f'Starting epoch {epoch}/{epochs}')
        print("Training")

        optimizer.zero_grad()
        for i, batch in tqdm(enumerate(train_dataloader)):
            # batch = resize(batch)
            torch.cuda.empty_cache()
            batch, paths = batch
            inputs, fouriers, cannys = preprocess(batch)
            inputs = inputs.to(DEVICE)
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                masked, mask = get_masked(inputs, mask_prob, mask_token, epoch, epochs, MASK_MODE)
                # masked = generate_single_patch_masked_images(inputs.to(DEVICE), patch_size=MASK_PATCH_SIZE, mask_token=mask_token)
                if cannys is not None:
                    masked = torch.cat([masked, cannys], dim=1)
                # outputs = checkpoint(model, masked, use_reentrant=False)
                outputs = model(masked)

                inv_mask = 1. - mask

                if torch.sum(inv_mask.detach()).item() == 0:
                    continue

                f_inputs = F.unfold(inputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE)
                f_inv_masked_inputs = f_inputs * inv_mask
                inv_masked_inputs = F.fold(f_inv_masked_inputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE, output_size=(IMG_SIZE, IMG_SIZE))

                f_outputs = F.unfold(outputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE)
                f_inv_masked_outputs = f_outputs * inv_mask
                inv_masked_outputs = F.fold(f_inv_masked_outputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE, output_size=(IMG_SIZE, IMG_SIZE))

                l1_all = l1_loss(inv_masked_outputs, inv_masked_inputs) / (torch.sum(inv_mask) * (MASK_PATCH_SIZE ** 2))
                loss = LAMBDA_L1_ALL * l1_all

                losses_dict['l1'] = l1_all.detach().item()
                t_losses_dict['l1'] += l1_all.detach().item()

                # l1_masked = masked_patches_l1_loss(outputs, inputs.to(DEVICE), mask.detach())
                # if l1_masked is not None:
                #     loss += LAMBDA_L1_MASKED * l1_masked
                #
                #     losses_dict['l1_masked'] = l1_masked.detach().item()
                #     t_losses_dict['l1_masked'] += l1_masked.detach().item()
                # else:
                #     losses_dict['l1_masked'] = None

                if USE_L2:
                    l2 = l2_loss(outputs, inputs.to(DEVICE))
                    loss += LAMBDA_L2 * l2

                    losses_dict['l2'] = l2.detach().item()
                    t_losses_dict['l2'] += l2.detach().item()

                if USE_FOURIER:
                    # fourier_loss = l1_loss(fourier(outputs), fouriers.to(DEVICE))
                    # loss += LAMBDA_FOURIER * fourier_loss
                    #
                    # losses_dict['fourier'] = fourier_loss.detach().item()
                    # t_losses_dict['fourier'] += fourier_loss.detach().item()

                    fourier_masked_loss = masked_patches_fourier_loss(outputs, fouriers.to(DEVICE), mask)
                    if fourier_masked_loss is not None:
                        loss += LAMBDA_FOURIER_MASKED * fourier_masked_loss

                        losses_dict['fourier_masked'] = fourier_masked_loss.detach().item()
                        t_losses_dict['fourier_masked'] += fourier_masked_loss.detach().item()

                if USE_PERC_STYLE:
                    perc_loss, style_loss = perc_style_loss(inv_masked_outputs, inv_masked_inputs)
                    loss += LAMBDA_P * perc_loss + LAMBDA_S * style_loss

                    losses_dict['perc'] = perc_loss.detach().item()
                    t_losses_dict['perc'] += perc_loss.detach().item()
                    losses_dict['style'] = style_loss.detach().item()
                    t_losses_dict['style'] += style_loss.detach().item()

                if USE_GAN and epoch >= GAN_START_EPOCH:
                    if disc_turn:
                        for _____ in range(2):
                            for ___ in range(UPDATE_EVERY_BATCHES):
                                torch.cuda.empty_cache()
                                disc_batch = next(disc_data_iter, None)
                                if disc_batch is None:
                                    disc_data_iter = iter(disc_train_dataloader)
                                    disc_batch = next(disc_data_iter)
                                disc_batch, _ = disc_batch
                                disc_inputs, _, __ = preprocess(disc_batch)
                                disc_inputs = disc_inputs.to(DEVICE)

                                if MASK_MODE == 'single':
                                    ____, disc_mask = get_masked(disc_inputs, 0., mask_tk=None, epo=0, t_epos=0, mode=MASK_MODE)
                                    inv_disc_mask = 1. - disc_mask
                                    f_disc_inputs = F.unfold(disc_inputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE)
                                    f_inv_masked_disc_inputs = f_disc_inputs * inv_disc_mask
                                    disc_inputs = F.fold(f_inv_masked_disc_inputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE, output_size=(IMG_SIZE, IMG_SIZE))

                                disc_real_outputs = discriminator(disc_inputs)
                                real_labels = torch.ones_like(disc_real_outputs, device=DEVICE)
                                disc_real_loss = gan_loss(disc_real_outputs, real_labels)

                                disc_fake_outputs = discriminator(inv_masked_outputs.detach())
                                fake_labels = torch.zeros_like(disc_fake_outputs, device=DEVICE)
                                disc_fake_loss = gan_loss(disc_fake_outputs, fake_labels)

                                disc_loss = disc_real_loss + disc_fake_loss
                                disc_loss.backward()
                            disc_optimizer.step()
                            disc_optimizer.zero_grad()
                        disc_turn = False
                        freeze_and_unfreeze([discriminator], [])
                        discriminator.eval()

                    disc_fake_outputs = discriminator(inv_masked_outputs)
                    real_labels = torch.ones_like(disc_fake_outputs, device=DEVICE)
                    adv_loss = gan_loss(disc_fake_outputs, real_labels)
                    # adv_loss = masked_patches_GAN_loss(outputs, mask, discriminator, gan_loss)
                    loss += LAMBDA_GAN * adv_loss

                    losses_dict['adv'] = adv_loss.detach().item()
                    t_losses_dict['adv'] += adv_loss.detach().item()

                if USE_SSIM:
                    rat = torch.numel(inv_mask) / torch.sum(inv_mask)

                    ms_ssim_loss = (1 - ssim(inv_masked_outputs, inv_masked_inputs)) * rat
                    loss += LAMBDA_SSIM * ms_ssim_loss

                    losses_dict['ssim'] = ms_ssim_loss.detach().item()
                    t_losses_dict['ssim'] += ms_ssim_loss.detach().item()
                    # ssim_loss = masked_patches_SSIM_loss(outputs, inputs.to(DEVICE), mask, ssim1, ssim2)
                    # if ssim_loss is not None:
                    #     ssim_loss1, ssim_loss2 = ssim_loss
                    #     loss += LAMBDA_SSIM * (ssim_loss1 + ssim_loss2)
                    #
                    #     losses_dict['ssim1'] = ssim_loss1.detach().item()
                    #     losses_dict['ssim2'] = ssim_loss2.detach().item()
                    #     t_losses_dict['ssim1'] += ssim_loss1.detach().item()
                    #     t_losses_dict['ssim2'] += ssim_loss2.detach().item()

            loss.backward()

            mask_prob = mask_prob_scheduler.step()
            if (i + 1) % UPDATE_EVERY_BATCHES == 0 or i + 1 == train_steps_per_epoch:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if USE_GAN:
                    disc_turn = True
                    freeze_and_unfreeze([], [discriminator])
                    discriminator.train()

            # if USE_GAN and ((i + 1) % (UPDATE_EVERY_BATCHES // 2) == 0 or i + 1 == train_steps_per_epoch):
            #     disc_optimizer.step()
            #     disc_optimizer.zero_grad()
                # disc_scheduler.step()

            with torch.no_grad():
                cur_train_loss += loss.item()

                # if i % 100 == 0:
                #     checks = weights_check(model)
                #     print(f'Batch num {i} #########')
                #     print(f"Max abs weights is {checks[0]}")
                #     print(f"Is there a nan weight? {checks[1]}")

                if i % 2000 == 0:
                    # if i == 0:
                    print(f'Epoch num {epoch}, Batch num {i+1}')
                    print(f'Cur train batch loss = {loss.item()}')
                    print(f"Losses vals:\n{losses_dict}")
                    print(f'Cur learning rate = {scheduler.get_last_lr()}')
                    print(f"Cur mask prob={mask_prob}")

                if i % 2000 == 0:
                    # if epoch % 5 == 0 and (i + 1) % 100 == 0:
                    print(f'Saving example images for epoch {epoch}, batch {i + 1}')
                    axs[0, 0].imshow(inputs[0, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[0, 1].imshow(masked[0, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[0, 2].imshow(outputs[0, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[1, 0].imshow(inputs[1, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[1, 1].imshow(masked[1, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[1, 2].imshow(outputs[1, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    # print(paths[0])
                    # print(paths[1])
                    axs[0, 1].set_title('/'.join(paths[0].split('/')[-3:]), fontsize=9)
                    axs[1, 1].set_title('/'.join(paths[1].split('/')[-3:]), fontsize=9)
                    plt.savefig(plots_folder + f'/TrainEpoch{epoch}Batch{i+1}.png')
                    axs[0, 0].clear()
                    axs[0, 1].clear()
                    axs[0, 2].clear()
                    axs[1, 0].clear()
                    axs[1, 1].clear()
                    axs[1, 2].clear()

        with torch.no_grad():
            model.eval()

            if USE_GAN:
                discriminator.eval()

            cur_train_loss /= train_steps_per_epoch
            t_losses_dict = {name: val / train_steps_per_epoch for (name, val) in t_losses_dict.items()}
            print(f"Avg train loss for epoch {epoch} is {cur_train_loss}")
            print(f'Average losses: {t_losses_dict}')
            t_losses_dict = {name: 0. for name in t_losses_dict}
            train_losses.append(cur_train_loss)

            # print("Running Validation")
            cur_val_loss = 0
            for i, batch in tqdm(enumerate(val_dataloader)):
                # torch.cuda.empty_cache()
                # inputs, fouriers, cannys = preprocess(batch)
                # with torch.cuda.amp.autocast():
                #     masked, mask = get_masked(inputs, mask_prob, mask_token, epoch, epochs, MASK_MODE)
                #     if cannys is not None:
                #         masked = torch.cat([masked, cannys], dim=1)
                #     outputs = model(masked)
                #
                #     loss = LAMBDA_L1_ALL * l1_loss(outputs, inputs.to(DEVICE))
                #     l1_masked = masked_patches_l1_loss(outputs, inputs.to(DEVICE), mask)
                #     if l1_masked is not None:
                #         loss += LAMBDA_L1_MASKED * l1_masked
                #
                #     if USE_L2:
                #         loss += LAMBDA_L2 * l2_loss(outputs, inputs.to(DEVICE))
                #
                #     if USE_FOURIER:
                #         loss += LAMBDA_FOURIER * l1_loss(fourier(outputs.to(torch.float32)), fouriers.to(DEVICE))
                #
                #         fourier_masked_loss = masked_patches_fourier_loss(outputs, fouriers.to(DEVICE), mask)
                #         if fourier_masked_loss is not None:
                #             loss += LAMBDA_FOURIER_MASKED * fourier_masked_loss
                #
                #     if USE_PERC_STYLE:
                #         perc_loss, style_loss = perc_style_loss(outputs, inputs.to(DEVICE))
                #         loss += LAMBDA_P * perc_loss + LAMBDA_S * style_loss
                #
                #     if USE_GAN:
                #         disc_fake_outputs = discriminator(outputs)
                #         real_labels = torch.ones_like(disc_fake_outputs, device=DEVICE)
                #         loss += LAMBDA_GAN * gan_loss(disc_fake_outputs, real_labels)
                if i == 1:
                    # if epoch % 5 == 0 and i == 1:
                    axs[0, 0].imshow(inputs[0, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[0, 1].imshow(masked[0, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[0, 2].imshow(outputs[0, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[1, 0].imshow(inputs[1, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[1, 1].imshow(masked[1, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    axs[1, 2].imshow(outputs[1, 0, :, :].to('cpu').numpy().T, cmap='gray')
                    plt.savefig(plots_folder + f'/ValEpoch{epoch}.png')
                    axs[0, 0].clear()
                    axs[0, 1].clear()
                    axs[0, 2].clear()
                    axs[1, 0].clear()
                    axs[1, 1].clear()
                    axs[1, 2].clear()

                    break
                # elif i == 1:
                #     break

                cur_val_loss += loss.item()

            cur_val_loss /= val_steps_per_epoch
            val_losses.append(cur_val_loss)
            # print(f"Avg val loss for epoch {epoch} is {cur_val_loss}")

            # if cur_val_loss < best_val_loss or epoch == epochs:
            if True:
                # if epoch % 20 == 0:
                print(f"Saving model on epoch {epoch} with validation loss {cur_val_loss}")
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_epoch = epoch
                checkpoint = {
                    'epoch': epoch,
                    'model_dict': model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'lr_scheduler': scheduler,
                    'mask_prob_scheduler_step': mask_prob_scheduler.get_step(),
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, PROJECT_FOLDER + f'saved_models/MIM/Checkpoint_id0_MaskGrad_Sig5_Expert_{"Perc_" if USE_PERC_STYLE else ""}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_NoFinding_Decoder5_Augs_{"Eff_ViT_" if "Eff" in LOAD_PATH else ""}Epoch{epoch}_{"MaskToken_" if USE_MASK_TOKEN else ""}{"MS-SSIM_" if USE_SSIM else ""}{"Adv2x_" if USE_GAN else ""}L1{"L2" if USE_L2 else ""}{"Fourier_masked" if USE_FOURIER else ""}{"_PosEmb" if USE_POS_EMBED else ""}_{"BN" if USE_BN else "GN"}.pt')

            continue
            print("Running Test")
            cur_test_loss = 0
            for batch in tqdm(test_dataloader):
                torch.cuda.empty_cache()
                inputs, fouriers, cannys = preprocess(batch)
                with torch.cuda.amp.autocast():
                    masked, mask = get_masked(inputs, mask_prob, mask_token, epoch, epochs, MASK_MODE)
                    if cannys is not None:
                        masked = torch.cat([masked, cannys], dim=1)
                    outputs = model(masked)

                    loss = LAMBDA_L1_ALL + l1_loss(outputs, inputs.to(DEVICE))
                    l1_masked = masked_patches_l1_loss(outputs, inputs.to(DEVICE), mask)
                    if l1_masked is not None:
                        loss += LAMBDA_L1_MASKED * l1_masked

                    if USE_L2:
                        loss += LAMBDA_L2 * l2_loss(outputs, inputs.to(DEVICE))

                    if USE_FOURIER:
                        loss += LAMBDA_FOURIER * l1_loss(fourier(outputs.to(torch.float32)), fouriers.to(DEVICE))

                        fourier_masked_loss = masked_patches_fourier_loss(outputs, fouriers.to(DEVICE), mask)
                        if fourier_masked_loss is not None:
                            loss += LAMBDA_FOURIER_MASKED * fourier_masked_loss

                    if USE_PERC_STYLE:
                        perc_loss, style_loss = perc_style_loss(outputs, inputs.to(DEVICE))
                        loss += LAMBDA_P * perc_loss + LAMBDA_S * style_loss

                    if USE_GAN:
                        disc_fake_outputs = discriminator(outputs)
                        real_labels = torch.ones_like(disc_fake_outputs, device=DEVICE)
                        loss += LAMBDA_GAN * gan_loss(disc_fake_outputs, real_labels)

                cur_test_loss += loss.item()

            cur_test_loss /= test_steps_per_epoch
            test_losses.append(cur_test_loss)
            print(f"Avg test loss for epoch {epoch} is {cur_test_loss}")

    print(f'train losses\n{train_losses}')
    print()
    print(f'val losses\n{val_losses}')
    print()
    print(f'test_losses\n{test_losses}')

    plt.clf()
    plt.plot(train_losses)
    plt.title("Train loss/Epoch")
    plt.savefig(plots_folder + "/TrainLoss.png")
    plt.clf()
    plt.plot(val_losses)
    plt.title("Val loss/Epoch")
    plt.savefig(plots_folder +"/ValLoss.png")
    plt.clf()
    plt.plot(test_losses)
    plt.title("Test loss/Improved Epoch")
    plt.savefig(plots_folder + "/TestLoss.png")
