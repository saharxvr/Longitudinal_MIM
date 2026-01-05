import os
import math
import random
from signal import signal, SIGINT

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import *
from datasets import LongitudinalMIMDataset
# from preprocessing import BatchPreprocessingImagewise
from constants import *
import matplotlib.pyplot as plt
from matplotlib import colors
from extra.vgg_losses import VGGPerceptualLoss
from piqa import SSIM, MS_SSIM
# from preprocessing import generate_single_patch_masked_images
from utils import MaskProbScheduler, generate_alpha_map
from time import time
from torchvision.transforms.v2.functional import adjust_sharpness
import kornia


def run_epoch(epoch_num, mode: str):
    assert mode in {'train', 'val', 'test'}
    if mode == 'train':
        # preprocess = train_preprocess
        dataloader = train_dataloader
        steps_per_epoch = train_steps_per_epoch
        model.train()
    elif mode == 'val':
        # preprocess = val_preprocess
        dataloader = val_dataloader
        steps_per_epoch = val_steps_per_epoch
        model.eval()
    else:
        raise 'wa??'
        # preprocess = val_preprocess
        dataloader = test_dataloader
        steps_per_epoch = test_steps_per_epoch
        model.eval()

    print_every_steps = steps_per_epoch // 40 + 2

    avg_loss = 0

    print(f'Starting {mode} epoch {epoch_num}/{epochs}')

    if mode == 'train':
        optimizer.zero_grad()
    for i, batch in tqdm(enumerate(dataloader)):
        # torch.cuda.empty_cache()

        bls, fus, gts, fu_mask = batch

        if sharpen:
            # bls = adjust_sharpness(bls, sharpness_factor=8.)
            # fus = adjust_sharpness(fus, sharpness_factor=8.)

            bls_min = bls.amin(dim=(1, 2, 3), keepdim=True)
            bls_max = bls.amax(dim=(1, 2, 3), keepdim=True)
            bls = (bls - bls_min) / (bls_max - bls_min)

            fus_min = fus.amin(dim=(1, 2, 3), keepdim=True)
            fus_max = fus.amax(dim=(1, 2, 3), keepdim=True)
            fus = (fus - fus_min) / (fus_max - fus_min)

            indic = random.random()
            if indic < 0.7:
                try:
                    clip_limit1 = random.random() * 0.35 + 0.9
                    clip_limit12 = random.random() * 0.35 + 0.9 if random.random() < 0.5 else clip_limit1
                    bls = kornia.enhance.equalize_clahe(bls, clip_limit=clip_limit1, grid_size=(8, 8))
                    fus = kornia.enhance.equalize_clahe(fus, clip_limit=clip_limit12, grid_size=(8, 8))
                except Exception as e:
                    print(f'Failed!!!!!')
                    print(type(bls))
                    print(bls.shape)
                    print(torch.min(bls))
                    print(torch.max(bls))

                    print(type(fus))
                    print(fus.shape)
                    print(torch.min(fus))
                    print(torch.max(fus))

                    print(type(gts))
                    print(gts.shape)
                    print(torch.min(gts))
                    print(torch.max(gts))
                    exit()

                bls = adjust_sharpness(bls, sharpness_factor=4.)
                fus = adjust_sharpness(fus, sharpness_factor=4.)
            elif indic < 0.96:
                factor1 = random.randint(6, 8)
                factor2 = random.randint(6, 8) if random.random() < 0.4 else factor1
                bls = adjust_sharpness(bls, sharpness_factor=factor1)
                fus = adjust_sharpness(fus, sharpness_factor=factor2)

            bls = np.clip(bls, a_min=0., a_max=1.)
            fus = np.clip(fus, a_min=0., a_max=1.)

        bls = bls.to(torch.float16).to(DEVICE)
        fus = fus.to(torch.float16).to(DEVICE)
        gts = gts.to(torch.float16).to(DEVICE)
        # fu_mask = fu_mask.bool().to(DEVICE)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # exit()
            # bls = bls.half()
            # fus = fus.half()
            # fus_gt = fus_gt.half()
            # fu_mask = fu_mask.half()

            # masked_fus, mask = generate_single_patch_masked_images(fus, patch_size=MASK_PATCH_SIZE, mask_token=mask_token)
            # outputs = model(bls, masked_fus)

            # gts = fus - fus_gt

            outputs = model(bls, fus)

            # outs = outputs * fu_mask
            # gts = (fus - fus_gt) * fu_mask
            outs = outputs

            # inv_mask = (1. - mask).bool()
            #
            # f_fus_gt = F.unfold(fus_gt, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE)
            #
            # if USE_PATCH_DEC:
            #     fus_gt_patch = torch.masked_select(f_fus_gt, inv_mask).view(BATCH_SIZE, 1, MASK_PATCH_SIZE, MASK_PATCH_SIZE)
            #     gts = fus_gt_patch
            #     outs = outputs
            # else:
            #     f_inv_masked_fus_gt = f_fus_gt * inv_mask
            #     inv_masked_fus_gt = F.fold(f_inv_masked_fus_gt, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE, output_size=(IMG_SIZE, IMG_SIZE))
            #
            #     f_outputs = F.unfold(outputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE)
            #     f_inv_masked_outputs = f_outputs * inv_mask
            #     inv_masked_outputs = F.fold(f_inv_masked_outputs, kernel_size=MASK_PATCH_SIZE, stride=MASK_PATCH_SIZE, output_size=(IMG_SIZE, IMG_SIZE))
            #
            #     gts = inv_masked_fus_gt
            #     outs = inv_masked_outputs

            if USE_L1:
                # l1_all = l1_loss(outputs_patch, fus_gt_patch) / (torch.sum(inv_mask) * (MASK_PATCH_SIZE ** 2))
                l1 = l1_loss(outs, gts)
                # if not USE_PATCH_DEC:
                #     l1 = l1 / (torch.sum(inv_mask) * (MASK_PATCH_SIZE ** 2))
                loss = LAMBDA_L1_ALL * l1

                losses_dict['l1'] = l1.detach().item()
                t_losses_dict['l1'] += l1.detach().item()

            if USE_L2:
                l2 = l2_loss(outs * 128., gts * 128.)
                loss += LAMBDA_L2 * l2

                losses_dict['l2'] = l2.detach().item()
                t_losses_dict['l2'] += l2.detach().item()

            if USE_PERC_STYLE:
                perc_loss, style_loss = perc_style_loss(outs, gts, style_layers=(2, 3, 4))
                loss += LAMBDA_P * perc_loss + LAMBDA_S * style_loss

                losses_dict['perc'] = perc_loss.detach().item()
                t_losses_dict['perc'] += perc_loss.detach().item()
                losses_dict['style'] = style_loss.detach().item()
                t_losses_dict['style'] += style_loss.detach().item()

            if USE_SSIM:
                ms_ssim_loss = 1 - ssim(outs, gts)
                # if not USE_PATCH_DEC:
                #     rat = torch.numel(inv_mask) / torch.sum(inv_mask)
                #     ms_ssim_loss = ms_ssim_loss * rat
                loss += LAMBDA_SSIM * ms_ssim_loss

                losses_dict['ssim'] = ms_ssim_loss.detach().item()
                t_losses_dict['ssim'] += ms_ssim_loss.detach().item()

            loss = loss / UPDATE_EVERY_BATCHES

        if mode == 'train':
            # loss.backward()
            scaler.scale(loss).backward()

            if (i + 1) % UPDATE_EVERY_BATCHES == 0 or i + 1 == steps_per_epoch:
                # optimizer.step()
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                # NaN gradients due to large scaling factor cause optimizer.step() to get skipped. Should call scheduler.step() then
                skip_lr_sched = (scale > scaler.get_scale())

                optimizer.zero_grad()
                if not skip_lr_sched:
                    scheduler.step()

        with torch.no_grad():
            avg_loss += loss.item()

            # if i % 100 == 0:
            #     checks = weights_check(model)
            #     print(f'Batch num {i} #########')
            #     print(f"Max abs weights is {checks[0]}")
            #     print(f"Is there a nan weight? {checks[1]}")

            if i % print_every_steps == 0 or (i < 35 and epoch_num == 1):
                print(f'Epoch num {epoch_num}, Batch num {i + 1}')
                print(f'Cur {mode} batch loss = {loss.item()}')
                print(f'Losses dict: {losses_dict}')
                print(f'Cur learning rate = {scheduler.get_last_lr()}')
                det_bls = bls[0, 0, :, :].detach().cpu()
                det_fus = fus[0, 0, :, :].detach().cpu()
                det_gts = gts[0, 0, :, :].detach().cpu()
                det_outs = outs[0, 0, :, :].detach().cpu()
                # det_outs = outs[0, 0, :, :].detach().cpu() * fu_mask[0, 0].cpu()
                axs[0, 0].imshow(det_bls.numpy(), cmap='gray')
                axs[0, 1].imshow(det_fus.numpy(), cmap='gray')

                det_outs[det_outs.abs() > 0.6] = 0.

                axs[1, 0].imshow(det_fus, cmap='gray')
                divnorm1 = colors.TwoSlopeNorm(vmin=min(torch.min(det_gts).item(), -0.01), vcenter=0., vmax=max(torch.max(det_gts).item(), 0.01))
                alpha_map1 = generate_alpha_map(det_gts).numpy().astype(float)
                imm1 = axs[1, 0].imshow(det_gts.numpy(), alpha=alpha_map1, cmap=differential_grad, norm=divnorm1)

                axs[1, 1].imshow(det_fus.numpy(), cmap='gray')
                divnorm2 = colors.TwoSlopeNorm(vmin=min(torch.min(det_outs).item(), -0.01), vcenter=0., vmax=max(torch.max(det_outs).item(), 0.01))
                alpha_map2 = generate_alpha_map(det_outs).numpy().astype(float)
                imm2 = axs[1, 1].imshow(det_outs.numpy(), alpha=alpha_map2, cmap=differential_grad, norm=divnorm2)

                cbar1 = axs[1, 0].figure.colorbar(imm1, ax=axs[1, 0])
                cbar2 = axs[1, 1].figure.colorbar(imm2, ax=axs[1, 1])
                plt.savefig(f'{plots_folder}/{mode}_epoch{epoch_num}_batch_{i}')
                axs[0, 0].clear()
                axs[0, 1].clear()
                cbar1.remove()
                cbar2.remove()
                axs[1, 0].clear()
                axs[1, 1].clear()

    with torch.no_grad():
        avg_loss = avg_loss / steps_per_epoch
        avg_losses_dict = {l: v / steps_per_epoch for l, v in t_losses_dict.items()}
        print(f'Avg loss for {mode} epoch {epoch_num}: {avg_loss}')
        print(f'Avg losses dict for {mode} epoch {epoch_num}: {avg_losses_dict}')

        for l in t_losses_dict:
            t_losses_dict[l] = 0.

    return avg_loss


if __name__ == '__main__':
    def get_cur_state_str(ep):
        # return f'id31_{f"Epoch{ep}_" if ep >= 0 else ""}Longitudinal_DeviceInvariant_DRRs_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_{"Perc_" if USE_PERC_STYLE else ""}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_Decoder6_{"Eff_ViT_" if "Eff" in LOAD_PATH else ""}{"MaskToken_" if USE_MASK_TOKEN else ""}{"MS-SSIM_" if USE_SSIM else ""}{"Adv2x_" if USE_GAN else ""}{"L1" if USE_L1 else ""}{"L2" if USE_L2 else ""}{"Fourier_masked" if USE_FOURIER else ""}{"_PosEmb" if USE_POS_EMBED else ""}_{"BN" if USE_BN else "GN"}'
        # return f'id45_{f"Epoch{ep}_" if ep >= 0 else ""}Longitudinal_AllEntities_DEVICES_FT_Cons_Sharpen_Dropout_ExtendedConvNet_{"Perc_" if USE_PERC_STYLE else ""}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_Decoder6_{"Eff_ViT_" if "Eff" in LOAD_PATH else ""}{"MaskToken_" if USE_MASK_TOKEN else ""}{"MS-SSIM_" if USE_SSIM else ""}{"Adv2x_" if USE_GAN else ""}{"L1" if USE_L1 else ""}{"L2" if USE_L2 else ""}{"Fourier_masked" if USE_FOURIER else ""}{"_PosEmb" if USE_POS_EMBED else ""}_{"BN" if USE_BN else "GN"}'
        return f'sahar_model'

        # return f'id18_FTid16_{f"Epoch{ep}_" if ep >= 0 else ""}Longitudinal_Devices2_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_{"Perc_" if USE_PERC_STYLE else ""}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_Decoder6_{"Eff_ViT_" if "Eff" in LOAD_PATH else ""}{"MaskToken_" if USE_MASK_TOKEN else ""}{"MS-SSIM_" if USE_SSIM else ""}{"Adv2x_" if USE_GAN else ""}{"L1" if USE_L1 else ""}{"L2" if USE_L2 else ""}{"Fourier_masked" if USE_FOURIER else ""}{"_PosEmb" if USE_POS_EMBED else ""}_{"BN" if USE_BN else "GN"}'

    def interrupt_handler(*args):
        inp = input("Keyboard interrupt detected. Do you want to save the current training state before exiting? yes/no\n")
        inp_up = inp.upper()
        if inp_up == 'YES' or inp_up == 'Y':
            print("Saving current state and existing")
            save_checkpoint = {
                'epoch': epoch,
                'model_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'lr_scheduler': scheduler,
                'mask_prob_scheduler_step': mask_prob_scheduler.get_step(),
                'best_val_loss': best_val_loss
            }
            torch.save(save_checkpoint, save_folder + f'/CheckpointInterrupt_{get_cur_state_str(epoch)}')
            exit()
        elif inp_up == 'NO' or inp_up == 'N':
            print("Exiting without saving")
            exit()
        else:
            print("Given input is not 'yes' or 'no'. Continuing run.")

    # differential_grad = colors.LinearSegmentedColormap.from_list('differential_grad', (
    #     # Edit this gradient at https://eltos.github.io/gradient/#0:440A57-12.5:5628A5-25:1256D4-37.5:119AB9-49:07C8C3-50:FFFFFF-51:00A64E-62.5:3CC647-75:60F132-87.5:A8FF3E-100:ECF800
    #     (0.000, (0.267, 0.039, 0.341)),
    #     (0.125, (0.337, 0.157, 0.647)),
    #     (0.250, (0.071, 0.337, 0.831)),
    #     (0.375, (0.067, 0.604, 0.725)),
    #     (0.490, (0.027, 0.784, 0.765)),
    #     (0.500, (1.000, 1.000, 1.000)),
    #     (0.510, (0.000, 0.651, 0.306)),
    #     (0.625, (0.235, 0.776, 0.278)),
    #     (0.750, (0.376, 0.945, 0.196)),
    #     (0.875, (0.659, 1.000, 0.243)),
    #     (1.000, (0.925, 0.973, 0.000))
    # ))
    differential_grad = colors.LinearSegmentedColormap.from_list('differential_grad', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:440A57-12.5:5628A5-25:1256D4-37.5:119AB9-49:07C8C3-50:FFFFFF-51:00A64E-62.5:3CC647-75:60F132-87.5:A8FF3E-100:ECF800
        (0.000, (0.016, 1.000, 0.000)),
        (0.175, (0.302, 1.000, 0.290)),
        (0.350, (0.639, 1.000, 0.631)),
        (0.446, (0.800, 1.000, 0.792)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.554, (1.000, 0.816, 0.816)),
        (0.675, (1.000, 0.584, 0.588)),
        (0.825, (1.000, 0.322, 0.310)),
        (1.000, (1.000, 0.000, 0.008))
    ))

    signal(SIGINT, interrupt_handler)
    save_folder = '/cs/usr/sahar_aharon/Desktop/sahar_aharon/Longitudinal_MIM'
    plots_folder = f'/cs/usr/sahar_aharon/Desktop/sahar_aharon/Longitudinal_MIM/training/{get_cur_state_str(-1).split(".")[0]}'

    os.makedirs(plots_folder, exist_ok=True)

    print(f'Losses used:\nL2: {USE_L2}\nFourier: {USE_FOURIER}\nPerceptual + Style: {USE_PERC_STYLE}\nGAN: {USE_GAN}\nSSIM: {USE_SSIM}')
    print(f'USE_BN={USE_BN}\nUSE_MASK_TOKEN={USE_MASK_TOKEN}\nUSE_POS_EMBED={USE_POS_EMBED}\nINIT_WEIGHTS={INIT_WEIGHTS}\nBATCH_SIZE={BATCH_SIZE}\nUPDATE_EVERY_BATCHES={UPDATE_EVERY_BATCHES}')
    print(f'WEIGHT_DECAY={WEIGHT_DECAY}\nMAX_LR={MAX_LR}')
    print(f"LOAD STATE FROM {LONGITUDINAL_LOAD_PATH}" if LOAD_PATH else "STATE NOT LOADED")

    # invariance = 'devices'
    invariance = None

    entity_dirs = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/images', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images']

    inpaint_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs'
    inpaint_dirs = [f'{inpaint_dir}/{d}' for d in os.listdir(inpaint_dir)]
    inpaint_dirs.remove('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/unrelated_healthy')
    # inpaint_dirs = [
        # '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/unrelated_healthy',
        # '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/inpainted_healthy'
    # ]
    # print(f'Inpaint_dirs = \n{inpaint_dirs}')

    # DRR_single_dirs = []
    DRR_single_dirs = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/DRRs']
    # DRR_pair_dirs = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_train',
    #                  '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_test_old/angles_15_25_25',
    #                  '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_test_old/angles_5_10_10',
    #                  '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_train/GE MEDICAL SYSTEMS',
    #                  '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_train/Philips',
    #                  '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_train/SIEMENS'
    #                  ]
    # DRR_pair_dirs = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/final/train', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/consolidation_training']

    DRR_pair_dirs = ['/cs/usr/sahar_aharon/Desktop/sahar_aharon/co_devices']

    DRR_single_dirs = []
    entity_dirs = []
    inpaint_dirs = []
    # DRR_single_dirs = []

    print(f'entity_dirs = \n{entity_dirs}')
    print(f'inpaint_dirs = \n{inpaint_dirs}')
    print(f'DRR_single_dirs = \n{DRR_single_dirs}')
    print(f'DRR_pair_dirs = \n{DRR_pair_dirs}')

    train_dataset = LongitudinalMIMDataset(entity_dirs=entity_dirs, inpaint_dirs=inpaint_dirs, DRR_single_dirs=DRR_single_dirs, DRR_pair_dirs=DRR_pair_dirs, invariance=invariance)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    len_train_ds = len(train_dataset)

    # val_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[], invariance=invariance)
    val_dataset = LongitudinalMIMDataset([], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/GE MEDICAL SYSTEMS/angles_20_20_20'], invariance=invariance)
    # val_dataset = LongitudinalMIMDataset([], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/GE MEDICAL SYSTEMS/angles_20_20_20', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/Philips/angles_20_20_20', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/SIEMENS/angles_20_20_20'], invariance=invariance)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
    len_val_ds = len(val_dataset)

    # test_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[], invariance=invariance)
    test_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[], invariance=invariance)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    len_test_ds = len(test_dataset)

    # model = LongitudinalMIMModel(use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=USE_PATCH_DEC, use_pos_embed=USE_POS_EMBED).to(DEVICE)
    model = LongitudinalMIMModelBig(use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=USE_PATCH_DEC, use_pos_embed=USE_POS_EMBED,
                                    use_technical_bottleneck=False).to(DEVICE)
    # model = LongitudinalMIMModelBigTransformer(use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=USE_PATCH_DEC, use_pos_embed=USE_POS_EMBED).to(DEVICE)
    # model = LongitudinalMIMModelTest().to(DEVICE)
    # train_preprocess = BatchPreprocessingImagewise(clip_limit=2.2, clahe_prob=1., rand_crop_prob=0., blur_prob=0.)
    # val_preprocess = BatchPreprocessingImagewise(clip_limit=2.2, clahe_prob=1., rand_crop_prob=0., blur_prob=0.)

    sharpen = True
    print(f'{"Sharpening" if sharpen else "Not sharpening"}')

    start_epoch = 1
    optimizer = optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=MAX_LR)
    # optimizer = optim.AdamW(model.technical_bottleneck.parameters(), weight_decay=WEIGHT_DECAY, lr=MAX_LR)
    epochs = LONGITUDINAL_MIM_EPOCHS
    train_steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    val_steps_per_epoch = math.ceil(len(val_dataset) / BATCH_SIZE)
    test_steps_per_epoch = math.ceil(len(test_dataset) / BATCH_SIZE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=epochs,
                                              steps_per_epoch=math.ceil(train_steps_per_epoch / UPDATE_EVERY_BATCHES),
                                              pct_start=0.15, anneal_strategy='cos')
    mask_prob_scheduler = MaskProbScheduler(epochs=epochs, steps_per_epoch=train_steps_per_epoch, init_val=INIT_MASK_PROB, max_val=MAX_MASK_PROB, end_val=END_MASK_PROB)
    best_val_loss = float('inf')

    scaler = torch.cuda.amp.GradScaler()

    if 'Checkpoint' in LONGITUDINAL_LOAD_PATH:
        checkpoint_dict = torch.load(LONGITUDINAL_LOAD_PATH)
        strict = False
        print(f"Loading model weights (strict={strict})")
        model.load_state_dict(checkpoint_dict['model_dict'], strict=strict)

        load_others = input('Load optimizer, scheduler, and starting epoch from checkpoint?')
        if load_others.upper() == 'YES':
            print("Fully loading checkpoint")
            optimizer.load_state_dict(checkpoint_dict['optimizer_dict'])
            scheduler = checkpoint_dict['lr_scheduler']
            start_epoch = checkpoint_dict['epoch'] if 'Interrupt' in LONGITUDINAL_LOAD_PATH else checkpoint_dict['epoch'] + 1
        else:
            print("Using default variables")
        # mask_prob_scheduler.set_step(checkpoint_dict['mask_prob_scheduler_step'])
        # best_val_loss = checkpoint_dict['best_val_loss']
        print("Finished loading")
    elif LONGITUDINAL_LOAD_PATH:
        # model.enc.load_state_dict(torch.load(LONGITUDINAL_LOAD_PATH))
        model.encoder.load_state_dict(torch.load(LONGITUDINAL_LOAD_PATH), strict=True)

    # freeze_and_unfreeze([model.encoder, model.encoded_bn, model.encoded_bn2, model.diff_processing, model.diff_processing2, model.dropout, model.decoder], [])

    mask_prob = mask_prob_scheduler.calc_cur_val()
    mask_token = model.get_mask_token()

    print(f'Len of train dataset: {len_train_ds}')
    print(f'Train steps per epoch: {train_steps_per_epoch}\n')
    print(f'Len of val dataset: {len_val_ds}')
    print(f'Val steps per epoch: {val_steps_per_epoch}\n')
    print(f'Len of test dataset: {len_test_ds}')
    print(f'Test steps per epoch: {test_steps_per_epoch}\n')

    t_losses_dict = {}
    losses_dict = {}

    # if USE_PATCH_DEC:
    #     l1_loss = nn.L1Loss()
    # else:
    #     l1_loss = nn.L1Loss(reduction='sum')

    if USE_L1:
        l1_loss = nn.L1Loss()
        t_losses_dict['l1'] = 0.
        losses_dict['l1'] = 0.

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

    if USE_SSIM:
        ssim = MS_SSIM(window_size=11, n_channels=1, k2=0.1).to(DEVICE)
        # ssim = SSIM(window_size=11, n_channels=1).to(DEVICE)

        t_losses_dict['ssim'] = 0.
        losses_dict['ssim'] = 0.

    train_losses = []
    val_losses = []
    test_losses = []

    fig, axs = plt.subplots(2, 2)

    torch.cuda.empty_cache()

    for epoch in range(start_epoch, epochs + 1):
        avg_train_loss = run_epoch(epoch, mode='train')
        print(f'Saving model for epoch {epoch}')
        checkpoint = {
            'epoch': epoch,
            'model_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'lr_scheduler': scheduler,
            'mask_prob_scheduler_step': mask_prob_scheduler.get_step(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, f'{save_folder}/Checkpoint_{get_cur_state_str(epoch)}.pt')
        # continue
        with torch.no_grad():
            avg_val_loss = run_epoch(epoch, mode='val')
            if avg_val_loss < best_val_loss:
                print(f'Epoch {epoch}, avg val los = {avg_val_loss}')
                print(f'Found best val loss: {avg_val_loss}, on epoch {epoch}')
                best_val_loss = avg_val_loss
                best_epoch = epoch
        #     avg_test_loss = run_epoch(epoch, mode='test')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        # test_losses.append(avg_test_loss)

        # print(f'Saving model for epoch {epoch}')
        # checkpoint = {
        #     'epoch': epoch,
        #     'model_dict': model.state_dict(),
        #     'optimizer_dict': optimizer.state_dict(),
        #     'lr_scheduler': scheduler,
        #     'mask_prob_scheduler_step': mask_prob_scheduler.get_step(),
        #     'best_val_loss': best_val_loss
        # }
        # torch.save(checkpoint, f'{save_folder}/Checkpoint_{get_cur_state_str(epoch)}.pt')