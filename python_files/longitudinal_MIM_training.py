"""
Longitudinal MIM Training Script.

This is the main training script for the Longitudinal CXR change detection model.

Usage:
    python longitudinal_MIM_training.py

Configuration:
    All hyperparameters are imported from constants.py:
    - BATCH_SIZE, UPDATE_EVERY_BATCHES: Batch and gradient accumulation
    - MAX_LR, WEIGHT_DECAY: Optimizer settings
    - USE_L1, USE_L2, USE_SSIM, USE_PERC_STYLE: Loss function flags
    - LONGITUDINAL_MIM_EPOCHS: Number of training epochs

Data Sources:
    Configure in main() section:
    - entity_dirs: CXR + segmentation directories
    - inpaint_dirs: Inpainted pairs
    - DRR_single_dirs: Single DRR variations
    - DRR_pair_dirs: Synthetic DRR pairs (main source)

Output:
    - Checkpoints saved to save_folder
    - Training plots saved to plots_folder
    - Loss curves saved as loss_curves.png
    - Training parameters saved as training_params.json
"""

import os
import math
import random
import json
import gc
import psutil
from datetime import datetime
from signal import signal, SIGINT

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import *
from datasets import LongitudinalMIMDataset
from constants import *
import matplotlib.pyplot as plt
from matplotlib import colors
from losses.vgg_losses import VGGPerceptualLoss
from piqa import SSIM, MS_SSIM
from utils import MaskProbScheduler, generate_alpha_map
from time import time


# =============================================================================
# MEMORY MANAGEMENT UTILITIES
# =============================================================================

def get_memory_usage_gb():
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def log_memory(label=""):
    """Log current memory usage with optional label."""
    mem_gb = get_memory_usage_gb()
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"[Memory] {label}: RAM={mem_gb:.2f}GB, GPU={gpu_mem:.2f}GB")
    return mem_gb


def cleanup_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_memory_and_cleanup(threshold_gb=20.0, label=""):
    """
    Check memory usage and perform cleanup if above threshold.
    
    Args:
        threshold_gb: RAM threshold in GB to trigger cleanup.
        label: Label for logging.
    Returns:
        bool: True if cleanup was performed.
    """
    mem_gb = get_memory_usage_gb()
    if mem_gb > threshold_gb:
        print(f"[Memory Warning] {label}: {mem_gb:.2f}GB > {threshold_gb}GB threshold")
        cleanup_memory()
        new_mem = get_memory_usage_gb()
        print(f"[Memory] After cleanup: {new_mem:.2f}GB (freed {mem_gb - new_mem:.2f}GB)")
        return True
    return False
from torchvision.transforms.v2.functional import adjust_sharpness
import kornia


def save_training_params(params_dict, save_path):
    """
    Save all training parameters to a JSON file.
    
    Args:
        params_dict: Dictionary containing all training parameters
        save_path: Path to save the JSON file
    """
    # Convert non-serializable types to strings
    serializable_dict = {}
    for key, value in params_dict.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable_dict[key] = value
        elif isinstance(value, torch.device):
            serializable_dict[key] = str(value)
        else:
            serializable_dict[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_dict, f, indent=4)
    print(f"Training parameters saved to {save_path}")


def save_loss_curves(train_losses, val_losses, save_path, params_dict=None):
    """
    Save training and validation loss curves as a plot.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
        params_dict: Optional dict with training params to display in title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Left plot: Both losses together
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training vs Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Individual losses with different scales if needed
    ax1 = axes[1]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_ylabel('Validation Loss', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    axes[1].set_title('Training and Validation Loss (Dual Scale)', fontsize=14)
    
    # Add summary text box
    if train_losses and val_losses:
        summary_text = f"Final Train Loss: {train_losses[-1]:.6f}\n"
        summary_text += f"Final Val Loss: {val_losses[-1]:.6f}\n"
        summary_text += f"Best Val Loss: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses)) + 1})"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.5, -0.05, summary_text, ha='center', fontsize=10, 
                 bbox=props, transform=fig.transFigure)
    
    # Add title with key parameters
    if params_dict:
        title = f"LR={params_dict.get('MAX_LR', 'N/A')}, BS={params_dict.get('BATCH_SIZE', 'N/A')}x{params_dict.get('UPDATE_EVERY_BATCHES', 'N/A')}, "
        title += f"L1={params_dict.get('USE_L1', 'N/A')}, L2={params_dict.get('USE_L2', 'N/A')}, SSIM={params_dict.get('USE_SSIM', 'N/A')}"
        fig.suptitle(title, fontsize=11, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Loss curves saved to {save_path}")


def save_detailed_loss_curves(loss_history, save_path):
    """
    Save detailed per-component loss curves.
    
    Args:
        loss_history: Dict with keys like 'train_l1', 'train_l2', 'val_l1', etc.
        save_path: Path to save the plot
    """
    # Get all unique loss types (l1, l2, ssim, perc, style)
    loss_types = set()
    for key in loss_history.keys():
        parts = key.split('_')
        if len(parts) >= 2:
            loss_types.add('_'.join(parts[1:]))
    
    if not loss_types:
        return
    
    n_types = len(loss_types)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4))
    
    if n_types == 1:
        axes = [axes]
    
    for idx, loss_type in enumerate(sorted(loss_types)):
        train_key = f'train_{loss_type}'
        val_key = f'val_{loss_type}'
        
        ax = axes[idx]
        
        if train_key in loss_history and loss_history[train_key]:
            epochs = range(1, len(loss_history[train_key]) + 1)
            ax.plot(epochs, loss_history[train_key], 'b-', label='Train', linewidth=2)
        
        if val_key in loss_history and loss_history[val_key]:
            epochs = range(1, len(loss_history[val_key]) + 1)
            ax.plot(epochs, loss_history[val_key], 'r-', label='Val', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{loss_type.upper()} Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Detailed loss curves saved to {save_path}")


def run_epoch(epoch_num, mode: str):
    """
    Run one training or validation epoch.
    
    Args:
        epoch_num: Current epoch number
        mode: 'train', 'val', or 'test'
    
    Returns:
        Average loss for the epoch
    """
    assert mode in {'train', 'val', 'test'}
    if mode == 'train':
        dataloader = train_dataloader
        steps_per_epoch = train_steps_per_epoch
        model.train()
    elif mode == 'val':
        dataloader = val_dataloader
        steps_per_epoch = val_steps_per_epoch
        model.eval()
    else:
        raise ValueError(f'Invalid mode: {mode}')

    print_every_steps = steps_per_epoch // 40 + 2

    avg_loss = 0

    print(f'Starting {mode} epoch {epoch_num}/{epochs}')

    if mode == 'train':
        optimizer.zero_grad()
    for i, batch in tqdm(enumerate(dataloader)):
        # Periodic memory cleanup every 50 batches
        if i > 0 and i % 50 == 0:
            cleanup_memory()
            check_memory_and_cleanup(threshold_gb=18.0, label=f"Batch {i}")

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
                
                # Clear matplotlib memory
                plt.clf()
                gc.collect()
        
        # Delete batch tensors to free memory
        del bls, fus, gts, fu_mask, outs, outputs, loss
        if i % 20 == 0:
            cleanup_memory()

    with torch.no_grad():
        avg_loss = avg_loss / steps_per_epoch
        avg_losses_dict = {l: v / steps_per_epoch for l, v in t_losses_dict.items()}
        print(f'Avg loss for {mode} epoch {epoch_num}: {avg_loss}')
        print(f'Avg losses dict for {mode} epoch {epoch_num}: {avg_losses_dict}')

        for l in t_losses_dict:
            t_losses_dict[l] = 0.

    # Memory cleanup after each epoch
    cleanup_memory()
    check_memory_and_cleanup(threshold_gb=20.0, label=f"End of {mode} epoch {epoch_num}")

    return avg_loss


if __name__ == '__main__':
    def get_cur_state_str(ep):
        # return f'id31_{f"Epoch{ep}_" if ep >= 0 else ""}Longitudinal_DeviceInvariant_DRRs_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_{"Perc_" if USE_PERC_STYLE else ""}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_Decoder6_{"Eff_ViT_" if "Eff" in LOAD_PATH else ""}{"MaskToken_" if USE_MASK_TOKEN else ""}{"MS-SSIM_" if USE_SSIM else ""}{"Adv2x_" if USE_GAN else ""}{"L1" if USE_L1 else ""}{"L2" if USE_L2 else ""}{"Fourier_masked" if USE_FOURIER else ""}{"_PosEmb" if USE_POS_EMBED else ""}_{"BN" if USE_BN else "GN"}'
        # return f'id45_{f"Epoch{ep}_" if ep >= 0 else ""}Longitudinal_AllEntities_DEVICES_FT_Cons_Sharpen_Dropout_ExtendedConvNet_{"Perc_" if USE_PERC_STYLE else ""}{MASKED_IN_CHANNELS}Channel_{MASK_MODE}{MASK_PATCH_SIZE}_Sched_Decoder6_{"Eff_ViT_" if "Eff" in LOAD_PATH else ""}{"MaskToken_" if USE_MASK_TOKEN else ""}{"MS-SSIM_" if USE_SSIM else ""}{"Adv2x_" if USE_GAN else ""}{"L1" if USE_L1 else ""}{"L2" if USE_L2 else ""}{"Fourier_masked" if USE_FOURIER else ""}{"_PosEmb" if USE_POS_EMBED else ""}_{"BN" if USE_BN else "GN"}'
        return f'sahar_model_test1'

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
    save_folder = '/cs/usr/sahar_aharon/Desktop/sahar_aharon/refactored code check'
    plots_folder = f'/cs/usr/sahar_aharon/Desktop/sahar_aharon/refactored code check/training/{get_cur_state_str(-1).split(".")[0]}'

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

    DRR_pair_dirs = ['/cs/usr/sahar_aharon/Desktop/sahar_aharon/refactored code check/train_set']

    DRR_single_dirs = []
    entity_dirs = []
    inpaint_dirs = []
    # DRR_single_dirs = []

    print(f'entity_dirs = \n{entity_dirs}')
    print(f'inpaint_dirs = \n{inpaint_dirs}')
    print(f'DRR_single_dirs = \n{DRR_single_dirs}')
    print(f'DRR_pair_dirs = \n{DRR_pair_dirs}')

    # Log initial memory usage
    log_memory("Before dataset loading")

    train_dataset = LongitudinalMIMDataset(entity_dirs=entity_dirs, inpaint_dirs=inpaint_dirs, DRR_single_dirs=DRR_single_dirs, DRR_pair_dirs=DRR_pair_dirs, invariance=invariance)
    # Use num_workers=0 to reduce memory footprint, or set to 2 max
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    len_train_ds = len(train_dataset)

    log_memory("After train dataset")

    # val_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[], invariance=invariance)
    val_dataset = LongitudinalMIMDataset([], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/GE MEDICAL SYSTEMS/angles_20_20_20'], invariance=invariance)
    # val_dataset = LongitudinalMIMDataset([], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/GE MEDICAL SYSTEMS/angles_20_20_20', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/Philips/angles_20_20_20', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/synthetic_pairs_test/SIEMENS/angles_20_20_20'], invariance=invariance)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    len_val_ds = len(val_dataset)

    # test_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[], invariance=invariance)
    test_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'], inpaint_dirs=[], DRR_single_dirs=[], DRR_pair_dirs=[], invariance=invariance)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    len_test_ds = len(test_dataset)

    log_memory("After all datasets loaded")

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
    
    # Detailed loss history for per-component tracking
    loss_history = {
        'train_total': [], 'val_total': [],
        'train_l1': [], 'val_l1': [],
        'train_l2': [], 'val_l2': [],
        'train_ssim': [], 'val_ssim': [],
        'train_perc': [], 'val_perc': [],
        'train_style': [], 'val_style': [],
    }
    
    # Collect all training parameters
    training_params = {
        # Timestamp
        'training_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Device & Model
        'device': str(DEVICE),
        'model_type': model.__class__.__name__,
        
        # Core hyperparameters
        'BATCH_SIZE': BATCH_SIZE,
        'UPDATE_EVERY_BATCHES': UPDATE_EVERY_BATCHES,
        'effective_batch_size': BATCH_SIZE * UPDATE_EVERY_BATCHES,
        'MAX_LR': MAX_LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'epochs': epochs,
        'start_epoch': start_epoch,
        
        # Image settings
        'IMG_SIZE': IMG_SIZE,
        'MASK_PATCH_SIZE': MASK_PATCH_SIZE,
        'MASK_MODE': MASK_MODE,
        
        # Loss functions
        'USE_L1': USE_L1,
        'USE_L2': USE_L2,
        'USE_SSIM': USE_SSIM,
        'USE_PERC_STYLE': USE_PERC_STYLE,
        'USE_FOURIER': USE_FOURIER,
        'USE_GAN': USE_GAN,
        
        # Loss weights
        'LAMBDA_L1_ALL': LAMBDA_L1_ALL,
        'LAMBDA_L2': LAMBDA_L2,
        'LAMBDA_SSIM': LAMBDA_SSIM,
        'LAMBDA_P': LAMBDA_P,
        'LAMBDA_S': LAMBDA_S,
        
        # Model settings
        'USE_BN': USE_BN,
        'USE_MASK_TOKEN': USE_MASK_TOKEN,
        'USE_POS_EMBED': USE_POS_EMBED,
        'USE_PATCH_DEC': USE_PATCH_DEC,
        'INIT_WEIGHTS': INIT_WEIGHTS,
        
        # Data augmentation
        'sharpen': sharpen,
        
        # Dataset info
        'train_dataset_size': len_train_ds,
        'val_dataset_size': len_val_ds,
        'test_dataset_size': len_test_ds,
        'train_steps_per_epoch': train_steps_per_epoch,
        'val_steps_per_epoch': val_steps_per_epoch,
        
        # Data directories
        'entity_dirs': entity_dirs,
        'inpaint_dirs': inpaint_dirs,
        'DRR_single_dirs': DRR_single_dirs,
        'DRR_pair_dirs': DRR_pair_dirs,
        'invariance': invariance,
        
        # Paths
        'save_folder': save_folder,
        'plots_folder': plots_folder,
        'LONGITUDINAL_LOAD_PATH': LONGITUDINAL_LOAD_PATH,
        
        # Scheduler settings
        'scheduler_type': 'OneCycleLR',
        'scheduler_pct_start': 0.15,
        'scheduler_anneal_strategy': 'cos',
        
        # Mask probability scheduling
        'INIT_MASK_PROB': INIT_MASK_PROB,
        'MAX_MASK_PROB': MAX_MASK_PROB,
        'END_MASK_PROB': END_MASK_PROB,
    }
    
    # Save initial training parameters
    save_training_params(training_params, f'{plots_folder}/training_params.json')

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
        
        # Update loss history
        loss_history['train_total'].append(avg_train_loss)
        loss_history['val_total'].append(avg_val_loss)
        
        # Save loss curves after each epoch
        save_loss_curves(train_losses, val_losses, 
                        f'{plots_folder}/loss_curves.png', 
                        params_dict=training_params)
        
        # Save detailed loss curves if we have component losses
        if any(loss_history[k] for k in ['train_l1', 'train_l2', 'train_ssim']):
            save_detailed_loss_curves(loss_history, f'{plots_folder}/detailed_loss_curves.png')
        
        # Update training params with current progress
        training_params['current_epoch'] = epoch
        training_params['best_val_loss'] = best_val_loss
        training_params['best_epoch'] = best_epoch if 'best_epoch' in dir() else epoch
        training_params['train_losses'] = train_losses
        training_params['val_losses'] = val_losses
        training_params['training_end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save updated training parameters
        save_training_params(training_params, f'{plots_folder}/training_params.json')

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