import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import imageio.v3 as iio
from models import *
from constants import *
from preprocessing import *

print(torch.cuda.is_available())
p = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/images/'
imgs = [torch.tensor(iio.imread(p + '00002237_002.png'))[None, None, ...], torch.tensor(iio.imread(p + '00001749_000.png'))[None, None, ...], torch.tensor(iio.imread(p + '00003297_000.png'))[None, None, ...]]
batch = torch.cat(imgs, dim=0)
preprocess = BatchPreprocessing(use_fourier=False)
model = MaskedReconstructionModel(use_mask_token=USE_MASK_TOKEN).to(DEVICE)
model.load_state_dict(torch.load(LOAD_PATH)['model_dict'])
model.eval()
fig, axs = plt.subplots(2, 3)
mask_probs = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
plots_folder = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14'
with torch.no_grad():
    for mask_prob in mask_probs:
        inputs, _ = preprocess(batch)
        inputs = inputs.to(DEVICE)
        masked = generate_masked_images(inputs, mask_prob)
        outputs = model(masked)

        for j in range(2):
            axs[j, 0].imshow(inputs[j, 0, :, :].to('cpu').numpy(), cmap='gray')
            axs[j, 1].imshow(masked[j, 0, :, :].to('cpu').numpy(), cmap='gray')
            axs[j, 2].imshow(outputs[j, 0, :, :].to('cpu').numpy(), cmap='gray')
        plt.savefig(plots_folder + f'/MaskProb{mask_prob}.png')
        print(f"saving to {plots_folder + f'/MaskProb{mask_prob}.png'}")
        for j in range(2):
            axs[j, 0].clear()
            axs[j, 1].clear()
            axs[j, 2].clear()