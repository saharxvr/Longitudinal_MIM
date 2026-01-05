import os

from tqdm import tqdm
from datasets import LongitudinalMIMDataset
import numpy as np
import imageio
import matplotlib.pyplot as plt


if __name__ == '__main__':
    out_folder = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/generated_samples'
    test_dataset = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/images',
                                           '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'],
                                          abnor_both_p=0.)

    samples_num = 100

    augs = ['masses', 'general_opacity', 'disordered_opacity', 'effusion_opacity']
    ab_obj = test_dataset.random_abnormalization_tf
    for aug in augs:
        print(f"Generating samples for augmentation \'{aug}\'")

        ab_obj.abnormalities_general = [getattr(ab_obj, aug)]
        ab_obj.abnormalities_local = [getattr(ab_obj, aug)]
        ab_obj.none_chance_to_update = 0.5
        test_dataset.shuffle()

        aug_dir = f'{out_folder}/{aug}'

        for i in tqdm(range(samples_num)):
            c_dir = f'{aug_dir}/{i}'
            os.makedirs(c_dir, exist_ok=True)

            bl, fu, gt, __ = test_dataset[i]
            bl = (bl * 255.).squeeze().numpy().astype(np.uint8)
            fu = (fu * 255.).squeeze().numpy().astype(np.uint8)
            # gt = (gt * 255.).squeeze().numpy().astype(np.uint8)
            imageio.imsave(f'{c_dir}/prior.png', bl)
            imageio.imsave(f'{c_dir}/current.png', fu)
            # imageio.imsave(f'{c_dir}/ground_truth.png', gt)
            plt.imsave(f'{c_dir}/ground_truth.png', gt.squeeze())
