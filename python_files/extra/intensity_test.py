import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from extra.case_filtering import AFFINE_DCM


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


if __name__ == '__main__':
    out_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/intensity_changes'
    im_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_sigal/images2/18B.nii.gz'
    im_n = nib.load(im_p)
    im_d = im_n.get_fdata()[None, ...]
    im_aff = im_n.affine

    # im_d = image_histogram_equalization(im_d)[0]
    im_d = im_d * 1.15
    im_d = np.clip(im_d, a_min=0, a_max=255).T

    plt.imsave(f'{out_dir}/image_histogram_equalization.png', im_d.squeeze(), cmap='gray')