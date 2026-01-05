from torch import Tensor
import torch
import random
import gryds
from constants import DEVICE
import gc
from typing import Any

from Entity3D import Entity3D


class Cardiomegaly(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def expand_heart_seg(heart, grids, heart_cent, heart_height, orig_scan, orig_lungs, bronchi, upsample):
        grid_x, grid_y, grid_z = grids

        upper_heart = heart * (grid_x < heart_cent[0] + 0.25 * heart_height)
        upper_heart_seg_coords = upper_heart.nonzero().T
        upper_min_h, upper_max_h = torch.min(upper_heart_seg_coords[0]), torch.max(upper_heart_seg_coords[0])
        upper_min_w, upper_max_w = torch.min(upper_heart_seg_coords[1]), torch.max(upper_heart_seg_coords[1])
        upper_min_d, upper_max_d = torch.min(upper_heart_seg_coords[2]), torch.max(upper_heart_seg_coords[2])

        upper_heart_roi = (grid_y > upper_min_w) * (grid_y < upper_max_w) * (grid_z > upper_min_d) * (grid_z < upper_max_d) * (grid_x < upper_max_h)

        downsample_fill_seg = torch.nn.Upsample(size=(int(heart.shape[-3] * 0.25), int(heart.shape[-2] * 1), int(heart.shape[-1] * 1))).to(DEVICE)
        downsample_fill_seg2 = torch.nn.Upsample(size=(int(heart.shape[-3] * 0.25), int(heart.shape[-2] * 0.25), int(heart.shape[-1] * 0.25))).to(DEVICE)

        upper_heart = downsample_fill_seg(upper_heart[None, None, ...].to(DEVICE)).squeeze()
        upper_heart = Cardiomegaly.fill_borders_gap(downsample_fill_seg(orig_scan[None, None, ...]).squeeze().to(DEVICE), upper_heart, upper_heart, include_th=-180, dilate_r=13, op='gt')
        upper_heart = Cardiomegaly.binary_closing(downsample_fill_seg2(upper_heart[None, None, ...]).squeeze(), ('ball', 9)) * (downsample_fill_seg2(orig_scan[None, None, ...].to(DEVICE)).squeeze() > -180)
        upper_heart = upsample(upper_heart[None, None, ...]).squeeze().cpu()
        upper_heart = upper_heart * upper_heart_roi

        dilated_bronchi = Cardiomegaly.binary_dilation(bronchi.to(DEVICE), ('ball', 8))
        dilated_bronchi = Cardiomegaly.binary_dilation(dilated_bronchi, ('ball', 8)).cpu()

        heart = (torch.maximum(heart, upper_heart) * (1 - orig_lungs) * (1 - dilated_bronchi)).to(DEVICE)

        heart = Cardiomegaly.binary_opening(heart, ('ball', 7))
        heart = Cardiomegaly.binary_closing(heart, ('ball', 7))

        return heart

    @staticmethod
    def get_grid_params_for_pair(progress):
        intensity_x_pos = [random.random() * 0.1 + 0.05, random.random() * 0.1 + 0.05]
        intensity_x_neg = [random.random() * 0.35 + 0.25, random.random() * 0.35 + 0.25]
        intensity_y_pos = [random.random() * 0.275 + 0.075, random.random() * 0.275 + 0.075]
        intensity_y_neg = [random.random() * 0.4 + 0.25, random.random() * 0.4 + 0.25]
        intensity_z_pos = [random.random() * 0.275 + 0.075, random.random() * 0.275 + 0.075]
        intensity_z_neg = [random.random() * 0.275 + 0.075, random.random() * 0.275 + 0.075]

        x_down_fac_pos = [random.random() * 0.1 + 0.85, random.random() * 0.1 + 0.85]
        x_down_fac_neg = [(1 - random.random() ** 2) * 0.65 + 0.2, (1 - random.random() ** 2) * 0.65 + 0.2]
        y_down_fac_pos = [(1 - random.random() ** 2.5) * 0.6 + 0.3, (1 - random.random() ** 2.5) * 0.6 + 0.3]
        y_down_fac_neg = [(1 - random.random() ** 2) * 0.7 + 0.15, (1 - random.random() ** 2) * 0.7 + 0.15]
        z_down_fac = [random.random() * 0.15 + 0.8, random.random() * 0.15 + 0.8]

        prior_choice_func1, current_choice_func1, prior_choice_func2, current_choice_func2 = {1: (min, max, max, min), -1: (max, min, min, max)}[progress]

        params_dict = {
            'intensity_x_pos': (prior_choice_func1(intensity_x_pos), current_choice_func1(intensity_x_pos)),
            'intensity_x_neg': (prior_choice_func1(intensity_x_neg), current_choice_func1(intensity_x_neg)),
            'intensity_y_pos': (prior_choice_func1(intensity_y_pos), current_choice_func1(intensity_y_pos)),
            'intensity_y_neg': (prior_choice_func1(intensity_y_neg), current_choice_func1(intensity_y_neg)),
            'intensity_z_pos': (prior_choice_func1(intensity_z_pos), current_choice_func1(intensity_z_pos)),
            'intensity_z_neg': (prior_choice_func1(intensity_z_neg), current_choice_func1(intensity_z_neg)),
            'x_down_fac_pos': (prior_choice_func2(x_down_fac_pos), current_choice_func2(x_down_fac_pos)),
            'x_down_fac_neg': (prior_choice_func2(x_down_fac_neg), current_choice_func2(x_down_fac_neg)),
            'y_down_fac_pos': (prior_choice_func2(y_down_fac_pos), current_choice_func2(y_down_fac_pos)),
            'y_down_fac_neg': (prior_choice_func2(y_down_fac_neg), current_choice_func2(y_down_fac_neg)),
            'z_down_fac': (prior_choice_func2(z_down_fac), current_choice_func2(z_down_fac)),
        }
        return params_dict

    @staticmethod
    def create_grids_for_single(grids, params, idx, heart, other_params, masks, orig_lungs, diff_x, upsample):
        grid_x, grid_y, grid_z = grids
        dial_r, avg_r, exp_mult, exp_bias = other_params
        mask_x_pos_y_pos, mask_x_pos_y_neg, mask_x_neg_y_pos, mask_x_neg_y_neg = masks

        intensity_x_pos = params['intensity_x_pos'][idx]
        intensity_x_neg = params['intensity_x_neg'][idx]
        grid_x[grid_x > 0] *= intensity_x_pos
        grid_x[grid_x <= 0] *= intensity_x_neg

        intensity_y_pos = 0.25
        intensity_y_neg = 0.5
        grid_y[grid_y > 0] *= intensity_y_pos
        grid_y[grid_y <= 0] *= intensity_y_neg

        intensity_z_pos = 0.12
        intensity_z_neg = 0.12
        grid_z[grid_z > 0] *= intensity_z_pos
        grid_z[grid_z <= 0] *= intensity_z_neg

        x_down_fac_pos = 0.9
        x_down_fac_neg = 0.7
        y_down_fac_pos = 0.7
        y_down_fac_neg = 0.7
        z_down_fac = 0.9

        downsample_x_pos_y_pos = torch.nn.Upsample(size=(int(heart.shape[-3] * x_down_fac_pos), int(heart.shape[-2] * y_down_fac_pos), int(heart.shape[-1] * z_down_fac)))
        downsample_x_pos_y_neg = torch.nn.Upsample(size=(int(heart.shape[-3] * x_down_fac_pos), int(heart.shape[-2] * y_down_fac_neg), int(heart.shape[-1] * z_down_fac)))
        downsample_x_neg_y_pos = torch.nn.Upsample(size=(int(heart.shape[-3] * x_down_fac_neg), int(heart.shape[-2] * y_down_fac_pos), int(heart.shape[-1] * z_down_fac)))
        downsample_x_neg_y_neg = torch.nn.Upsample(size=(int(heart.shape[-3] * x_down_fac_neg), int(heart.shape[-2] * y_down_fac_neg), int(heart.shape[-1] * z_down_fac)))

        def_mult_x_pos_y_pos = downsample_x_pos_y_pos((heart * mask_x_pos_y_pos.to(DEVICE))[None, None, ...]).squeeze()
        def_mult_x_pos_y_neg = downsample_x_pos_y_neg((heart * mask_x_pos_y_neg.to(DEVICE))[None, None, ...]).squeeze()
        def_mult_x_neg_y_pos = downsample_x_neg_y_pos((heart * mask_x_neg_y_pos.to(DEVICE))[None, None, ...]).squeeze()
        def_mult_x_neg_y_neg = downsample_x_neg_y_neg((heart * mask_x_neg_y_neg.to(DEVICE))[None, None, ...]).squeeze()

        def_mult_x_pos_y_pos = Cardiomegaly.binary_dilation(def_mult_x_pos_y_pos, ('ball', dial_r))
        def_mult_x_pos_y_neg = Cardiomegaly.binary_dilation(def_mult_x_pos_y_neg, ('ball', dial_r))
        def_mult_x_neg_y_pos = Cardiomegaly.binary_dilation(def_mult_x_neg_y_pos, ('ball', dial_r))
        def_mult_x_neg_y_neg = Cardiomegaly.binary_dilation(def_mult_x_neg_y_neg, ('ball', dial_r))

        def_mult_x_pos_y_pos = upsample(def_mult_x_pos_y_pos[None, None, ...]).squeeze()
        def_mult_x_pos_y_neg = upsample(def_mult_x_pos_y_neg[None, None, ...]).squeeze()
        def_mult_x_neg_y_pos = upsample(def_mult_x_neg_y_pos[None, None, ...]).squeeze()
        def_mult_x_neg_y_neg = upsample(def_mult_x_neg_y_neg[None, None, ...]).squeeze()

        def_mult = torch.maximum(torch.maximum(def_mult_x_pos_y_pos, def_mult_x_pos_y_neg), torch.maximum(def_mult_x_neg_y_pos, def_mult_x_neg_y_neg))
        def_mult = Cardiomegaly.binary_closing(def_mult, ('ball', 15))
        def_mult = Cardiomegaly.average_pooling_3d(def_mult, ('ball', avg_r)).cpu()
        def_mult *= orig_lungs

        grid_x *= def_mult
        grid_y *= def_mult
        grid_z *= def_mult

        y_mult_x_pos_y_neg = torch.ones_like(orig_lungs)
        c_grid_y_mult: torch.Tensor = diff_x * mask_x_pos_y_neg
        c_grid_y_mult[c_grid_y_mult > 0] = (torch.max(c_grid_y_mult) - c_grid_y_mult)[c_grid_y_mult > 0]
        c_grid_y_mult = c_grid_y_mult / torch.max(c_grid_y_mult)
        c_grid_y_mult[c_grid_y_mult > 0] = (1 / (1 + torch.exp(- (exp_mult * (c_grid_y_mult - exp_bias)))))[c_grid_y_mult > 0]
        msk = mask_x_pos_y_neg == 1
        y_mult_x_pos_y_neg[msk] = c_grid_y_mult[msk]

        grid_y *= y_mult_x_pos_y_neg

        return grid_x, grid_y, grid_z


    @staticmethod
    def create_grids_for_pair(orig_grids, heart, progress, heart_cent, lungs_width, orig_lungs, upsample):
        grid_x, grid_y, grid_z = orig_grids

        diff_x = (heart_cent[0] - grid_x)
        diff_y = (heart_cent[1] - grid_y)
        diff_z = (heart_cent[2] - grid_z)

        grid_x = (diff_x / lungs_width)
        grid_y = (diff_y / lungs_width)
        grid_z = (diff_z / lungs_width)

        params = Cardiomegaly.get_grid_params_for_pair(progress)

        dial_r = random.randint(6, 12)
        avg_r = 8
        exp_mult = random.randint(8, 12)
        exp_bias = random.random() * 0.4 + 0.35
        other_params = (dial_r, avg_r, exp_mult, exp_bias)

        mask_x_pos_y_pos = (grid_x > 0) * (grid_y > 0)
        mask_x_pos_y_neg = (grid_x > 0) * (grid_y <= 0)
        mask_x_neg_y_pos = (grid_x <= 0) * (grid_y > 0)
        mask_x_neg_y_neg = (grid_x <= 0) * (grid_y <= 0)
        masks = (mask_x_pos_y_pos, mask_x_pos_y_neg, mask_x_neg_y_pos, mask_x_neg_y_neg)

        if progress == -1 or random.random() < 0.5:
            grid_x_prior, grid_y_prior, grid_z_prior = Cardiomegaly.create_grids_for_single((grid_x.clone(), grid_y.clone(), grid_z.clone()), params, 0, heart, other_params, masks, orig_lungs, diff_x, upsample)
        else:
            grid_x_prior, grid_y_prior, grid_z_prior = None, None, None
        if progress == 1 or random.random() < 0.5:
            grid_x_current, grid_y_current, grid_z_current = Cardiomegaly.create_grids_for_single((grid_x.clone(), grid_y.clone(), grid_z.clone()), params, 1, heart, other_params, masks, orig_lungs, diff_x, upsample)
        else:
            grid_x_current, grid_y_current, grid_z_current = None, None, None

        return (grid_x_prior, grid_y_prior, grid_z_prior), (grid_x_current, grid_y_current, grid_z_current)
    
    @staticmethod
    def apply_deformations_and_postprocess(scan, heart, bspline):
        interpolator = gryds.BSplineInterpolatorCuda(scan.squeeze().to(DEVICE))
        deformed_scan = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

        interpolator = gryds.BSplineInterpolatorCuda(heart.squeeze())
        deformed_heart = torch.tensor(interpolator.transform(bspline)).to(DEVICE)
        deformed_heart = torch.round(deformed_heart)

        deformed_scan[deformed_heart == 0] = scan.to(DEVICE)[deformed_heart == 0]

        r_avg = random.randint(8, 18) / 2
        deformed_heart_eroded_rim = Cardiomegaly.binary_erosion(deformed_heart, ('ball', r_avg))
        deformed_heart_eroded_rim = deformed_heart * (1 - deformed_heart_eroded_rim)
        deformed_current_avg = Cardiomegaly.average_pooling_3d(deformed_scan, ('ball', r_avg))
        deformed_scan[deformed_heart_eroded_rim == 1] = deformed_current_avg[deformed_heart_eroded_rim == 1]
        
        return deformed_scan, deformed_heart
        
    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> dict[str, Any]:
        assert torch.equal(segs[0]['heart'], segs[1]['heart']), "\'Cardiomegaly\' requires that prior and current have the same heart segmentations"

        prior = scans[0]
        current = scans[1]
        registrated_prior = kwargs['registrated_prior']

        heart = segs[0]['heart']
        bronchi = segs[0]['bronchi']
        orig_lungs = kwargs['orig_lungs']
        orig_scan = kwargs['orig_scan']
        dev_orig_lungs = orig_lungs.to(DEVICE)
        orig_lungs = Cardiomegaly.fill_borders_gap(orig_scan.to(DEVICE), dev_orig_lungs, dev_orig_lungs, include_th=150, dilate_r=5).to(orig_lungs.device)

        del dev_orig_lungs
        torch.cuda.empty_cache()
        gc.collect()

        if 'fluidoverload_progress' in kwargs:
            fo_progress = kwargs['fluidoverload_progress']
        else:
            fo_progress = None

        if fo_progress is None or fo_progress == 0:
            progress = random.choices([1, -1], weights=[0.5, 0.5])[0]
        else:
            progress = random.choices([fo_progress, -fo_progress], weights=[0.95, 0.05])[0]

        lungs_coords = orig_lungs.nonzero().T
        lungs_min_w, lungs_max_w = torch.min(lungs_coords[1]), torch.max(lungs_coords[1])
        lungs_width = lungs_max_w - lungs_min_w

        heart_seg_coords = heart.nonzero().T
        heart_cent = torch.mean(heart_seg_coords.float(), dim=1)
        min_h, max_h = torch.min(heart_seg_coords[0]), torch.max(heart_seg_coords[0])
        heart_height = (max_h - min_h).item()

        orig_shape = heart.squeeze().shape
        upsample = torch.nn.Upsample(size=orig_shape).to(DEVICE)

        x = torch.arange(heart.shape[0])
        y = torch.arange(heart.shape[1])
        z = torch.arange(heart.shape[2])

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

        heart = Cardiomegaly.expand_heart_seg(heart, (grid_x, grid_y, grid_z), heart_cent, heart_height, orig_scan, orig_lungs, bronchi, upsample)

        (grid_x_prior, grid_y_prior, grid_z_prior), (grid_x_current, grid_y_current, grid_z_current) = Cardiomegaly.create_grids_for_pair((grid_x, grid_y, grid_z), heart, progress, heart_cent, lungs_width, orig_lungs, upsample)

        if grid_x_prior is not None:
            bspline = gryds.BSplineTransformationCuda([grid_x_prior, grid_y_prior, grid_z_prior], order=1)
            deformed_prior, deformed_prior_heart = Cardiomegaly.apply_deformations_and_postprocess(prior, heart, bspline)
        else:
            deformed_prior = prior
            deformed_prior_heart = heart
        
        if grid_x_current is not None:
            bspline = gryds.BSplineTransformationCuda([grid_x_current, grid_y_current, grid_z_current], order=1)
            deformed_current, deformed_current_heart = Cardiomegaly.apply_deformations_and_postprocess(current, heart, bspline)
            
            interpolator = gryds.BSplineInterpolatorCuda(registrated_prior.squeeze().to(DEVICE))
            deformed_registrated_prior = torch.tensor(interpolator.transform(bspline)).to(DEVICE)
        else:
            deformed_current = current
            deformed_current_heart = heart

            deformed_registrated_prior = registrated_prior
            
        prior_heart_seg_def_coords = deformed_prior_heart.nonzero().T
        prior_min_w, prior_max_w = torch.min(prior_heart_seg_def_coords[1]), torch.max(prior_heart_seg_def_coords[1])
        prior_heart_def_width = (prior_max_w - prior_min_w).item()
        
        current_heart_seg_def_coords = deformed_current_heart.nonzero().T
        current_min_w, current_max_w = torch.min(current_heart_seg_def_coords[1]), torch.max(current_heart_seg_def_coords[1])
        current_heart_def_width = (current_max_w - current_min_w).item()
        
        prior_ctr = prior_heart_def_width / lungs_width
        current_ctr = current_heart_def_width / lungs_width
        if abs(current_ctr - prior_ctr) > 0.075:
            final_progress = 1 if current_ctr > prior_ctr else -1
        else:
            final_progress = 0

        cardiomegaly_seg = deformed_current_heart - deformed_prior_heart

        print(f'Prior CTR = {prior_ctr}')
        print(f'Current CTR = {current_ctr}')

        scans = [deformed_prior, deformed_current]

        ret_dict = {'scans': scans, 'segs': segs, 'params': {}, 'registrated_prior': deformed_registrated_prior,
                    'cardiomegaly_seg': cardiomegaly_seg, 'cardiomegaly_progress': final_progress}

        return ret_dict