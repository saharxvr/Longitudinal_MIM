from torch import Tensor
import torch
import random
import gc
import operator
from typing import Any

from scipy.spatial import Voronoi
from skimage.draw import line_nd
from itertools import combinations
import numpy as np

from constants import DEVICE
from Entity3D import Entity3D


class FluidOverload(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def generate_pair_stages():
        progress = 0 if random.random() < 0.2 else (1 if random.random() < 0.5 else -1)

        progress_dict = {
            'vessels_dilation': progress if random.random() < 0.05 else 0,
            'peribronchial_cuffing': progress if random.random() < 0.25 else 0,
            'vessels_haziness': progress if random.random() < 0.4 else 0,
            # 'fissural_thickening': progress if random.random() < 0.4 else 0,
            'septal_thickening': progress if random.random() < 0.4 else 0,
            'alveolar_edema': progress if random.random() < 0.4 else 0
        }
        # TODO: MAKE LATER STAGES LESS LIKELY (IN ADD_TO_CT AS WELL)
        # TODO: CHANGE BOUNDARY SEGS TO BE 'CROISSANTS' INSTEAD OF FLAT. CREATE EXAMPLES AND SEND BENNY.

        return progress_dict, progress

    # @staticmethod
    # def temp_dilation_holder(self):
        # init_add_hu = 400
        # final_add_hu = 200

        # init_add_hu = 300
        # final_add_hu = 0
        #
        # x = torch.arange(bronchi_seg.shape[0])
        # y = torch.arange(bronchi_seg.shape[1])
        # z = torch.arange(bronchi_seg.shape[2])
        #
        # grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        # # cent = (vessels_seg.shape[0] // 2, vessels_seg.shape[1] // 2, vessels_seg.shape[2] // 2)
        # cent = (bronchi_seg.shape[0] // 2, bronchi_seg.shape[1] // 2, 0)
        #
        # cpu_b_seg = boundary_seg.cpu()
        #
        # grid_x = (grid_x - cent[0]) / bronchi_seg.shape[0]
        # grid_x *= cpu_b_seg
        #
        # grid_y = (grid_y - cent[1]) / bronchi_seg.shape[1]
        # grid_y *= cpu_b_seg
        #
        # grid_z = (grid_z - cent[2]) / bronchi_seg.shape[2]
        # grid_z *= cpu_b_seg
        #
        # spread_coef = 0.72  # [~0.2-0.8], lower = more spread
        # add_map = torch.sqrt(grid_x ** 2 + grid_y ** 2 + grid_z ** 2)
        # add_map = (add_map - torch.min(add_map)) / (torch.max(add_map) - torch.min(add_map))
        # add_map = 1 / (1 + torch.exp(- (20 * (add_map - spread_coef))))
        # add_map *= init_add_hu
        # add_map = add_map.to(DEVICE)

    # @staticmethod
    # def temp_septal_thickening_outer_bias_holder():
        # left_lung_cent = torch.mean(left_Lung.nonzero().T.float(), dim=1)
        # right_lung_cent = torch.mean(right_Lung.nonzero().T.float(), dim=1)
        #
        # x = torch.arange(lungs_seg.shape[0])
        # y = torch.arange(lungs_seg.shape[1])
        # z = torch.arange(lungs_seg.shape[2])
        #
        # grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        #
        # grid_x_left = (grid_x - left_lung_cent[0])
        # grid_x_left *= left_Lung
        # grid_x_left = (grid_x_left - grid_x_left.min()) / (grid_x_left.max() - grid_x_left.min())
        #
        # grid_y_left = (grid_y - left_lung_cent[1]).abs()
        # grid_y_left *= left_Lung
        # grid_y_left = (grid_y_left - grid_y_left.min()) / (grid_y_left.max() - grid_y_left.min())
        #
        # grid_z_left = (grid_z - left_lung_cent[2]).abs()
        # grid_z_left *= left_Lung
        # grid_z_left = (grid_z_left - grid_z_left.min()) / (grid_z_left.max() - grid_z_left.min())
        #
        # grid_x_right = (grid_x - right_lung_cent[0])
        # grid_x_right *= right_Lung
        # grid_x_right = (grid_x_right - grid_x_right.min()) / (grid_x_right.max() - grid_x_right.min())
        #
        # grid_y_right = (grid_y - right_lung_cent[1]).abs()
        # grid_y_right *= right_Lung
        # grid_y_right = (grid_y_right - grid_y_right.min()) / (grid_y_right.max() - grid_y_right.min())
        #
        # grid_z_right = (grid_z - right_lung_cent[2]).abs()
        # grid_z_right *= right_Lung
        # grid_z_right = (grid_z_right - grid_z_right.min()) / (grid_z_right.max() - grid_z_right.min())
        #
        # if random.random() < 1: # TODO: CHANGE BACK
        #     if patient_mode == 'supine':
        #         x_add = (0, 0) if random.random() < 1 else (grid_x_left ** 2, grid_x_right ** 2) #TODO: CHANGE BACK
        #         add_grid_left = torch.sqrt(grid_y_left ** 2 + grid_z_left ** 2 + x_add[0])
        #         add_grid_right = torch.sqrt(grid_y_right ** 2 + grid_z_right ** 2 + x_add[1])
        #     else:
        #         add_grid_left = torch.sqrt(grid_x_left ** 2 + grid_y_left ** 2 + grid_z_left ** 2)
        #         add_grid_right = torch.sqrt(grid_x_right ** 2 + grid_y_right ** 2 + grid_z_right ** 2)
        #
        #     low = 0 # TODO: MAKE PARAM
        #     add_grid_left = (add_grid_left / add_grid_left.max()) * (1 - low) + low
        #     add_grid_right = (add_grid_right / add_grid_right.max()) * (1 - low) + low
        #
        #     min_val = -800.
        #     add_grid_left = (1 - add_grid_left) * min_val
        #     add_grid_right = (1 - add_grid_right) * min_val
        #
        #     intensity_map[left_Lung == 1] += add_grid_left.to(DEVICE)[left_Lung == 1]
        #     intensity_map[right_Lung == 1] += add_grid_right.to(DEVICE)[right_Lung == 1]
    @staticmethod
    def max_pool_scan_with_seg_and_boundaries(pair, seg, boundary_segs, max_pool_r, return_dilated_segs=False):
        seg1 = seg * boundary_segs[0]
        seg2 = seg * boundary_segs[1]
        dilated_seg1 = FluidOverload.binary_dilation(seg1.to(DEVICE), ('ball', max_pool_r))
        dilated_seg2 = FluidOverload.binary_dilation(seg2.to(DEVICE), ('ball', max_pool_r))
        max_pool_k = int(max_pool_r * 2)
        if max_pool_k % 2 == 0:
            max_pool_k += 1
        max_pool_prior = torch.max_pool3d(pair[0][None, None, ...], kernel_size=max_pool_k, stride=1, padding=max_pool_k // 2).squeeze()
        max_pool_current = torch.max_pool3d(pair[1][None, None, ...], kernel_size=max_pool_k, stride=1, padding=max_pool_k // 2).squeeze()
        pair[0][dilated_seg1 == 1.] = max_pool_prior[dilated_seg1 == 1.]
        pair[1][dilated_seg2 == 1.] = max_pool_current[dilated_seg2 == 1.]

        if return_dilated_segs:
            return pair, (dilated_seg1, dilated_seg2)

        return pair

    @staticmethod
    def max_pool_scan_with_seg_and_radiuses(pair, seg, dilation_rs, return_dilated_seg=False):
        if dilation_rs[0] >= 1:
            dilated_seg1 = FluidOverload.binary_dilation(seg.to(DEVICE), ('ball', dilation_rs[0]))
            max_pool_k1 = int(dilation_rs[0] * 2)
            if max_pool_k1 % 2 == 0:
                max_pool_k1 += 1
            max_pool_prior = torch.max_pool3d(pair[0][None, None, ...], kernel_size=max_pool_k1, stride=1, padding=max_pool_k1 // 2).squeeze()
            pair[0][dilated_seg1 == 1.] = max_pool_prior[dilated_seg1 == 1.]
        else:
            dilated_seg1 = seg.to(DEVICE)

        if dilation_rs[1] >= 1:
            dilated_seg2 = FluidOverload.binary_dilation(seg.to(DEVICE), ('ball', dilation_rs[1]))
            max_pool_k2 = int(dilation_rs[1] * 2)
            if max_pool_k2 % 2 == 0:
                max_pool_k2 += 1
            max_pool_current = torch.max_pool3d(pair[1][None, None, ...], kernel_size=max_pool_k2, stride=1, padding=max_pool_k2 // 2).squeeze()
            pair[1][dilated_seg2 == 1.] = max_pool_current[dilated_seg2 == 1.]
        else:
            dilated_seg2 = seg.to(DEVICE)

        if return_dilated_seg:
            return pair, [dilated_seg1, dilated_seg2]

        return pair

    @staticmethod
    def avg_pool_scan_with_seg_and_radiuses(pair, c_segs, dilation_rs, return_new_seg=False):
        if dilation_rs[0] >= 1:
            dilated_seg1 = FluidOverload.binary_dilation(c_segs[0].to(DEVICE), ('ball', dilation_rs[0]))
            avg_prior = FluidOverload.average_pooling_3d(pair[0], ('ball', dilation_rs[0]))
            pair[0][dilated_seg1 == 1.] = avg_prior[dilated_seg1 == 1.]
        else:
            dilated_seg1 = c_segs[0]

        if dilation_rs[1] >= 1:
            dilated_seg2 = FluidOverload.binary_dilation(c_segs[1].to(DEVICE), ('ball', dilation_rs[1]))
            avg_current = FluidOverload.average_pooling_3d(pair[1], ('ball', dilation_rs[1]))
            pair[1][dilated_seg2 == 1.] = avg_current[dilated_seg2 == 1.]
        else:
            dilated_seg2 = c_segs[1]

        if return_new_seg:
            return pair, (dilated_seg1, dilated_seg2)

        return pair

    @staticmethod
    def get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, patient_mode, supine_params):
        if patient_mode == 'supine':
            th_small_mult, th_small_bias, th_min_diff, yes_prog_0_p, yes_prog_all_p, no_prog_0_p, exp = supine_params
            # for vessels dilation: 0.45, 0.05, 0.45, 0.5, 0.15, 0.15

            th_small = (random.random() ** exp) * th_small_mult + th_small_bias
            min_th_dif = th_small + th_min_diff
            th_big = (random.random() ** exp) * (1 - min_th_dif) + min_th_dif
            th_small, th_big = 1 - th_small, 1 - th_big

            lungs_cropped_small = lungs_seg.clone()
            lungs_cropped_small[:, :, :lungs_shape_dict['min_d'] + int(lungs_shape_dict['depth'] * th_small)] = 0
            lungs_cropped_big = lungs_seg.clone()
            lungs_cropped_big[:, :, :lungs_shape_dict['min_d'] + int(lungs_shape_dict['depth'] * th_big)] = 0
            if c_progress == 1:
                prior_boundary_seg = torch.zeros_like(lungs_seg) if random.random() < yes_prog_0_p else lungs_cropped_small
                current_boundary_seg = lungs_seg.clone() if random.random() < yes_prog_all_p else lungs_cropped_big
            elif c_progress == -1:
                prior_boundary_seg = lungs_seg.clone() if random.random() < yes_prog_all_p else lungs_cropped_big
                current_boundary_seg = torch.zeros_like(lungs_seg) if random.random() < yes_prog_0_p else lungs_cropped_small
            else:
                prior_boundary_seg = torch.zeros_like(lungs_seg) if random.random() < no_prog_0_p else (lungs_cropped_small if random.random() < 0.5 else lungs_cropped_big)
                current_boundary_seg = prior_boundary_seg
        else:
            empty_seg = torch.zeros_like(lungs_seg)
            low_seg = torch.maximum(lobe_segs['lower_right_lobe'], lobe_segs['lower_left_lobe'])
            middle_seg = torch.maximum(low_seg, lobe_segs['middle_right_lobe'])
            full_seg = lungs_seg.clone()

            segs_list = [empty_seg, low_seg, middle_seg, full_seg]

            # Adding some asymmetric segmentations as noise:
            if random.random() < 0.05:
                if random.random() < 0.5:
                    segs_list.insert(1, lobe_segs['lower_right_lobe'])
                else:
                    segs_list.insert(1, lobe_segs['lower_left_lobe'])
            if random.random() < 0.05:
                if random.random() < 0.5:
                    segs_list.insert(3, torch.maximum(middle_seg, lobe_segs['upper_right_lobe']))
                else:
                    segs_list.insert(3, torch.maximum(middle_seg, lobe_segs['upper_left_lobe']))

            len_segs_list = len(segs_list)

            if c_progress == 1:
                idx_prior = random.randint(0, len_segs_list - 2)
                idx_current = random.randint(idx_prior + 1, len_segs_list - 1)
            elif c_progress == -1:
                idx_prior = random.randint(1, len_segs_list - 1)
                idx_current = random.randint(0, idx_prior - 1)
            else:
                idx_prior = random.randint(0, len_segs_list - 1)
                idx_current = idx_prior

            prior_boundary_seg = segs_list[idx_prior]
            current_boundary_seg = segs_list[idx_current]

        return prior_boundary_seg, current_boundary_seg


    @staticmethod
    def get_pair_values_with_ranges_and_min_diff(r_min, r_max, r_min_diff, c_progress, allow_halves=True):
        if allow_halves:
            # Because we divide by 2 later so we can get x.5 radiuses
            r_min *= 2
            r_max *= 2
            r_min_diff *= 2

        if c_progress == 1:
            if random.random() < 0.5:
                value_prior = random.randint(r_min, r_max - r_min_diff)
                value_current = random.randint(value_prior + r_min_diff, r_max)
            else:
                value_current = random.randint(r_min + r_min_diff, r_max)
                value_prior = random.randint(r_min, value_current - r_min_diff)
        elif c_progress == -1:
            if random.random() < 0.5:
                value_prior = random.randint(r_min + r_min_diff, r_max)
                value_current = random.randint(r_min, value_prior - r_min_diff)
            else:
                value_current = random.randint(r_min, r_max - r_min_diff)
                value_prior = random.randint(value_current + r_min_diff, r_max)
        else:
            value_prior = random.randint(r_min, r_max)
            value_current = value_prior

        if allow_halves:
            value_prior /= 2
            value_current /= 2

        return [value_prior, value_current]

    @staticmethod
    def get_max_and_avg_pool_r_pairs(r_min, r_max, r_min_diff, c_progress, p_zero_small, allow_halves=True):
        max_pool_rs = FluidOverload.get_pair_values_with_ranges_and_min_diff(r_min, r_max, r_min_diff, c_progress, allow_halves=allow_halves)
        avg_pool_rs = FluidOverload.get_pair_values_with_ranges_and_min_diff(r_min, r_max, r_min_diff, c_progress, allow_halves=allow_halves)

        if avg_pool_rs[0] < 1.:
            max_pool_rs[0] = 0
        if avg_pool_rs[1] < 1.:
            max_pool_rs[1] = 0

        if random.random() < p_zero_small:
            if c_progress == 1:
                max_pool_rs[0] = 0
                avg_pool_rs[0] = 0
            if c_progress == -1:
                max_pool_rs[1] = 0
                avg_pool_rs[1] = 0

        return max_pool_rs, avg_pool_rs

    @staticmethod
    def add_vessels_dilation(pair, vessels_seg, lungs_seg, lobe_segs, patient_mode, progress_dict, lungs_shape_dict):
        # Erect position is problematic due to the fact that CTs are taken in a supine position, thus the vessels are already equalized.
        c_progress = progress_dict['vessels_dilation']

        if patient_mode == 'erect':
            upper_lobes_seg = torch.maximum(lobe_segs['upper_right_lobe'], lobe_segs['upper_left_lobe'])

            if c_progress == 1:
                prior_boundary_seg = torch.zeros_like(lungs_seg)
                current_boundary_seg = upper_lobes_seg
            elif c_progress == -1:
                prior_boundary_seg = upper_lobes_seg
                current_boundary_seg = torch.zeros_like(lungs_seg)
            else:
                prior_boundary_seg = torch.zeros_like(lungs_seg) if random.random() < 0.35 else upper_lobes_seg
                current_boundary_seg = prior_boundary_seg

            if random.random() < 0.25:
                supine_params = (0.45, 0.05, 0.45, 0.5, 0.1, 0.4, 1)
                prior_opp_boundary_seg, current_opp_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, 'supine', supine_params)

                prior_boundary_seg *= prior_opp_boundary_seg
                current_boundary_seg *= current_opp_boundary_seg
        else:
            supine_params = (0.45, 0.05, 0.45, 0.5, 0.15, 0.15, 1.5)
            prior_boundary_seg, current_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, patient_mode, supine_params)

        if prior_boundary_seg.sum() == 0 and current_boundary_seg.sum() == 0:
            return pair, [vessels_seg.clone().to(DEVICE), vessels_seg.clone().to(DEVICE)]

        max_pool_r = random.choice([1, 1.5])
        new_pair = [pair[0].clone(), pair[1].clone()]
        new_pair, dilated_segs = FluidOverload.max_pool_scan_with_seg_and_boundaries(new_pair, vessels_seg, (prior_boundary_seg, current_boundary_seg), max_pool_r, return_dilated_segs=True)

        new_pair[0] = torch.maximum(new_pair[0], pair[0])
        new_pair[1] = torch.maximum(new_pair[1], pair[1])

        dilated_segs = [torch.maximum(dilated_segs[0], vessels_seg.to(DEVICE)), torch.maximum(dilated_segs[1], vessels_seg.to(DEVICE))]

        return new_pair, dilated_segs

    @staticmethod
    def add_vessels_haziness(pair, new_vessels_segs, lungs_seg, lobe_segs, patient_mode, progress_dict, lungs_shape_dict):
        c_progress = progress_dict['vessels_haziness']

        supine_params = (0.55, 0.1, 0.25, 0.4, 0.15, 0.4, 2)
        prior_boundary_seg, current_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, patient_mode, supine_params)

        if (patient_mode == 'erect' and random.random() < 0.85) or (patient_mode == 'supine' and random.random() < 0.1):
            opposite_patient_mode = 'erect' if patient_mode == 'supine' else 'supine'
            prior_opp_boundary_seg, current_opp_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, opposite_patient_mode, supine_params)

            prior_boundary_seg *= prior_opp_boundary_seg
            current_boundary_seg *= current_opp_boundary_seg

        if prior_boundary_seg.sum() == 0 and current_boundary_seg.sum() == 0:
            return pair

        r_min = 0
        r_max = 3.5
        r_min_diff = 2.
        dilation_rs, _ = FluidOverload.get_max_and_avg_pool_r_pairs(r_min, r_max, r_min_diff, c_progress, p_zero_small=0.125, allow_halves=True)

        if dilation_rs[0] >= 1:
            dilated_seg1 = FluidOverload.binary_dilation(new_vessels_segs[0], ('ball', dilation_rs[0]))
        else:
            dilated_seg1 = new_vessels_segs[0]
        if dilation_rs[1] >= 1:
            dilated_seg2 = FluidOverload.binary_dilation(new_vessels_segs[1], ('ball', dilation_rs[1]))
        else:
            dilated_seg2 = new_vessels_segs[1]

        add_map = FluidOverload.get_intensity_map(lungs_seg, inv_freq=4, scale=550, shift=100)

        def_int = max(dilation_rs) * 0.025
        def_int = tuple((def_int, def_int) for _ in range(3))

        to_add_seg_prior, to_add_seg_current = FluidOverload.random_deform([dilated_seg1 * prior_boundary_seg.to(DEVICE), dilated_seg2 * current_boundary_seg.to(DEVICE)], deform_resolution=8, frequencies=((4, 1), (4, 1), (4, 1)), intensities=def_int)

        to_add_seg_prior = to_add_seg_prior.squeeze() * (1 - new_vessels_segs[0])
        to_add_seg_current = to_add_seg_current.squeeze() * (1 - new_vessels_segs[1])

        new_pair = [pair[0].clone(), pair[1].clone()]

        new_pair[0] += add_map * to_add_seg_prior
        new_pair[1] += add_map * to_add_seg_current

        avg_pool_r = random.randint(4, 7) / 2
        new_pair[0][to_add_seg_prior == 1] = FluidOverload.average_pooling_3d(new_pair[0], ('ball', avg_pool_r))[to_add_seg_prior == 1]
        new_pair[1][to_add_seg_current == 1] = FluidOverload.average_pooling_3d(new_pair[1], ('ball', avg_pool_r))[to_add_seg_current == 1]

        return new_pair

    @staticmethod
    def add_peribronchial_cuffing(pair, bronchi_seg, progress_dict):
        c_progress = progress_dict['peribronchial_cuffing']

        r_min = 0
        r_max = 5
        r_min_diff = 2

        max_pool_rs, avg_pool_rs = FluidOverload.get_max_and_avg_pool_r_pairs(r_min, r_max, r_min_diff, c_progress, p_zero_small=0.125, allow_halves=True)
        avg_pool_rs = [min(avg_pool_rs[i], max_pool_rs[i]) for i in range(2)]

        new_pair = [pair[0].clone(), pair[1].clone()]
        new_pair, dilated_segs = FluidOverload.max_pool_scan_with_seg_and_radiuses(new_pair, bronchi_seg, max_pool_rs, return_dilated_seg=True)
        new_pair = FluidOverload.avg_pool_scan_with_seg_and_radiuses(new_pair, dilated_segs, avg_pool_rs, return_new_seg=False)

        new_pair[0] = torch.maximum(new_pair[0], pair[0])
        new_pair[1] = torch.maximum(new_pair[1], pair[1])

        return new_pair

    @staticmethod
    def add_line_to_seg(c_lines_seg, outer_rim_coords, ops, length_range, val=1.):
        p1 = outer_rim_coords[random.randint(0, outer_rim_coords.shape[0] - 1)]
        c_length = random.randint(length_range[0], length_range[1])
        p2 = torch.tensor([p1[0], ops[2](p1[1], c_length), p1[2]])
        l = line_nd(p1.cpu(), p2.cpu())
        c_lines_seg[l] = val

    @staticmethod
    def add_kerley_B_lines(pair, lungs_seg, left_lung_seg, right_lung_seg, progress_dict, lungs_shape_dict):
        c_progress = progress_dict['septal_thickening']

        n_lines_both = int((random.random() ** 2) * 49 + 1) if random.random() < 0.35 else 0
        n_lines_prog = int((random.random() ** 1.5) * 49 + 1) if c_progress != 0 else 0

        if n_lines_both == 0 and n_lines_prog == 0:
            return pair

        prior = pair[0]
        current = pair[1]

        # lungs_coords = lungs_seg.nonzero().float()
        lungs_x_edges = [lungs_shape_dict['min_w'], lungs_shape_dict['max_w']]
        lungs_center = lungs_shape_dict['center']
        left_lung_center = torch.mean(left_lung_seg.nonzero().float(), dim=0)
        right_lung_center = torch.mean(right_lung_seg.nonzero().float(), dim=0)

        ops_dict = {-1: (operator.lt, 0, operator.add), 1: (operator.gt, 1, operator.sub), 0: (operator.lt, 0, operator.add)}
        left_lung_ops = ops_dict[torch.sign(left_lung_center[1] - lungs_center[1]).item()]
        right_lung_ops = ops_dict[torch.sign(right_lung_center[1] - lungs_center[1]).item()]

        grid_x = torch.arange(0, lungs_seg.shape[1])[None, ..., None]

        dilated_left_lung = FluidOverload.binary_dilation(left_lung_seg, ('ball', 1.5))
        left_outer_coef = 0.75
        left_lung_x_th = left_lung_center[1] * (1 - left_outer_coef) + lungs_x_edges[left_lung_ops[1]] * left_outer_coef
        side_lung1 = left_lung_ops[0](grid_x, left_lung_x_th)
        outer_rim_left_lung = dilated_left_lung * (1. - left_lung_seg) * side_lung1
        outer_rim_left_lung_coords = outer_rim_left_lung.nonzero()

        dilated_right_lung = FluidOverload.binary_dilation(right_lung_seg, ('ball', 1.5))
        right_outer_coef = 0.75
        right_lung_x_th = right_lung_center[1] * (1 - right_outer_coef) + lungs_x_edges[right_lung_ops[1]] * right_outer_coef
        side_lung1 = right_lung_ops[0](grid_x, right_lung_x_th)
        outer_rim_right_lung = dilated_right_lung * (1. - right_lung_seg) * side_lung1
        outer_rim_right_lung_coords = outer_rim_right_lung.nonzero()

        lines_seg_low = np.zeros_like(lungs_seg.cpu().numpy())

        line_lengths = (10, 30)

        sides_vars = [(outer_rim_left_lung_coords, left_lung_ops), (outer_rim_right_lung_coords, right_lung_ops)]

        for _ in range(n_lines_both):
            c_vars = sides_vars[random.randint(0, 1)]
            FluidOverload.add_line_to_seg(lines_seg_low, c_vars[0], c_vars[1], line_lengths, val=1.)

        lines_seg_high = lines_seg_low.copy()

        for _ in range(n_lines_prog):
            c_vars = sides_vars[random.randint(0, 1)]
            FluidOverload.add_line_to_seg(lines_seg_high, c_vars[0], c_vars[1], line_lengths, val=1.)

        lines_seg_low = torch.tensor(lines_seg_low).to(DEVICE)
        lines_seg_high = torch.tensor(lines_seg_high).to(DEVICE)

        low_dilation_side = random.randint(2, 4)
        high_dilation_side = random.randint(2, 4)
        lines_seg_low = FluidOverload.binary_dilation(lines_seg_low, ('square', low_dilation_side))
        lines_seg_high = FluidOverload.binary_dilation(lines_seg_high, ('square', high_dilation_side))

        lines_seg_low, lines_seg_high = FluidOverload.random_deform([lines_seg_low, lines_seg_high], deform_resolution=8, frequencies=((5, 5), (3, 3), (3, 3)), intensities=((0.025, 0.025), (0.025, 0.025), (0.025, 0.025)))

        lines_seg_low = torch.round(lines_seg_low).squeeze()
        lines_seg_high = torch.round(lines_seg_high).squeeze()

        lines_seg_low = FluidOverload.binary_dilation(lines_seg_low, ('ball', 1.5))
        lines_seg_high = FluidOverload.binary_dilation(lines_seg_high, ('ball', 1.5))

        inv_freq = 10
        scale = 400
        shift = random.random() * 350 + 150
        intensity_map = FluidOverload.get_intensity_map(lungs_seg, inv_freq, scale, shift)

        dev_lungs = lungs_seg.to(DEVICE)
        border_filled_seg = FluidOverload.fill_borders_gap(prior, dev_lungs, dev_lungs, include_th=0)

        del dev_lungs
        cuda_lungs = None
        torch.cuda.empty_cache()
        gc.collect()

        line_segs_arrange = {-1: (lines_seg_high, lines_seg_low), 0: (lines_seg_high, lines_seg_high), 1: (lines_seg_low, lines_seg_high)}[c_progress]

        new_prior = prior.clone()
        new_current = current.clone()

        kernel = 1.5
        new_prior = FluidOverload.add_intensity(new_prior, line_segs_arrange[0], intensity_map, border_filled_seg, avg_radius=kernel, add_or_set='set')
        new_current = FluidOverload.add_intensity(new_current, line_segs_arrange[1], intensity_map, border_filled_seg, avg_radius=kernel, add_or_set='set')

        new_prior = torch.maximum(new_prior, prior)
        new_current = torch.maximum(new_current, current)

        return new_prior, new_current

    @staticmethod
    def add_septal_thickening(pair, lungs_seg, lobe_segs, progress_dict, patient_mode, lungs_shape_dict):
        c_progress = progress_dict['septal_thickening']

        prior = pair[0]
        current = pair[1]

        num_v = random.randint(175, 350)

        lungs_seg = FluidOverload.binary_dilation(lungs_seg, ('ball', 1.5))

        supine_params = (0.5, 0.05, 0.4, 0.2, 0.15, 0.5, 1)
        prior_boundary_seg, current_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, patient_mode, supine_params)

        if (patient_mode == 'erect' and random.random() < 0.85) or (patient_mode == 'supine' and random.random() < 0.4):
            opposite_patient_mode = 'erect' if patient_mode == 'supine' else 'supine'
            prior_opp_boundary_seg, current_opp_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, opposite_patient_mode, supine_params)

            prior_boundary_seg *= prior_opp_boundary_seg
            current_boundary_seg *= current_opp_boundary_seg

        if prior_boundary_seg.sum() == 0 and current_boundary_seg.sum() == 0:
            return pair

        numpy_seg = lungs_seg.cpu().numpy()
        coords = numpy_seg.nonzero()
        voronoi_mask = np.zeros_like(numpy_seg)

        points = []
        for _ in range(num_v):
            idx = random.randint(0, len(coords[0]) - 1)
            points.append([coords[0][idx], coords[1][idx], coords[2][idx]])

        v = Voronoi(points)

        vertices = v.vertices
        vertices[:, 0] = vertices[:, 0].clip(0, voronoi_mask.shape[0] - 1)
        vertices[:, 1] = vertices[:, 1].clip(0, voronoi_mask.shape[1] - 1)
        vertices[:, 2] = vertices[:, 2].clip(0, voronoi_mask.shape[2] - 1)

        for ver in v.ridge_vertices:
            for idxs in combinations(ver, 2):
                if idxs[0] == -1 or idxs[1] == -1:
                    break
                l = line_nd(vertices[idxs[0]], vertices[idxs[1]])
                voronoi_mask[l] = 1

        voronoi_mask = torch.tensor(voronoi_mask).float().to(DEVICE)
        voronoi_mask = FluidOverload.random_deform(voronoi_mask, deform_resolution=8, frequencies=((3, 2), (3, 2), (3, 2)), intensities=((0.035, 0.03), (0.035, 0.03), (0.035, 0.03))).squeeze()

        voronoi_mask_prior = voronoi_mask * prior_boundary_seg.to(DEVICE)
        voronoi_mask_current = voronoi_mask * current_boundary_seg.to(DEVICE)

        del voronoi_mask
        torch.cuda.empty_cache()
        gc.collect()

        inv_freq = 20
        scale = 550
        shift = -400.
        intensity_map = FluidOverload.get_intensity_map(lungs_seg, inv_freq, scale, shift)

        dev_lungs_seg = lungs_seg.to(DEVICE)
        border_filled_seg = FluidOverload.fill_borders_gap(prior, dev_lungs_seg, dev_lungs_seg, include_th=shift)

        del dev_lungs_seg
        torch.cuda.empty_cache()
        gc.collect()

        new_prior = prior.clone()
        new_current = current.clone()

        avg_r = random.choice([1, 2, 2.5])

        new_prior = FluidOverload.add_intensity(new_prior, voronoi_mask_prior, intensity_map, border_filled_seg, avg_radius=avg_r, add_or_set='set')
        new_current = FluidOverload.add_intensity(new_current, voronoi_mask_current, intensity_map, border_filled_seg, avg_radius=avg_r, add_or_set='set')

        new_prior = torch.maximum(new_prior, prior)
        new_current = torch.maximum(new_current, current)

        return new_prior, new_current

    # @staticmethod
    # def add_fissural_thickening(pair, lungs_seg, lobe_segs, sep_lungs, progress_dict):
    #     c_progress = progress_dict['fissural_thickening']
    #
    #     middle_right_lobe = lobe_segs['middle_right_lobe']
    #     upper_right_lobe = lobe_segs['upper_right_lobe']
    #
    #     left_lung = sep_lungs[0]
    #     right_lung = sep_lungs[1]
    #
    #     middle_right_lobe_mod = FluidOverload.binary_dilation(middle_right_lobe.to(DEVICE), ('ball', 1))
    #     # upper_right_lobe_mod = FluidOverload.binary_dilation(upper_right_lobe.to(DEVICE), ('ball', 1))
    #     upper_right_lobe_mod = upper_right_lobe.to(DEVICE)
    #
    #     intersection = middle_right_lobe_mod * upper_right_lobe_mod
    #     intersection = FluidOverload.binary_closing(intersection, ('ball', 2))
    #
    #     intersection = FluidOverload.random_deform(intersection, deform_resolution=8, frequencies=((8, 10), (1, 1), (1, 1)), intensities=((0.035, 0.035), (0, 0), (0, 0)))
    #
    #     inv_freq = 20
    #     scale = 100
    #     shift = 0
    #     intensity_map = FluidOverload.get_intensity_map(lungs_seg, inv_freq, scale, shift)
    #
    #     dev_intersection = intersection.to(DEVICE)
    #     dev_intersection = FluidOverload.fill_borders_gap(pair[1], dev_intersection, dev_intersection, include_th=50)
    #
    #     avg_r = 2
    #
    #     # new_prior = FluidOverload.add_intensity(new_prior, edema_seg_prior, intensity_map, border_filled_seg, avg_radius=avg_r, add_or_set='add')
    #     new_current = FluidOverload.add_intensity(pair[1], dev_intersection, intensity_map, dev_intersection, avg_radius=avg_r, add_or_set='set')
    #
    #     return pair[0], new_current

    @staticmethod
    def add_alveolar_edema(pair, lungs_seg, vessels_seg, bronchi_seg, lobe_segs, progress_dict, patient_mode, lungs_shape_dict):
        c_progress = progress_dict['alveolar_edema']

        prior = pair[0]
        current = pair[1]

        bronchi_seg = bronchi_seg * lungs_seg
        vessels_seg = vessels_seg * lungs_seg

        supine_params = (0.55, 0.15, 0.25, 0.35, 0.075, 0.5, 1)
        prior_boundary_seg, current_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, patient_mode, supine_params)

        if (patient_mode == 'erect' and random.random() < 0.65) or (patient_mode == 'supine' and random.random() < 0.4):
            opposite_patient_mode = 'erect' if patient_mode == 'supine' else 'supine'
            prior_opp_boundary_seg, current_opp_boundary_seg = FluidOverload.get_pair_boundary_segs(lungs_seg, lobe_segs, lungs_shape_dict, c_progress, opposite_patient_mode, supine_params)

            prior_boundary_seg *= prior_opp_boundary_seg
            current_boundary_seg *= current_opp_boundary_seg

        if prior_boundary_seg.sum() == 0 and current_boundary_seg.sum() == 0:
            return pair

        num_balls, r_range = random.choice([
            (random.randint(200, 350), (random.random() * 0.03 + 0.02, random.random() * 0.035 + 0.065)),
            (random.randint(60, 120), (random.random() * 0.04 + 0.03, random.random() * 0.05 + 0.085)),
            (random.randint(15, 35), (random.random() * 0.075 + 0.1, random.random() * 0.085 + 0.2)),
        ])

        lungs_h = lungs_shape_dict['height']
        r_range = (max(int(r_range[0] * lungs_h), 1), max(int(r_range[1] * lungs_h), 1))

        if vessels_seg.sum() > 0 and bronchi_seg.sum() > 0:
            init_seg = vessels_seg if random.random() < 0.75 else bronchi_seg
        else:
            init_seg = lungs_seg

        # edema_seg = FluidOverload.get_balls(vessels_seg, num_balls, r_range)
        edema_seg = FluidOverload.get_balls(init_seg, num_balls, r_range)

        (def_freqs1, def_ints1), (def_freqs2, def_ints2) = random.choice([
            ((((18, 8), (18, 8), (18, 8)), ((0.18, 0.12), (0.18, 0.12), (0.18, 0.12))),  (((4, 1), (4, 1), (4, 1)), ((0.18, 0.06), (0.18, 0.06), (0.18, 0.06)))),
            ((((18, 8), (18, 8), (18, 8)), ((0.12, 0.12), (0.18, 0.3), (0.12, 0.12))), (((4, 1), (4, 1), (4, 1)), ((0.12, 0.06), (0.18, 0.24), (0.12, 0.06))))
        ])

        edema_seg = FluidOverload.random_deform(edema_seg, deform_resolution=8, frequencies=def_freqs1, intensities=def_ints1)
        edema_seg = FluidOverload.random_deform(edema_seg, deform_resolution=8, frequencies=def_freqs2, intensities=def_ints2)

        inv_freq = random.randint(4, 20)
        scale = random.random() * 250 + 400
        shift = 300.
        intensity_map = FluidOverload.get_intensity_map(lungs_seg, inv_freq, scale, shift)

        dev_lungs_seg = lungs_seg.to(DEVICE)
        border_filled_seg = FluidOverload.fill_borders_gap(prior, dev_lungs_seg, dev_lungs_seg, include_th=20)

        del dev_lungs_seg
        torch.cuda.empty_cache()
        gc.collect()

        new_prior = prior.clone()
        new_current = current.clone()

        edema_seg_prior = edema_seg * prior_boundary_seg.to(DEVICE)
        edema_seg_current = edema_seg * current_boundary_seg.to(DEVICE)

        del edema_seg
        torch.cuda.empty_cache()
        gc.collect()

        avg_r = round((random.random() ** 2) * 8 + 6) / 2

        new_prior = FluidOverload.add_intensity(new_prior, edema_seg_prior, intensity_map, border_filled_seg, avg_radius=avg_r, add_or_set='add')
        new_current = FluidOverload.add_intensity(new_current, edema_seg_current, intensity_map, border_filled_seg, avg_radius=avg_r, add_or_set='add')

        new_prior = torch.maximum(new_prior, prior)
        new_current = torch.maximum(new_current, current)

        return new_prior, new_current

    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> dict[str, Any]:
        patient_mode = kwargs['patient_mode']

        progress_dict, progress = FluidOverload.generate_pair_stages()

        prior = scans[0]
        current = scans[1]
        lungs_seg = segs[0]['lungs']
        vessels_seg = segs[0]['lung_vessels']
        bronchi_seg = segs[0]['bronchi'] * lungs_seg
        lobe_names = ['upper_right_lobe', 'upper_left_lobe', 'middle_right_lobe', 'lower_right_lobe', 'lower_left_lobe']
        lobe_segs = {lobe_name: segs[0][lobe_name] for lobe_name in lobe_names}
        left_lung_seg = segs[0]['lung_left']
        right_lung_seg = segs[0]['lung_right']

        lungs_seg_coords = lungs_seg.nonzero().T
        lungs_center = torch.mean(lungs_seg_coords.float(), dim=1)
        min_h, max_h = torch.min(lungs_seg_coords[0]), torch.max(lungs_seg_coords[0])
        min_w, max_w = torch.min(lungs_seg_coords[1]), torch.max(lungs_seg_coords[1])
        min_d, max_d = torch.min(lungs_seg_coords[2]), torch.max(lungs_seg_coords[2])
        lungs_height = max_h - min_h
        lungs_width = max_w - min_w
        lungs_depth = max_d - min_d

        lungs_shape_dict = {'center': lungs_center, 'height': lungs_height, 'width': lungs_width, 'depth': lungs_depth, 'min_h': min_h, 'max_h': max_h, 'min_w': min_w, 'max_w': max_w, 'min_d': min_d, 'max_d': max_d}

        (prior, current), dilated_vessel_segs = FluidOverload.add_vessels_dilation((prior, current), vessels_seg, lungs_seg, lobe_segs, patient_mode, progress_dict, lungs_shape_dict)
        prior, current = FluidOverload.add_peribronchial_cuffing((prior, current), bronchi_seg, progress_dict)
        prior, current = FluidOverload.add_vessels_haziness((prior, current), dilated_vessel_segs, lungs_seg, lobe_segs, patient_mode, progress_dict, lungs_shape_dict)

        dil_segs0 = dilated_vessel_segs[0]
        dil_segs1 = dilated_vessel_segs[1]
        del dil_segs1
        del dil_segs0
        torch.cuda.empty_cache()
        gc.collect()

        # prior, current = FluidOverload.add_fissural_thickening((prior, current), lungs_seg, lobe_segs, (left_lung_seg, right_lung_seg), progress_dict)
        prior, current = FluidOverload.add_kerley_B_lines((prior, current), lungs_seg, left_lung_seg, right_lung_seg, progress_dict, lungs_shape_dict)
        prior, current = FluidOverload.add_septal_thickening((prior, current), lungs_seg, lobe_segs, progress_dict, patient_mode, lungs_shape_dict)
        prior, current = FluidOverload.add_alveolar_edema((prior, current), lungs_seg, vessels_seg, bronchi_seg, lobe_segs, progress_dict, patient_mode, lungs_shape_dict)

        air_broncho_choices = random.choices([(0, 0), (0, 1), (1, 0), (1, 1)], weights=[0.46, 0.04, 0.04, 0.46])[0]
        if air_broncho_choices[0]:
            FluidOverload.add_air_bronchogram(prior, bronchi_seg, orig_scan=kwargs['orig_scan'].to(DEVICE), boundary_seg=lungs_seg)
        if air_broncho_choices[1]:
            FluidOverload.add_air_bronchogram(current, bronchi_seg, orig_scan=kwargs['orig_scan'].to(DEVICE), boundary_seg=lungs_seg)

        scans = [prior, current]

        ret_dict = {'scans': scans, 'segs': segs, 'params': {}, 'fluidoverload_progress': progress}
        return ret_dict
