from abc import ABC, abstractmethod
from torch import Tensor
import torch
import random
import gryds
from constants import DEVICE
import gc
from skimage.morphology import ball, octahedron, rectangle, square
from typing import Tuple, Any
import scipy.ndimage as ndi
import numpy as np

STRUCT3D = np.ones((3, 3, 3))

class Entity3D(ABC):

    @staticmethod
    @abstractmethod
    # def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> dict[str, Any]:
        pass

    @staticmethod
    def calc_lungs_height(lungs_seg: torch.Tensor):
        lungs_coords = lungs_seg.nonzero()
        z_coords = lungs_coords.T[0]
        return (torch.max(z_coords) - torch.min(z_coords)).item()

    @staticmethod
    def get_balls(boundary_seg: torch.Tensor, num_balls: int, radius_range: Tuple[int, int]):
        boundary_seg = torch.nn.functional.pad(boundary_seg, (radius_range[1], radius_range[1], radius_range[1], radius_range[1], radius_range[1], radius_range[1]))
        balls = torch.zeros_like(boundary_seg, dtype=torch.float32).to(DEVICE)

        if num_balls == 0:
            balls = balls[radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1]]
            return balls

        boundary_coords = boundary_seg.nonzero()

        for m in range(num_balls):
            radius = random.randint(radius_range[0], radius_range[1])
            coord = boundary_coords[random.randint(0, boundary_coords.shape[0] - 1)]
            c_ball = torch.tensor(ball(radius, dtype=float), dtype=torch.float32).to(DEVICE)
            balls[coord[0] - radius: coord[0] + radius + 1, coord[1] - radius: coord[1] + radius + 1, coord[2] - radius: coord[2] + radius + 1] = c_ball

        balls = balls[radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1]]

        return balls

    @staticmethod
    def fill_borders_gap(ct_scan, entity, seg, include_th=-30., dilate_r=3, op='lt'):
        # eroded_seg = -torch.nn.functional.max_pool3d(-seg[None, None, ...], kernel_size=3, stride=1, padding=1).squeeze()
        eroded_seg = Entity3D.binary_erosion(seg, ('ball', 1))
        seg_hull = seg * (1. - eroded_seg)
        edge_voxels = entity * seg_hull
        # dilated_edge_voxels = torch.nn.functional.max_pool3d(edge_voxels[None, None, ...], kernel_size=5, stride=1, padding=2).squeeze()
        dilated_edge_voxels = Entity3D.binary_dilation(edge_voxels, ('ball', dilate_r))
        if op == 'lt':
            voxels_to_include = (ct_scan < include_th) * dilated_edge_voxels
        else:
            voxels_to_include = (ct_scan > include_th) * dilated_edge_voxels
        filled_gap_entity = torch.maximum(entity, voxels_to_include)
        return filled_gap_entity

    @staticmethod
    def get_intensity_map(lungs_seg, inv_freq, scale, shift):
        upsample = torch.nn.Upsample(size=(lungs_seg.shape[-3], lungs_seg.shape[-2], lungs_seg.shape[-1]), mode='trilinear')
        if inv_freq > 1:
            intensity_map = (torch.rand((1, 1, lungs_seg.shape[-3] // inv_freq, lungs_seg.shape[-2] // inv_freq, lungs_seg.shape[-1] // inv_freq)) * scale + shift).to(torch.float32).to(DEVICE)
            intensity_map = upsample(intensity_map).squeeze()
        else:
            intensity_map = (torch.rand((lungs_seg.shape[-3], lungs_seg.shape[-2], lungs_seg.shape[-1])) * scale + shift).to(torch.float32).to(DEVICE)
        intensity_map = torch.nn.functional.pad(intensity_map, (0, lungs_seg.shape[-1] - intensity_map.shape[-1], 0, lungs_seg.shape[-2] - intensity_map.shape[-2], 0, lungs_seg.shape[-3] - intensity_map.shape[-3]))
        return intensity_map

    @staticmethod
    def override_with_intensity(ct_scan, entity, intensity_map, avg_kernel=3):
        entity_coords = entity != 0.
        ct_scan[entity_coords] = intensity_map[entity_coords]
        ct_scan_avg = torch.nn.functional.avg_pool3d(ct_scan[None, None, ...], kernel_size=avg_kernel, stride=1, padding=avg_kernel // 2).squeeze()
        ct_scan[entity_coords] = ct_scan_avg[entity_coords]
        return ct_scan

    @staticmethod
    def postprocess_intensity_for_set(ct_scan, entity, intensity_map, seg):
        entity_coords = (entity != 0.).float()

        coords_kernel = avg_kernel // 2
        if coords_kernel % 2 == 0:
            coords_kernel += 1

        entity_coords_dilated = torch.nn.functional.max_pool3d(entity_coords[None, None, ...], kernel_size=coords_kernel, stride=1, padding=coords_kernel // 2).squeeze()
        entity_coords_eroded = -torch.nn.functional.max_pool3d(-entity_coords[None, None, ...], kernel_size=coords_kernel, stride=1, padding=coords_kernel // 2).squeeze()
        intensity_map *= entity_coords
        intensity_map *= seg
        # if max_val is not None:
        #     intensity_map[ct_scan > max_val] = 0.
        intensity_map_avg = intensity_map.clone()
        intensity_map_avg[intensity_map == 0.] = ct_scan[intensity_map == 0.]
        intensity_map_avg = torch.nn.functional.avg_pool3d(intensity_map_avg[None, None, ...], kernel_size=avg_kernel, stride=1, padding=avg_kernel // 2).squeeze()
        int_indic = (entity_coords_dilated * (1. - entity_coords_eroded)).bool()
        intensity_map[int_indic] = intensity_map_avg[int_indic]
        intensity_map *= seg

    @staticmethod
    def postprocess_intensity_for_add(ct_scan, entity, intensity_map, seg):
        entity_coords = (entity != 0.).float()

        coords_kernel = avg_kernel // 2
        if coords_kernel % 2 == 0:
            coords_kernel += 1

        entity_coords_dilated = torch.nn.functional.max_pool3d(entity_coords[None, None, ...], kernel_size=coords_kernel, stride=1, padding=coords_kernel // 2).squeeze()
        entity_coords_eroded = -torch.nn.functional.max_pool3d(-entity_coords[None, None, ...], kernel_size=coords_kernel, stride=1, padding=coords_kernel // 2).squeeze()
        intensity_map *= entity_coords
        intensity_map *= seg
        # if max_val is not None:
        #     intensity_map[ct_scan > max_val] = 0.
        intensity_map_avg = torch.nn.functional.avg_pool3d(intensity_map[None, None, ...], kernel_size=avg_kernel, stride=1, padding=avg_kernel // 2).squeeze()
        int_indic = (entity_coords_dilated * (1. - entity_coords_eroded)).bool()
        intensity_map[int_indic] = intensity_map_avg[int_indic]
        intensity_map *= seg

    @staticmethod
    def add_intensity(ct_scan, entity, intensity_map, seg, avg_radius=5, add_or_set='add', max_val=None):
        assert add_or_set in {'add', 'set'}, f'\'add_or_set\' needs to be either add or set'

        intensity_map = intensity_map.clone()

        entity_coords = (entity != 0.).float()
        intensity_map *= entity_coords

        if avg_radius > 1:
            coords_radius = avg_radius // 2
            # if coords_kernel % 2 == 0:
            #     coords_kernel += 1

            # entity_coords_dilated = torch.nn.functional.max_pool3d(entity_coords[None, None, ...], kernel_size=coords_kernel, stride=1, padding=coords_kernel // 2).squeeze()
            # entity_coords_eroded = -torch.nn.functional.max_pool3d(-entity_coords[None, None, ...], kernel_size=coords_kernel, stride=1, padding=coords_kernel // 2).squeeze()
            entity_coords_dilated = Entity3D.binary_dilation(entity_coords, ('ball', coords_radius))
            entity_coords_eroded = Entity3D.binary_erosion(entity_coords, ('ball', coords_radius))
            intensity_map *= seg
            # if max_val is not None:
            #     intensity_map[ct_scan > max_val] = 0.
            intensity_map_avg = intensity_map.clone()
            if add_or_set == 'set':
                slcs = intensity_map == 0.
                intensity_map_avg[slcs] = ct_scan[slcs]

                del slcs
                torch.cuda.empty_cache()
                gc.collect()
            else:
                intensity_map_avg[intensity_map == 0.] = 0.
            # intensity_map_avg = torch.nn.functional.avg_pool3d(intensity_map_avg[None, None, ...], kernel_size=avg_kernel, stride=1, padding=avg_kernel // 2).squeeze()
            intensity_map_avg = Entity3D.average_pooling_3d(intensity_map_avg, ('ball', avg_radius))
            int_indic = (entity_coords_dilated * (1. - entity_coords_eroded)).bool()
            intensity_map[int_indic] = intensity_map_avg[int_indic]
        intensity_map *= seg

        if add_or_set == 'add':
            ct_scan += intensity_map
            # if max_val is not None:
            #     ct_scan[intensity_map != 0.] = torch.minimum(ct_scan[intensity_map != 0.], torch.tensor(max_val))
        else:
            ct_scan[intensity_map != 0.] = intensity_map[intensity_map != 0.]

        return ct_scan

    @staticmethod
    def random_deform(entity, deform_resolution, frequencies, intensities):
        if type(entity) == list:
            shape = entity[0].shape
        elif type(entity) == torch.Tensor:
            shape = entity.shape
        else:
            raise Exception('Wrong input type for random_deform')

        upsample_shape = min(shape[-3], shape[-2], shape[-1])
        deform_resize = torch.nn.Upsample(size=(shape[-3] // deform_resolution, shape[-2] // deform_resolution, shape[-1] // deform_resolution), mode='trilinear')

        x_freq = random.random() * frequencies[0][0] + frequencies[0][1]
        y_freq = random.random() * frequencies[1][0] + frequencies[1][1]
        z_freq = random.random() * frequencies[2][0] + frequencies[2][1]

        x_deform_intensity = random.random() * intensities[0][0] + intensities[0][1]
        y_deform_intensity = random.random() * intensities[1][0] + intensities[1][1]
        z_deform_intensity = random.random() * intensities[2][0] + intensities[2][1]

        grid_x = deform_resize((torch.rand((1, 1, int(upsample_shape // x_freq), int(upsample_shape // x_freq), int(upsample_shape // x_freq))) - 0.5) * x_deform_intensity).squeeze()
        grid_y = deform_resize((torch.rand((1, 1, int(upsample_shape // y_freq), int(upsample_shape // y_freq), int(upsample_shape // y_freq))) - 0.5) * y_deform_intensity).squeeze()
        grid_z = deform_resize((torch.rand((1, 1, int(upsample_shape // z_freq), int(upsample_shape // z_freq), int(upsample_shape // z_freq))) - 0.5) * z_deform_intensity).squeeze()

        # x_deform_intensity = torch.rand(shape[-3] // deform_resolution, shape[-2] // deform_resolution, shape[-1] // deform_resolution) * intensities[0][0] + intensities[0][1]
        # y_deform_intensity = torch.rand(shape[-3] // deform_resolution, shape[-2] // deform_resolution, shape[-1] // deform_resolution) * intensities[1][0] + intensities[1][1]
        # z_deform_intensity = torch.rand(shape[-3] // deform_resolution, shape[-2] // deform_resolution, shape[-1] // deform_resolution) * intensities[2][0] + intensities[2][1]

        # grid_x = deform_resize((torch.rand((1, 1, int(upsample_shape // x_freq), int(upsample_shape // x_freq), int(upsample_shape // x_freq))) - 0.5)).squeeze() * x_deform_intensity
        # grid_y = deform_resize((torch.rand((1, 1, int(upsample_shape // y_freq), int(upsample_shape // y_freq), int(upsample_shape // y_freq))) - 0.5)).squeeze() * y_deform_intensity
        # grid_z = deform_resize((torch.rand((1, 1, int(upsample_shape // z_freq), int(upsample_shape // z_freq), int(upsample_shape // z_freq))) - 0.5)).squeeze() * z_deform_intensity

        bspline = gryds.BSplineTransformationCuda([grid_x, grid_y, grid_z], order=1)

        if type(entity) == torch.Tensor:
            interpolator = gryds.BSplineInterpolatorCuda(entity.squeeze())

            torch.cuda.empty_cache()
            gc.collect()

            deformed_entity = torch.tensor(interpolator.transform(bspline)).to(DEVICE)
            return deformed_entity

        def_entities = []
        for c_ent in entity:
            interpolator = gryds.BSplineInterpolatorCuda(c_ent.squeeze())

            torch.cuda.empty_cache()
            gc.collect()

            def_entities.append(torch.tensor(interpolator.transform(bspline)).to(DEVICE))
        return def_entities

    @staticmethod
    def center_isotropic_deform_pleural_effusion(entity: torch.Tensor, lungs_cent, patient_mode, seg_name='', scan_name='', prev_params=None):
        def random_choose_grid_sides(grid, cent_v, t_size, intensity_r=0.17, intensity_b=0.08, lt_p=0.1, gt_p=0.05, grid_name=''):
            c_params = {}

            grid = grid.clone()
            grid = (grid - cent_v) / t_size
            if random.random() < 1 - lt_p:
                grid[grid < 0] = 0

                c_params[f'{params_prefix}_{grid_name}_neg_side'] = False
            else:
                intensity_lt = random.random() * intensity_r + intensity_b
                grid[grid < 0] *= intensity_lt

                c_params[f'{params_prefix}_{grid_name}_neg_side'] = True
                c_params[f'{params_prefix}_{grid_name}_neg_intensity'] = intensity_lt
            if random.random() < 1 - gt_p:
                grid[grid > 0] = 0

                c_params[f'{params_prefix}_{grid_name}_pos_side'] = False
            else:
                intensity_gt = random.random() * intensity_r + intensity_b
                grid[grid > 0] *= intensity_gt

                c_params[f'{params_prefix}_{grid_name}_pos_side'] = True
                c_params[f'{params_prefix}_{grid_name}_pos_intensity'] = intensity_gt
            return grid, c_params

        def mult_grid_at_random(grid, p=0.33, grid_name=''):
            c_params = {}

            if random.random() < p and torch.any(grid != 0):
                intensity_range = random.random() * 0.6
                freq = random.randint(12, 20)
                mult_grid = deform_resize(torch.rand((1, 1, int(shape[-3] // freq), int(shape[-3] // freq), int(shape[-3] // freq))) * (2 * intensity_range) + (1 - intensity_range)).squeeze()
                grid_to_use = grid * mult_grid

                c_params[f'{params_prefix}_{grid_name}_mult'] = True
                c_params[f'{params_prefix}_{grid_name}_mult_intensity_range'] = intensity_range
                c_params[f'{params_prefix}_{grid_name}_mult_frequency'] = freq
            else:
                grid_to_use = grid

                c_params[f'{params_prefix}_{grid_name}_mult'] = False
            return grid_to_use, c_params

        params_prefix = f'{scan_name}_{seg_name}'
        prev_params_prefix = f'prior_{seg_name}'

        def_params = {}

        entity = entity.clone().cpu()

        p_use_prev_params = 0.55
        if (prev_params is None) or (f'{prev_params_prefix}_grid_x' not in prev_params) or (random.random() > p_use_prev_params):
            cent = torch.argwhere(entity).float().mean(dim=0).round()
            lungs_cent = lungs_cent.cpu()

            if cent[1] < lungs_cent[1]:
                lt_w = 1
            else:
                lt_w = 0

            entity_mask = entity != 0
            entity_coords = entity.nonzero().T
            min_h, max_h = torch.min(entity_coords[0]), torch.max(entity_coords[0])
            min_w, max_w = torch.min(entity_coords[1]), torch.max(entity_coords[1])
            min_d, max_d = torch.min(entity_coords[2]), torch.max(entity_coords[2])
            entity_height = (max_h - min_h).cpu()
            entity_width = (max_w - min_w).cpu()
            entity_depth = (max_d - min_d).cpu()

            x = torch.arange(entity.shape[0])
            y = torch.arange(entity.shape[1])
            z = torch.arange(entity.shape[2])

            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

            # x = height
            # y = width
            # z = depth

            # patient_mode = random.choices(['supine', 'erect', 'semi-supine'], weights=[0.45, 0.1, 0.45])[0]

            def_params[f'{scan_name}_patient_position'] = patient_mode

            if patient_mode == 'supine':
                if random.random() < 0.02:
                    intensity = 100
                else:
                    intensity = (random.random() ** 2.5) * 0.65 + 0.1

                z_th_ind = random.random()
                if z_th_ind < 0.35:
                    intensity *= 2
                elif z_th_ind < 0.7:
                    cent[2] = min_d
                else:
                    th = random.random() * 0.75
                    cent[2] = (1 - th) * min_d + th * max_d
                    intensity /= 1 - th

                    def_params[f'{params_prefix}_supine_anchor_th'] = th

                grid_z = (grid_z - cent[2]) / entity_depth
                grid_z[grid_z < 0] = 0
                grid_z *= intensity

                grid_x, params = random_choose_grid_sides(grid_x, cent[0], entity_height, intensity_r=0.5, intensity_b=0.25, lt_p=0.04, gt_p=0.04, grid_name='height')

                def_params[f'{params_prefix}_supine_intensity'] = intensity
                def_params[f'{params_prefix}_supine_z_th_ind'] = z_th_ind
                def_params = def_params | params
            elif patient_mode == 'erect':
                intensity = ((random.random() * 0.75) ** 2) + 0.1

                x_th_ind = random.random()
                if x_th_ind < 0.5:
                    th = random.random() * 0.5
                    cent[0] = (1 - th) * min_h + th * max_h
                    intensity /= (1 - th)

                    def_params[f'{params_prefix}_erect_anchor_th'] = th
                else:
                    intensity *= 2

                grid_x = (grid_x - cent[0]) / (entity_height / 2)
                grid_x[grid_x < 0] = 0

                r = 0
                if random.random() < 0.5:
                    orig_shape = entity.squeeze().shape
                    downsample = torch.nn.Upsample(size=(entity.shape[-3] // 2, entity.shape[-2] // 2, entity.shape[-1] // 2))
                    upsample = torch.nn.Upsample(size=orig_shape)
                    r = random.randint(1, 8)
                    entity = downsample(entity.clone().to(DEVICE).squeeze()[None, None, ...]).squeeze()
                    entity = Entity3D.binary_opening(entity, ('ball', r))
                    entity = upsample(entity[None, None, ...]).squeeze()

                grid_x *= intensity
                grid_z = torch.zeros_like(grid_z)

                def_params[f'{params_prefix}_erect_intensity'] = intensity
                def_params[f'{params_prefix}_erect_opening_radius'] = r
                def_params[f'{params_prefix}_erect_x_th_ind'] = x_th_ind
            elif patient_mode == 'semi-supine':
                is_angle = random.random() < 0.2

                if is_angle:
                    intensity = 7
                    th = random.random() * 0.6 + 1.25
                    lower_part_coef = 10
                    upper_part_coef = 10
                else:
                    intensity = ((random.random() ** 2.75) * 2.5) + 0.1
                    th = 1.2 + (random.random() ** 1.75) * 0.6 * (random.randint(0, 1) * 2 - 1)  # Low = Cutoff in upper lung
                    lower_part_coef = random.randint(1, 40) * 0.25                       # High = More fluid
                    upper_part_coef = random.randint(2, 30) * 0.25                       # High = Less fluid

                cent[2] = min_d

                grid_x_norm = (grid_x - min_h) / entity_height
                grid_z = (grid_z - cent[2]) / entity_depth
                grid_z[grid_z < 0] = 0

                grid_x_norm[grid_x_norm < 0] = 0

                grid_x_norm = grid_x_norm * entity_mask
                idx = grid_x_norm[entity_mask]
                grid_x_norm[entity_mask] = (idx - torch.min(idx)) / (torch.max(idx) - torch.min(idx))
                grid_x_norm *= 2

                offset = 1 - th
                grid_x_norm[grid_x_norm > th] = torch.clamp_min(grid_x_norm + offset, 0)[grid_x_norm > th] ** lower_part_coef
                grid_x_norm[grid_x_norm < th] = torch.clamp_min(grid_x_norm + offset, 0)[grid_x_norm < th] ** upper_part_coef

                grid_z *= grid_x_norm
                grid_z *= intensity

                grid_x, params = random_choose_grid_sides(grid_x, cent[0], entity_height, intensity_r=0.5, intensity_b=0.25, lt_p=0.04, gt_p=0, grid_name='height')

                def_params = def_params | params
                def_params[f'{params_prefix}_semi-supine_intensity'] = intensity
                def_params[f'{params_prefix}_semi-supine_threshold'] = th
                def_params[f'{params_prefix}_semi-supine_lower_coef'] = lower_part_coef
                def_params[f'{params_prefix}_semi-supine_upper_coef'] = upper_part_coef
            else:
                raise Exception("Wadahek exception")

            grid_y, params = random_choose_grid_sides(grid_y, cent[1], entity_width, intensity_r=0.1, intensity_b=0.025, lt_p=0.07 * lt_w, gt_p=0.07 * (1 - lt_w), grid_name='width')

            def_params = def_params | params

            grid_x *= entity_mask
            grid_y *= entity_mask
            grid_z *= entity_mask

            shape = entity.shape
            deform_resize = torch.nn.Upsample(size=(shape[-3], shape[-2], shape[-1]), mode='trilinear')

            grid_x_to_use, params1 = mult_grid_at_random(grid_x, grid_name='height')
            grid_y_to_use, params2 = mult_grid_at_random(grid_y, grid_name='width')
            grid_z_to_use, params3 = mult_grid_at_random(grid_z, grid_name='depth')

            def_params = {**def_params, **params1, **params2, **params3}

            def_params[f'{params_prefix}_grid_x'] = grid_x
            def_params[f'{params_prefix}_grid_y'] = grid_y
            def_params[f'{params_prefix}_grid_z'] = grid_z
            def_params[f'{params_prefix}_grid_x_to_use'] = grid_x_to_use
            def_params[f'{params_prefix}_grid_y_to_use'] = grid_y_to_use
            def_params[f'{params_prefix}_grid_z_to_use'] = grid_z_to_use
        else:
            grid_x = prev_params[f'{prev_params_prefix}_grid_x']
            grid_y = prev_params[f'{prev_params_prefix}_grid_y']
            grid_z = prev_params[f'{prev_params_prefix}_grid_z']
            grid_x_to_use = prev_params[f'{prev_params_prefix}_grid_x_to_use']
            grid_y_to_use = prev_params[f'{prev_params_prefix}_grid_y_to_use']
            grid_z_to_use = prev_params[f'{prev_params_prefix}_grid_z_to_use']

        bspline = gryds.BSplineTransformationCuda([grid_x_to_use, grid_y_to_use, grid_z_to_use], order=1)
        inter_entity = entity.to(DEVICE).squeeze()
        interpolator = gryds.BSplineInterpolatorCuda(inter_entity)
        deformed_entity = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

        return deformed_entity, (grid_x, grid_y, grid_z), def_params

    @staticmethod
    def center_isotropic_deform_pneumothorax(entity: torch.Tensor, lungs_cent, patient_mode, seg_name='', scan_name='', prev_params=None, added_sulcus=False):
        def random_choose_grid_sides(grid, cent_v, t_size, intensity_r=0.17, intensity_b=0.08, lt_p=0.1, gt_p=0.05, intensity_exp=1., grid_name=''):
            c_params = {}

            grid = grid.clone()
            grid = (grid - cent_v) / t_size
            if random.random() < 1 - lt_p:
                grid[grid <= 0] = 0

                c_params[f'{params_prefix}_{grid_name}_neg_side'] = False
            else:
                intensity_lt = (random.random() ** intensity_exp) * intensity_r + intensity_b
                grid[grid <= 0] *= intensity_lt

                c_params[f'{params_prefix}_{grid_name}_neg_side'] = True
                c_params[f'{params_prefix}_{grid_name}_neg_intensity'] = intensity_lt
            if random.random() < 1 - gt_p:
                grid[grid > 0] = 0

                c_params[f'{params_prefix}_{grid_name}_pos_side'] = False
            else:
                intensity_gt = (random.random() ** intensity_exp) * intensity_r + intensity_b
                grid[grid > 0] *= intensity_gt

                c_params[f'{params_prefix}_{grid_name}_pos_side'] = True
                c_params[f'{params_prefix}_{grid_name}_pos_intensity'] = intensity_gt
            return grid, c_params

        def mult_grid_at_random(grid, p=0.33, intensity_range=0.2, freq_range=13, freq_bias=13, grid_name=''):
            c_params = {}

            if random.random() < p and torch.any(grid != 0):
                # intensity_range = random.random() * 0.75
                # freq = random.randint(12, 20)
                freq = random.randint(freq_bias, freq_bias + freq_range)
                mult_grid = deform_resize(torch.rand((1, 1, int(shape[-3] // freq), int(shape[-3] // freq), int(shape[-3] // freq))) * (2 * intensity_range) + (1 - intensity_range)).squeeze()
                grid_to_use = grid * mult_grid

                c_params[f'{params_prefix}_{grid_name}_mult'] = True
                c_params[f'{params_prefix}_{grid_name}_mult_intensity_range'] = intensity_range
                c_params[f'{params_prefix}_{grid_name}_mult_frequency'] = freq
            else:
                grid_to_use = grid

                c_params[f'{params_prefix}_{grid_name}_mult'] = False
            return grid_to_use, c_params

        def_params = {}
        params_prefix = f'{scan_name}_{seg_name}'
        prev_params_prefix = f'prior_{seg_name}'

        if patient_mode == 'supine' and added_sulcus and random.random() < 0.3:
            grid_x = torch.zeros_like(entity).cpu()
            grid_y = torch.zeros_like(entity).cpu()
            grid_z = torch.zeros_like(entity).cpu()
            grid_x_to_use = torch.zeros_like(entity).cpu()
            grid_y_to_use = torch.zeros_like(entity).cpu()
            grid_z_to_use = torch.zeros_like(entity).cpu()
            def_params[f'{params_prefix}_grid_x'] = grid_x
            def_params[f'{params_prefix}_grid_y'] = grid_y
            def_params[f'{params_prefix}_grid_z'] = grid_z
            def_params[f'{params_prefix}_grid_x_to_use'] = grid_x_to_use
            def_params[f'{params_prefix}_grid_y_to_use'] = grid_y_to_use
            def_params[f'{params_prefix}_grid_z_to_use'] = grid_z_to_use
            return entity.clone(), (grid_x, grid_y, grid_z), patient_mode, def_params

        entity = entity.clone().cpu()

        p_use_prev_params = 0.55
        if (prev_params is None) or (f'{prev_params_prefix}_grid_x' not in prev_params) or (random.random() > p_use_prev_params):
            cent = torch.argwhere(entity).float().mean(dim=0).round()
            lungs_cent = lungs_cent.cpu()

            if cent[1] < lungs_cent[1]:
                lt_w = 1
            else:
                lt_w = 0

            entity_mask = entity != 0
            entity_coords = entity.nonzero().T
            min_h, max_h = torch.min(entity_coords[0]), torch.max(entity_coords[0])
            min_w, max_w = torch.min(entity_coords[1]), torch.max(entity_coords[1])
            min_d, max_d = torch.min(entity_coords[2]), torch.max(entity_coords[2])
            entity_height = (max_h - min_h).cpu()
            entity_width = (max_w - min_w).cpu()
            entity_depth = (max_d - min_d).cpu()

            x = torch.arange(entity.shape[0])
            y = torch.arange(entity.shape[1])
            z = torch.arange(entity.shape[2])

            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

            # x = height
            # y = width
            # z = depth

            def_params[f'{scan_name}_patient_position'] = patient_mode
            if patient_mode == 'supine':
                use_grid_x_norm = random.random() < 0.5
                if use_grid_x_norm:
                    grid_x_norm = grid_x.clone()
                    grid_x_norm_cent_offset = -((random.random() ** 1.25) * 0.7 - 0.2)
                    grid_x_norm_mult = (random.random() ** 3) * 6. + 1.
                    grid_x_norm = (grid_x_norm - (cent[0] + (grid_x_norm_cent_offset * entity_width))) / entity_height
                    grid_x_norm *= grid_x_norm_mult
                    grid_x_norm[grid_x_norm < 0] = 0
                    use_x_lt = 0.65 if grid_x_norm_cent_offset < -0.25 else 0.25
                else:
                    use_x_lt = 0.65

                grid_x, params_x = random_choose_grid_sides(grid_x, lungs_cent[0], entity_height, intensity_r=0.3, intensity_b=0.05, lt_p=use_x_lt, gt_p=0.05, intensity_exp=3.25, grid_name='height')
                grid_y, params_y = random_choose_grid_sides(grid_y, lungs_cent[1], entity_width, intensity_r=0.3, intensity_b=0.05, lt_p=1. * lt_w, gt_p=1. * (1 - lt_w), intensity_exp=3.25, grid_name='width')
                grid_z, params_z = random_choose_grid_sides(grid_z, lungs_cent[2], entity_depth, intensity_r=0.0, intensity_b=0, lt_p=0., gt_p=0., grid_name='depth')

                if use_grid_x_norm:
                    grid_y *= grid_x_norm

                def_params = def_params | params_x
                def_params = def_params | params_y
                # def_params = def_params | params_z

                # #TODO: FIX
                # r = 0
                # if random.random() < 1:
                #     orig_shape = entity.squeeze().shape
                #     downsample = torch.nn.Upsample(size=(entity.shape[-3] // 2, entity.shape[-2] // 2, entity.shape[-1] // 2))
                #     upsample = torch.nn.Upsample(size=orig_shape)
                #     r = random.randint(3, 8)
                #     r = 4
                #     entity = downsample(entity.clone().to(DEVICE).squeeze()[None, None, ...]).squeeze()
                #     entity = Entity3D.binary_opening(entity, ('ball', r))
                #     entity = upsample(entity[None, None, ...]).squeeze()
                #
                # def_params[f'{params_prefix}_supine_opening_radius'] = r
            elif patient_mode == 'erect':
                use_grid_x_norm = random.random() < 0.5
                if use_grid_x_norm:
                    grid_x_norm = grid_x.clone()
                    grid_x_norm_cent_offset = (random.random() ** 1.25) * 0.7 - 0.2
                    grid_x_norm_mult = (random.random() ** 2.5) * 6. + 1.
                    grid_x_norm = (grid_x_norm - (cent[0] - (grid_x_norm_cent_offset * entity_width))) / entity_height
                    grid_x_norm *= grid_x_norm_mult
                    grid_x_norm[grid_x_norm > 0] = 0
                    grid_x_norm *= -1.

                grid_x, params_x = random_choose_grid_sides(grid_x, lungs_cent[0], entity_height, intensity_r=0.3, intensity_b=0.05, lt_p=1., gt_p=0, intensity_exp=3, grid_name='height')
                grid_y, params_y = random_choose_grid_sides(grid_y, lungs_cent[1], entity_width, intensity_r=0.3, intensity_b=0.05, lt_p=0.85 * lt_w, gt_p=0.85 * (1 - lt_w), intensity_exp=3, grid_name='width')
                grid_z, params_z = random_choose_grid_sides(grid_z, lungs_cent[2], entity_depth, intensity_r=0, intensity_b=0, lt_p=0., gt_p=0., grid_name='depth')

                if use_grid_x_norm:
                    grid_y *= grid_x_norm

                def_params = def_params | params_x
                def_params = def_params | params_y
                # def_params = def_params | params_z

                # r = 0
                # if random.random() < 1:
                #     orig_shape = entity.squeeze().shape
                #     downsample = torch.nn.Upsample(size=(entity.shape[-3] // 2, entity.shape[-2] // 2, entity.shape[-1] // 2))
                #     upsample = torch.nn.Upsample(size=orig_shape)
                #     r = random.randint(3, 8)
                #     entity = downsample(entity.clone().to(DEVICE).squeeze()[None, None, ...]).squeeze()
                #     entity = Entity3D.binary_opening(entity, ('ball', r))
                #     entity = upsample(entity[None, None, ...]).squeeze()
                #
                # def_params[f'{params_prefix}_erect_opening_radius'] = r
            else:
                raise Exception("Wadahek exception")

            # grid_y, params = random_choose_grid_sides(grid_y, cent[1], entity_width, intensity_r=0.1, intensity_b=0.025, lt_p=0.07, gt_p=0, grid_name='width')

            # def_params = def_params | params
            grid_x *= entity_mask
            grid_y *= entity_mask
            grid_z *= entity_mask

            shape = entity.shape
            deform_resize = torch.nn.Upsample(size=(shape[-3], shape[-2], shape[-1]), mode='trilinear')

            # grid_x_to_use, params1 = mult_grid_at_random(grid_x, p=1., intensity_range=0.2, freq_range=12, freq_bias=18, grid_name='height')
            # grid_y_to_use, params2 = mult_grid_at_random(grid_y, p=1., intensity_range=0.2, freq_range=12, freq_bias=18, grid_name='width')
            # grid_z_to_use, params3 = mult_grid_at_random(grid_z, p=0., grid_name='depth')

            grid_x_to_use = grid_x
            grid_y_to_use = grid_y
            grid_z_to_use = grid_z

            def_params = {**def_params}

            def_params[f'{params_prefix}_grid_x'] = grid_x
            def_params[f'{params_prefix}_grid_y'] = grid_y
            def_params[f'{params_prefix}_grid_z'] = grid_z
            def_params[f'{params_prefix}_grid_x_to_use'] = grid_x_to_use
            def_params[f'{params_prefix}_grid_y_to_use'] = grid_y_to_use
            def_params[f'{params_prefix}_grid_z_to_use'] = grid_z_to_use
        else:
            grid_x = prev_params[f'{prev_params_prefix}_grid_x']
            grid_y = prev_params[f'{prev_params_prefix}_grid_y']
            grid_z = prev_params[f'{prev_params_prefix}_grid_z']
            grid_x_to_use = prev_params[f'{prev_params_prefix}_grid_x_to_use']
            grid_y_to_use = prev_params[f'{prev_params_prefix}_grid_y_to_use']
            grid_z_to_use = prev_params[f'{prev_params_prefix}_grid_z_to_use']

        # def_params = {**def_params, **params1, **params2, **params3}

        bspline = gryds.BSplineTransformationCuda([grid_x_to_use, grid_y_to_use, grid_z_to_use], order=1)
        inter_entity = entity.to(DEVICE).squeeze()
        interpolator = gryds.BSplineInterpolatorCuda(inter_entity)
        deformed_entity = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

        # inter_vessels_seg = vessels_seg.to(DEVICE).squeeze() * inter_entity
        # interpolator_vessels = gryds.BSplineInterpolatorCuda(inter_vessels_seg)
        # deformed_vessels = torch.tensor(interpolator_vessels.transform(bspline)).to(DEVICE)

        return deformed_entity, (grid_x, grid_y, grid_z), patient_mode, def_params

    @staticmethod
    def add_air_bronchogram(cur_scan, bronchi_seg, orig_scan, boundary_seg):
        bronchi_seg *= boundary_seg
        # d_struct = ball(radius=11)
        # bronchi_seg = torch.tensor(binary_closing(bronchi_seg.cpu().numpy(), d_struct)).float().to(DEVICE)
        # bronchi_seg *= lungs_seg

        # cur_scan[bronchi_seg == 1.] = -1000.
        cur_scan[bronchi_seg == 1.] = orig_scan[bronchi_seg == 1.]

    @staticmethod
    def binary_dilation(nif: torch.Tensor, footprint_tuple):
        footprint_dict = {'ball': ball, 'octahedron': octahedron, 'rectangle': rectangle, 'square': square}
        footprint = footprint_dict[footprint_tuple[0]]

        orig_type = nif.dtype

        struct = torch.tensor(footprint(footprint_tuple[1]))[None, None, ...].float().to(nif.device)

        if footprint_tuple[0] == 'square':
            struct = struct[None, ...]

        dilated_nif = torch.nn.functional.conv3d(nif.float().squeeze()[None, None, ...], weight=struct, padding='same').squeeze().to(orig_type)
        dilated_nif[dilated_nif > 0] = 1

        return dilated_nif

    @staticmethod
    def binary_erosion(nif, footprint_tuple):
        eroded_nif = (1. - (Entity3D.binary_dilation(1. - nif.float(), footprint_tuple))).to(nif.dtype)

        return eroded_nif

    @staticmethod
    def binary_closing(nif, footprint_tuple):
        closed_nif = Entity3D.binary_dilation(nif, footprint_tuple)
        closed_nif = Entity3D.binary_erosion(closed_nif, footprint_tuple)

        return closed_nif

    @staticmethod
    def binary_opening(nif, footprint_tuple):
        opened_nif = Entity3D.binary_erosion(nif, footprint_tuple)
        opened_nif = Entity3D.binary_dilation(opened_nif, footprint_tuple)

        return opened_nif

    @staticmethod
    def average_pooling_3d(nif, footprint_tuple):
        footprint_dict = {'ball': ball, 'octahedron': octahedron, 'rectangle': rectangle}
        footprint = footprint_dict[footprint_tuple[0]]

        struct = torch.tensor(footprint(footprint_tuple[1]))[None, None, ...].float().to(nif.device)
        struct /= torch.sum(struct)

        dilated_nif = torch.nn.functional.conv3d(nif.squeeze()[None, None, ...], weight=struct, padding='same').squeeze()

        return dilated_nif

    @staticmethod
    def get_sep_lung_masks(seg: torch.Tensor):
        def determine_labels(c_seg, _init_call=True):
            c_label, _ = ndi.label(c_seg, structure=STRUCT3D)
            ccs, counts = np.unique(c_label, return_counts=True)
            num_ccs = len(ccs)
            w = ""

            if num_ccs == 3:
                return c_label, 1, 2, w
            else:
                if num_ccs > 3:
                    w = "Warning: More than 2 connectivity components found. Using 2 largest ones"
                    ocs = list(zip(ccs, counts))
                    sorted_ocs = sorted(ocs, key=lambda x: x[1], reverse=True)
                    return c_label, sorted_ocs[1][0], sorted_ocs[2][0], w
                elif num_ccs < 3:
                    if _init_call:
                        c_seg[:, c_seg.shape[1] // 2, :] = 0
                        return determine_labels(c_seg, _init_call=False)
                    w = "Warning: 1 connectivity component found. Using it twice"
                    return c_label, 1, 1, w

        orig_type = type(seg)
        if orig_type == torch.Tensor:
            dev = seg.device
            seg = seg.cpu().numpy()

        label, label1, label2, warning = determine_labels(seg.copy())
        if warning != "":
            print(warning)
        if '1' in warning:
            if orig_type == torch.Tensor:
                seg = torch.tensor(seg).float().to(dev)
                return seg, seg.clone(), warning
            return seg, seg.copy(), warning

        lung1 = label == label1
        lung2 = label == label2

        if orig_type == torch.Tensor:
            lung1 = torch.tensor(lung1).float().to(dev)
            lung2 = torch.tensor(lung2).float().to(dev)

        return lung1, lung2, warning

    @staticmethod
    def choose_random_lung_parts_for_pair(segs, same_parts_p=0.65, return_parameters=False):
        lungs_names = ['lungs', 'lung_left', 'lung_right']
        lobes_names = ['lung_right', 'lung_left', 'middle_right_lobe', 'upper_right_lobe', 'lower_right_lobe', 'upper_left_lobe', 'lower_left_lobe']

        def choose_random_lung_parts_image(lungs_p=0.3, additional_lobe_p=0.2):
            if random.random() < lungs_p:
                chosen_n = [random.choice(lungs_names)]
            else:
                lobe_n = random.choice(lobes_names)
                chosen_n = [lobe_n]

                c_lobes_n = lobes_names.copy()
                c_lobes_n.remove(lobe_n)
                while random.random() < additional_lobe_p and len(c_lobes_n) > 1:
                    lobe_n = random.choice(c_lobes_n)
                    chosen_n.append(lobe_n)
                    c_lobes_n.remove(lobe_n)

            return chosen_n

        both_chosen_n = choose_random_lung_parts_image()

        if random.random() < same_parts_p:
            prior_chosen_n = both_chosen_n
        else:
            prior_chosen_n = choose_random_lung_parts_image()
        if random.random() < same_parts_p:
            current_chosen_n = both_chosen_n
        else:
            current_chosen_n = choose_random_lung_parts_image()

        prior_chosen_seg = torch.zeros_like(segs['lungs'])
        current_chosen_seg = torch.zeros_like(segs['lungs'])
        both_chosen_seg = torch.zeros_like(segs['lungs'])

        for n in prior_chosen_n:
            prior_chosen_seg = torch.maximum(prior_chosen_seg, segs[n])
        for n in current_chosen_n:
            current_chosen_seg = torch.maximum(current_chosen_seg, segs[n])
        for n in both_chosen_n:
            both_chosen_seg = torch.maximum(both_chosen_seg, segs[n])

        if not return_parameters:
            return prior_chosen_seg, current_chosen_seg, both_chosen_seg
        return prior_chosen_seg, current_chosen_seg, both_chosen_seg, {"prior_chosen_regions": prior_chosen_n, "current_chosen_regions": current_chosen_n, "both_chosen_regions": both_chosen_n}

    @staticmethod
    def choose_lungs_randomly(lung1_1: torch.Tensor, lung1_2: torch.Tensor, lung2_1: torch.Tensor, lung2_2: torch.Tensor, same_p=0.75, return_parameters=False):
        each_same_p = same_p / 3
        each_non_same_p = (1 - same_p) / 12
        choice1, choice2 = random.choices([
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((0, 0), (1, 1)),
            ((0, 1), (0, 0)),
            ((0, 1), (1, 0)),
            ((0, 1), (0, 1)),
            ((0, 1), (1, 1)),
            ((1, 0), (0, 0)),
            ((1, 0), (0, 1)),
            ((1, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (0, 0)),
            ((1, 1), (0, 1)),
            ((1, 1), (1, 0)),
            ((1, 1), (1, 1)),
        ], weights=[each_non_same_p, each_non_same_p, each_non_same_p, each_non_same_p, each_non_same_p, each_same_p, each_non_same_p, each_non_same_p,
                    each_non_same_p, each_same_p, each_non_same_p, each_non_same_p, each_non_same_p, each_non_same_p, each_same_p])[0]
        params = {'lungs_chosen_prior': choice1, 'lungs_chosen_current': choice2}

        both_lungs1 = [lung1_1, lung1_2]
        both_lungs2 = [lung2_1, lung2_2]
        choices1, choices2 = [], []
        for i, ind in enumerate(choice1):
            if ind:
                choices1.append(both_lungs1[i])
        for i, ind in enumerate(choice2):
            if ind:
                choices2.append(both_lungs2[i])

        to_ret = [choices1, choices2, (choice1, choice2)]
        if return_parameters:
            to_ret.append(params)
        return to_ret

    @staticmethod
    def save_arr_as_nifti(arr, output_path):
        import nibabel as nib
        if type(arr) == torch.Tensor:
            arr = np.array(arr.cpu())
        if arr.ndim == 3:
            arr = np.flip(arr, axis=0)
            arr = np.transpose(arr, (1, 2, 0))
        affine = np.eye(4)
        xray_nif = nib.Nifti1Image(arr, affine)
        nib.save(xray_nif, output_path)