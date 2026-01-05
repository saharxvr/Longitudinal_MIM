import torch
from torch import Tensor
import gryds
from constants import DEVICE
from typing import Any
import gc
import random

from Entity3D import Entity3D


class Pneumothorax(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def get_sulcus_params():
        lower_part_mult = random.random() * 0.2 + 0.65
        add_mult = random.random() * 0.175 + 0.125
        x_intensity = random.random() * 0.089 + 0.91
        y_norm_exp = random.randint(10, 16) * 0.25
        z_norm_exp = random.randint(6, 8) * 0.25
        params = {'lower_part_mult': lower_part_mult, 'add_mult': add_mult, 'x_intensity': x_intensity, 'y_norm_exp': y_norm_exp, 'z_norm_exp': z_norm_exp}
        return params

    @staticmethod
    def add_deep_sulcus_sign(scan, lung_seg, lungs_seg_cent_w, params, r_prior=None):
        lung_seg = lung_seg.cpu()

        cent = torch.argwhere(lung_seg).float().mean(dim=0).round()

        seg_coords = lung_seg.nonzero().T
        min_h, max_h = torch.min(seg_coords[0]), torch.max(seg_coords[0])
        entity_height = max_h - min_h

        # Taking the bottom ~25% of seg and calculating measures
        lower_part_cutoff = torch.round(min_h + entity_height * params['lower_part_mult']).int()
        cropped_seg = lung_seg.clone()
        cropped_seg[:lower_part_cutoff] = 0
        cropped_seg_coords = cropped_seg.nonzero().T
        c_min_h, c_max_h = torch.min(cropped_seg_coords[0]), torch.max(cropped_seg_coords[0])
        c_min_w, c_max_w = torch.min(cropped_seg_coords[1]), torch.max(cropped_seg_coords[1])
        c_min_d, c_max_d = torch.min(cropped_seg_coords[2]), torch.max(cropped_seg_coords[2])
        c_entity_width = c_max_w - c_min_w
        c_entity_depth = c_max_d - c_min_d

        if cent[1] < lungs_seg_cent_w:
            relev_w = c_min_w
        else:
            relev_w = c_max_w

        angle_coord = torch.tensor([c_max_h, relev_w, c_max_d])

        h_add = entity_height * params['add_mult']

        x = torch.arange(lung_seg.shape[0])
        y = torch.arange(lung_seg.shape[1])
        z = torch.arange(lung_seg.shape[2])

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

        grid_x = (grid_x - (angle_coord[0] + h_add)) / entity_height
        grid_x[grid_x > 0] = 0
        grid_x = -grid_x + grid_x[angle_coord[0], angle_coord[1], angle_coord[2]]
        grid_x[:lower_part_cutoff, :, :] = 0
        grid_x -= torch.max(grid_x)
        grid_x[:lower_part_cutoff, :, :] = 0

        grid_x *= params['x_intensity']

        ######

        grid_y_norm = torch.zeros_like(grid_x)
        y_range = torch.linspace(0, 1, c_entity_width.item())
        if relev_w == c_min_w:
            y_range = torch.flip(y_range, dims=[0])
        y_range = y_range.repeat(grid_x.shape[-1], 1).T
        grid_y_norm[:, c_min_w: c_max_w, :] = y_range
        grid_y_norm = grid_y_norm ** params['y_norm_exp']

        ######

        grid_z_norm = torch.zeros_like(grid_x)
        z_range = torch.linspace(1, 0, c_entity_depth.item())
        grid_z_norm[:, :, c_min_d: c_max_d] = z_range
        grid_z_norm = grid_z_norm ** params['z_norm_exp']

        #####

        grid_x *= grid_y_norm
        grid_x *= grid_z_norm

        grid_y = torch.zeros_like(grid_x)
        grid_z = torch.zeros_like(grid_x)

        bspline = gryds.BSplineTransformationCuda([grid_x, grid_y, grid_z], order=1)
        lung_seg = lung_seg.to(DEVICE).squeeze()
        interpolator = gryds.BSplineInterpolatorCuda(lung_seg)
        deformed_lung_seg = torch.tensor(interpolator.transform(bspline)).to(DEVICE)
        deformed_lung_seg[deformed_lung_seg > 0] = 1.

        # grid_x *= deformed_lung_seg.cpu()

        bspline = gryds.BSplineTransformationCuda([grid_x, grid_y, grid_z], order=1)
        scan = scan.to(DEVICE).squeeze()
        interpolator = gryds.BSplineInterpolatorCuda(scan)
        deformed_scan = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

        # diff_seg = lung_seg * (1. - deformed_lung_seg)
        # deformed_scan[diff_seg == 1] = 0
        out_def_seg = deformed_lung_seg == 0
        deformed_scan[out_def_seg] = scan[out_def_seg]

        if r_prior is not None:
            r_prior = r_prior.to(DEVICE).squeeze()
            interpolator = gryds.BSplineInterpolatorCuda(r_prior)
            deformed_r_prior = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

            deformed_r_prior[out_def_seg] = scan[out_def_seg]
            # deformed_r_prior[diff_seg == 1] = 0

            return deformed_scan, deformed_lung_seg, deformed_r_prior

        return deformed_scan, deformed_lung_seg

    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> dict[str, Any]:
        def create_deformed_scan(segs_to_use, segs_choices, scan_idx, c_registrated_prior, prev_params=None):
            def apply_def_to_scan(c_scan):
                prev_deformed_scan = c_scan.clone().squeeze().to(DEVICE)
                for grids in grids_lst:
                    bspline = gryds.BSplineTransformationCuda(grids, order=1)
                    interpolator = gryds.BSplineInterpolatorCuda(prev_deformed_scan)
                    c_deformed_scan = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

                    # c_deformed_scan[msk_out] = prev_deformed_scan[msk_out]

                    prev_deformed_scan = c_deformed_scan.clone()

                del prev_deformed_scan
                prev_deformed_scan = None
                torch.cuda.empty_cache()
                gc.collect()

                return c_deformed_scan

            if segs_choices == (0, 0):
                deformed_scan = scans[scan_idx].clone().to(DEVICE)
                msk_in = torch.zeros_like(deformed_scan, device='cpu')
                sulcus_diff = torch.zeros_like(deformed_scan, device='cpu')
                to_ret = [deformed_scan, msk_in, sulcus_diff]
                if scan_idx == 1:
                    to_ret.append(c_registrated_prior)
                to_ret.append({})
                return to_ret

            scan = scans[scan_idx]
            lungs_seg = segs[scan_idx]['lungs']
            # vessels_seg = segs[scan_idx]['lung_vessels'].to(DEVICE)
            scan_name = 'prior' if scan_idx == 0 else 'current'

            lungs_seg_cent = torch.argwhere(lungs_seg).float().mean(dim=0).round().to(DEVICE)
            cent_w = lungs_seg_cent[1]

            def_seg = torch.zeros_like(scan).to(DEVICE)
            sulcus_diff = torch.zeros_like(scan).to(DEVICE)
            combined_seg = torch.zeros_like(scan).to(DEVICE)
            grids_lst = []

            c_sulcus_lung_params = [sulcus_lung_params[i] for i, c in enumerate(segs_choices) if c == 1]

            names = ['lung_left', 'lung_right']
            segs_names = [names[k] for k in range(len(segs_choices)) if segs_choices[k] == 1]

            c_log_params = {}
            for i, (c_seg, c_name) in enumerate(zip(segs_to_use, segs_names)):
                added_sulcus = False
                if patient_mode == 'supine' and random.random() < 0.65:
                    orig_c_seg = c_seg.clone()
                    if scan_idx == 0:
                        scan, c_seg = Pneumothorax.add_deep_sulcus_sign(scan, c_seg, cent_w, c_sulcus_lung_params[i])
                    else:
                        scan, c_seg, c_registrated_prior = Pneumothorax.add_deep_sulcus_sign(scan, c_seg, cent_w, c_sulcus_lung_params[i], r_prior=c_registrated_prior)
                    added_sulcus = True
                    c_sulcus_diff = c_seg.to(DEVICE) - orig_c_seg.to(DEVICE)
                    sulcus_diff = torch.maximum(sulcus_diff, c_sulcus_diff)
                    del orig_c_seg
                    del c_sulcus_diff
                combined_seg = torch.maximum(combined_seg, c_seg.to(DEVICE))
                c_def_seg, c_grids, prev_mode, c_params = Entity3D.center_isotropic_deform_pneumothorax(c_seg, patient_mode=patient_mode, seg_name=c_name, lungs_cent=lungs_seg_cent, scan_name=scan_name, prev_params=prev_params, added_sulcus=added_sulcus)

                c_def_seg[c_def_seg > 0] = 1.
                segs[scan_idx][c_name] = c_def_seg.squeeze().cpu()

                def_seg = torch.maximum(def_seg, c_def_seg)
                # def_vessels = torch.maximum(def_vessels, c_def_vessels)
                grids_lst.append(c_grids)

                c_log_params = c_log_params | c_params

            del c_def_seg
            sulcus_diff = sulcus_diff.cpu().float()
            torch.cuda.empty_cache()
            gc.collect()

            def_seg = def_seg.squeeze()

            # msk_out = ~(combined_seg.bool())
            msk_in = torch.logical_and(combined_seg.bool(), ~(def_seg.bool()))
            # msk_in = Pneumothorax.binary_erosion(msk_in, ('ball', 2))
            # msk_in = Pneumothorax.binary_dilation(msk_in, ('ball', 3))
            # msk_in = Pneumothorax.binary_opening(msk_in, ('ball', 2))

            # combined_seg = Pneumothorax.binary_dilation(combined_seg, ('ball', 1))
            # msk_in *= combined_seg.bool()

            deformed_scan = apply_def_to_scan(scan)
            # deformed_scan[def_vessels == 1] += 1000.
            if scan_idx == 1:
                c_registrated_prior = apply_def_to_scan(c_registrated_prior)
                # deformed_r_prior_scan[def_vessels == 1] += 1000.

            gap = torch.zeros_like(deformed_scan) - 950.
            msk_in = msk_in.float()

            # msk_in = Pneumothorax.fill_borders_gap(deformed_scan, msk_in, combined_seg, include_th=-400.)
            msk_in = Pneumothorax.binary_closing(msk_in, ('ball', 2))
            msk_in = Pneumothorax.binary_dilation(msk_in, ('ball', 2)).bool()

            deformed_scan[msk_in] = gap[msk_in]
            # deformed_scan[msk_in] = -1000

            if scan_idx == 1:
                c_registrated_prior[msk_in] = gap[msk_in]

            del gap
            gap = None
            torch.cuda.empty_cache()
            gc.collect()

            dilated_combined_seg = torch.maximum(Pneumothorax.binary_dilation(combined_seg, ('ball', 2)), msk_in)
            eroded_combined_seg = Pneumothorax.binary_erosion(combined_seg, ('ball', 1))
            outer_avg_region = dilated_combined_seg - eroded_combined_seg
            outer_avg_region = (outer_avg_region * msk_in) == 1
            avg_intensities = Pneumothorax.average_pooling_3d(deformed_scan, ('ball', 3))
            deformed_scan[outer_avg_region] = avg_intensities[outer_avg_region]

            if scan_idx == 1:
                c_registrated_prior[outer_avg_region] = avg_intensities[outer_avg_region]

            del dilated_combined_seg
            del eroded_combined_seg
            del outer_avg_region
            del avg_intensities
            torch.cuda.empty_cache()
            gc.collect()

            msk_in = msk_in.float()
            msk_in = msk_in.cpu()

            torch.cuda.empty_cache()
            gc.collect()

            erod_def_seg = Pneumothorax.binary_erosion(def_seg, ('ball', 1))
            # dial_def_seg = Pneumothorax.binary_dilation(def_seg, ('ball', 1))
            dial_def_seg = def_seg
            pleural_line_region_def_seg = torch.logical_and(dial_def_seg.bool(), ~(erod_def_seg.bool()))
            pleural_line_region_def_seg *= Pneumothorax.binary_erosion(combined_seg, ('ball', 2)).bool()

            del combined_seg
            combined_seg = None
            torch.cuda.empty_cache()
            gc.collect()

            pleural_line_deformed_scan = torch.zeros_like(deformed_scan)
            pleural_line_deformed_scan[pleural_line_region_def_seg] = 900
            pleural_line_deformed_scan = torch.nn.functional.avg_pool3d(pleural_line_deformed_scan[None, None, ...], kernel_size=3, stride=1, padding=1).squeeze()
            pleural_indic = pleural_line_deformed_scan != 0
            pleural_line_dec = random.random() * 200.
            deformed_scan[pleural_indic] = pleural_line_deformed_scan[pleural_indic] - 900 - pleural_line_dec

            if scan_idx == 1:
                c_registrated_prior[pleural_indic] = pleural_line_deformed_scan[pleural_indic] - 900 - pleural_line_dec

                # del avg_deformed_r_p_scan
                # avg_deformed_r_p_scan = None

            del pleural_line_region_def_seg
            del pleural_line_deformed_scan
            del pleural_indic
            pleural_line_region_def_seg = None
            pleural_line_deformed_scan = None
            pleural_indic = None

            # deformed_scan = deformed_scan.cpu()
            # if scan_idx == 1:
            #     c_registrated_prior = c_registrated_prior.cpu()

            # del avg_deformed_scan
            # avg_deformed_scan = None
            torch.cuda.empty_cache()
            gc.collect()

            segs[scan_idx]['lungs'] = torch.maximum(segs[scan_idx]['lung_right'], segs[scan_idx]['lung_right'])

            to_ret = [deformed_scan, msk_in, sulcus_diff]
            if scan_idx == 1:
                to_ret.append(c_registrated_prior)
            to_ret.append(c_log_params)

            return to_ret

        assert 'registrated_prior' in kwargs, 'Registrated prior has to be in kwargs'
        assert 'pleural_effusion_patient_mode' in kwargs, 'Pneumothorax requires a patient_mode parameter to be passed'

        registrated_prior = kwargs['registrated_prior']
        patient_mode = kwargs['patient_mode']

        # seg_prior_left = Pneumothorax.binary_erosion(segs[0]['lung_left'], ('ball', 1))
        # seg_prior_right = Pneumothorax.binary_erosion(segs[0]['lung_right'], ('ball', 1))
        # seg_current_left = Pneumothorax.binary_erosion(segs[1]['lung_left'], ('ball', 1))
        # seg_current_right = Pneumothorax.binary_erosion(segs[1]['lung_right'], ('ball', 1))

        seg_prior_left = segs[0]['lung_left']
        seg_prior_right = segs[0]['lung_right']
        seg_current_left = segs[1]['lung_left']
        seg_current_right = segs[1]['lung_right']

        if 'log_params' in kwargs:
            log_params = kwargs['log_params']
        else:
            log_params = False

        # Choose lung/s
        segs_to_use_prior, segs_to_use_current, choices, lung_params = Pneumothorax.choose_lungs_randomly(seg_prior_left, seg_prior_right, seg_current_left, seg_current_right, same_p=0.45, return_parameters=True)

        sulcus_lung_params = [Pneumothorax.get_sulcus_params(), Pneumothorax.get_sulcus_params()]

        pneumothorax_prior, prior_msk_in, prior_sulcus_seg, def_params_prior = create_deformed_scan(segs_to_use_prior, choices[0], 0, registrated_prior)
        pneumothorax_current, current_msk_in, current_sulcus_seg, pneumo_registrated_prior, def_params_current = create_deformed_scan(segs_to_use_current, choices[1], 1, registrated_prior, prev_params=def_params_prior)

        both_msk_in = current_msk_in - prior_msk_in
        both_sulcus_seg = current_sulcus_seg - prior_sulcus_seg

        ret_dict = {'scans': (pneumothorax_prior, pneumothorax_current), 'segs': segs, 'pneumothorax_seg': both_msk_in, 'pneumothorax_sulcus_seg': both_sulcus_seg, 'registrated_prior': pneumo_registrated_prior}

        if log_params:
            params_log = {**lung_params, **def_params_prior, **def_params_current}
            ret_dict['params'] = params_log

        return ret_dict
