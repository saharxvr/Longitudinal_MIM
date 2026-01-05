import torch
from torch import Tensor
import gryds
from constants import DEVICE
from typing import Any
import gc

from Entity3D import Entity3D


class PleuralEffusion(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> dict[str, Any]:
        def create_deformed_scan(segs_to_use, segs_choices, scan_idx, c_registrated_prior, prev_params=None):
            def apply_def_to_scan(c_scan):
                prev_deformed_scan = c_scan.clone().squeeze().to(DEVICE)
                for grids in grids_lst:
                    bspline = gryds.BSplineTransformationCuda(grids, order=1)
                    interpolator = gryds.BSplineInterpolatorCuda(prev_deformed_scan)
                    c_deformed_scan = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

                    c_deformed_scan[msk_out] = prev_deformed_scan[msk_out]

                    prev_deformed_scan = c_deformed_scan.clone()

                del prev_deformed_scan
                prev_deformed_scan = None
                torch.cuda.empty_cache()
                gc.collect()

                return c_deformed_scan

            if segs_choices == (0, 0):
                deformed_scan = scans[scan_idx].clone().to(DEVICE)
                msk_in = torch.zeros_like(deformed_scan, device='cpu')
                to_ret = [deformed_scan, msk_in]
                if scan_idx == 1:
                    to_ret.append(c_registrated_prior)
                to_ret.append({})
                return to_ret

            scan = scans[scan_idx]
            lungs_seg = segs[scan_idx]['lungs'].to(DEVICE)
            scan_name = 'prior' if scan_idx == 0 else 'current'

            lungs_seg_cent = torch.argwhere(lungs_seg).float().mean(dim=0).round()

            def_seg = torch.zeros_like(lungs_seg)
            grids_lst = []

            names = ['lung_left', 'lung_right']
            segs_names = [names[k] for k in range(len(segs_choices)) if segs_choices[k] == 1]

            c_log_params = {}
            for c_seg, c_name in zip(segs_to_use, segs_names):
                c_def_seg, c_grids, c_params = Entity3D.center_isotropic_deform_pleural_effusion(c_seg, lungs_seg_cent, patient_mode=patient_mode, seg_name=c_name, scan_name=scan_name, prev_params=prev_params)

                c_def_seg *= lungs_seg
                c_def_seg = torch.round(c_def_seg)
                segs[scan_idx][c_name] = c_def_seg.squeeze().cpu()

                def_seg = torch.maximum(def_seg, c_def_seg)
                grids_lst.append(c_grids)

                c_log_params = c_log_params | c_params

            # def_seg *= lungs_seg.to(DEVICE)
            # def_seg = def_seg.squeeze()
            # def_seg = torch.round(def_seg)

            combined_seg = torch.zeros_like(lungs_seg)
            for c_seg in segs_to_use:
                combined_seg = torch.maximum(combined_seg, c_seg.to(DEVICE))

            msk_out = (~(combined_seg.bool())).to(DEVICE)
            msk_in = torch.logical_and(combined_seg.bool().to(DEVICE), ~(def_seg.bool()))
            msk_in = PleuralEffusion.binary_dilation(msk_in, ('ball', 1))

            del combined_seg
            combined_seg = None
            torch.cuda.empty_cache()
            gc.collect()

            deformed_scan = apply_def_to_scan(scan)
            if scan_idx == 1:
                c_registrated_prior = apply_def_to_scan(c_registrated_prior)

            del msk_out
            msk_out = None
            torch.cuda.empty_cache()
            gc.collect()

            gap = torch.rand_like(deformed_scan) * 150. - 40.
            msk_in = msk_in.float()

            msk_in = PleuralEffusion.fill_borders_gap(scan, msk_in, msk_in).bool()
            deformed_scan[msk_in] = gap[msk_in]
            # deformed_scan[msk_in] = -1000

            if scan_idx == 1:
                c_registrated_prior[msk_in] = gap[msk_in]

            del gap
            gap = None
            torch.cuda.empty_cache()
            gc.collect()

            msk_in = msk_in.float()
            erod_msk_in = PleuralEffusion.binary_erosion(msk_in, ('ball', 1))
            dial_msk_in = PleuralEffusion.binary_dilation(msk_in, ('ball', 1))
            avg_region_msk_in = torch.logical_and(dial_msk_in.bool(), ~(erod_msk_in.bool()))

            msk_in = msk_in.cpu()

            # del msk_in
            del erod_msk_in
            del dial_msk_in
            # msk_in = None
            erod_msk_in = None
            dial_msk_in = None
            torch.cuda.empty_cache()
            gc.collect()

            avg_def_scan = PleuralEffusion.average_pooling_3d(deformed_scan, ('ball', 2))
            # avg_def_scan = torch.nn.functional.avg_pool3d(deformed_scan[None, None, ...], kernel_size=7, stride=1, padding=3).squeeze()
            deformed_scan[avg_region_msk_in] = avg_def_scan[avg_region_msk_in]

            if scan_idx == 1:
                avg_def_r_p_scan = PleuralEffusion.average_pooling_3d(c_registrated_prior, ('ball', 2))
                c_registrated_prior[avg_region_msk_in] = avg_def_r_p_scan[avg_region_msk_in]

                del avg_def_r_p_scan
                avg_def_r_p_scan = None

            del avg_def_scan
            del avg_region_msk_in
            avg_def_scan = None
            avg_region_msk_in = None
            torch.cuda.empty_cache()
            gc.collect()

            erod_def_seg = PleuralEffusion.binary_erosion(def_seg, ('ball', 1)).cpu()
            dial_def_seg = PleuralEffusion.binary_dilation(def_seg, ('ball', 1)).cpu()
            avg_region_def_seg = torch.logical_and(dial_def_seg.bool(), ~(erod_def_seg.bool())).to(DEVICE)

            avg_deformed_scan = PleuralEffusion.average_pooling_3d(deformed_scan, ('ball', 1))

            deformed_scan[avg_region_def_seg] = avg_deformed_scan[avg_region_def_seg]

            if scan_idx == 1:
                avg_deformed_r_p_scan = PleuralEffusion.average_pooling_3d(c_registrated_prior, ('ball', 1))
                c_registrated_prior[avg_region_def_seg] = avg_deformed_r_p_scan[avg_region_def_seg]

                del avg_deformed_r_p_scan
                avg_deformed_r_p_scan = None

            del avg_region_def_seg
            del avg_deformed_scan
            avg_deformed_scan = None
            avg_region_def_seg = None
            torch.cuda.empty_cache()
            gc.collect()

            segs[scan_idx]['lungs'] = torch.maximum(segs[scan_idx]['lung_right'], segs[scan_idx]['lung_left'])

            to_ret = [deformed_scan, msk_in]
            if scan_idx == 1:
                to_ret.append(c_registrated_prior)
            to_ret.append(c_log_params)

            return to_ret

        assert 'registrated_prior' in kwargs, 'Registrated prior has to be in kwargs'
        assert 'pleural_effusion_patient_mode' in kwargs, 'Pleural Effusion requires a pleural_effusion_patient_mode parameter to be passed'

        registrated_prior = kwargs['registrated_prior']
        patient_mode = kwargs['pleural_effusion_patient_mode']

        # seg_prior = segs[0]['lungs'].to(DEVICE)
        # seg_current = segs[1]['lungs'].to(DEVICE)

        seg_prior_left = segs[0]['lung_left']
        seg_prior_right = segs[0]['lung_right']
        seg_current_left = segs[1]['lung_left']
        seg_current_right = segs[1]['lung_right']

        if 'log_params' in kwargs:
            log_params = kwargs['log_params']
        else:
            log_params = False

        # Choose lung/s
        segs_to_use_prior, segs_to_use_current, choices, lung_params = PleuralEffusion.choose_lungs_randomly(seg_prior_left, seg_prior_right, seg_current_left, seg_current_right, same_p=0.75, return_parameters=True)

        segs_to_use_prior = [seg.to(DEVICE) for seg in segs_to_use_prior]
        segs_to_use_current = [seg.to(DEVICE) for seg in segs_to_use_current]

        effusion_prior, prior_msk_in, def_params_prior = create_deformed_scan(segs_to_use_prior, choices[0], 0, registrated_prior, prev_params=None)
        effusion_current, current_msk_in, effusion_registrated_prior, def_params_current = create_deformed_scan(segs_to_use_current, choices[1], 1, registrated_prior, prev_params=def_params_prior)

        # both_msk_in = torch.maximum(prior_msk_in, current_msk_in)
        both_msk_in = current_msk_in - prior_msk_in

        ret_dict = {'scans': (effusion_prior, effusion_current), 'segs': segs, 'pleural_effusion_seg': both_msk_in, 'registrated_prior': effusion_registrated_prior}

        if log_params:
            params_log = {**lung_params, **def_params_prior, **def_params_current}
            ret_dict['params'] = params_log

        return ret_dict
