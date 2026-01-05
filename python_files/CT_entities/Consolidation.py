from torch import Tensor
import torch
import random
import gc
from typing import Any
from constants import DEVICE

from Entity3D import Entity3D


class Consolidation(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> dict[str, Any]:
        def choose_num_and_radiuses():
            num_and_radiuses_lst = [((80, 400), (0.02, 0.05)), ((25, 70), (0.02, 0.09)), ((1, 3), (0.08, 0.25)), ((1, 2), (0.12, 0.4))]

            idx_bo = random.randint(0, len(num_and_radiuses_lst) - 1)
            if random.random() < 0.65:
                idx_pr = idx_bo
            else:
                idx_pr = random.randint(0, len(num_and_radiuses_lst) - 1)
            if random.random() < 0.65:
                idx_cu = idx_bo
            else:
                idx_cu = random.randint(0, len(num_and_radiuses_lst) - 1)

            p_no_change = 0.35
            apply_options = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
            p_change = (1 - p_no_change) / (len(apply_options) - 1)
            apply = random.choices(apply_options, weights=[p_change, p_change, p_no_change, p_change, p_change, p_change, p_change], k=1)[0]
            determine = lambda apply_idx, lst_idx: num_and_radiuses_lst[lst_idx] if apply[apply_idx] else ((0, 0), (0, 0))

            (num_min_pr, num_max_pr), (radius_min_pr, radius_max_pr) = determine(0, idx_pr)
            (num_min_cu, num_max_cu), (radius_min_cu, radius_max_cu) = determine(1, idx_cu)
            (num_min_bo, num_max_bo), (radius_min_bo, radius_max_bo) = determine(2, idx_bo)

            num_nodules_pr = random.randint(num_min_pr, num_max_pr)
            num_nodules_cu = random.randint(num_min_cu, num_max_cu)
            num_nodules_bo = random.randint(num_min_bo, num_max_bo)

            radius_range_pr = (max(round(radius_min_pr * lungs_h), 1), max(round(radius_max_pr * lungs_h), 1))
            radius_range_cu = (max(round(radius_min_cu * lungs_h), 1), max(round(radius_max_cu * lungs_h), 1))
            radius_range_bo = (max(round(radius_min_bo * lungs_h), 1), max(round(radius_max_bo * lungs_h), 1))

            return (num_nodules_pr, num_nodules_cu, num_nodules_bo), (radius_range_pr, radius_range_cu, radius_range_bo)

        def choose_deformation_params():
            if random.random() < 0.6:
                def_params = [(((16, 10), (16, 10), (16, 10)), ((0.21, 0.07), (0.21, 0.07), (0.21, 0.07))), (((18, 3), (18, 3), (18, 3)), ((0.2, 0.06), (0.2, 0.06), (0.2, 0.06)))]
                freqs1, ints1 = random.choice(def_params)
                freqs2, ints2 = None, None
            else:
                freqs1, ints1 = ((14, 14), (14, 14), (14, 14)), ((0.18, 0.12), (0.18, 0.12), (0.18, 0.12))
                freqs2, ints2 = ((4, 1), (4, 1), (4, 1)), ((0.18, 0.06), (0.18, 0.06), (0.18, 0.06))
            return freqs1, ints1, freqs2, ints2

        assert len(scans) == 2, "\'scans\' should contain only a prior and a current"
        assert len(segs) == 2, "\'segs\' should contain segmentations for both the prior and the current"
        assert 'lungs' in segs[0] and 'lungs' in segs[1], '\'Consolidation\' requires lungs segmentations for both the prior and the current'
        assert torch.equal(segs[0]['lungs'], segs[1]['lungs']), "\'Consolidation\' requires that prior and current have the same lung segmentations"

        if 'log_params' in kwargs:
            log_params = kwargs['log_params']
        else:
            log_params = False
        if 'orig_scan' in kwargs:
            orig_scan = kwargs['orig_scan']
        else:
            orig_scan = scans[0].clone()

        prior = scans[0]
        current = scans[1]

        lungs_seg = segs[0]['lungs']
        lungs_h = Consolidation.calc_lungs_height(lungs_seg)
        bronchi_seg = segs[0]['bronchi']

        ### Initializing random parameters ###

        # Regions
        prior_boundary_seg, current_boundary_seg, both_boundary_seg, parts_params = Consolidation.choose_random_lung_parts_for_pair(segs[0], return_parameters=log_params)

        # Number of balls and radiuses
        (num_nodules_prior, num_nodules_current, num_nodules_both), (radius_range_prior, radius_range_current, radius_range_both) = choose_num_and_radiuses()

        # Deformations
        def_freqs_1, def_ints_1, def_freqs_2, def_ints_2 = choose_deformation_params()

        # Intensity map
        inv_freq = random.randint(2, 8)
        scale, shift = random.choice([(150, -30), (300, -300), (150, -300), (200, -150), (520, -400)])
        # scale = 150
        # shift = -30.

        # Blur radius
        prior_radius = random.randint(2, 6)
        current_radius = random.randint(2, 6)
        both_radius = random.randint(2, 6)

        # Edge blur
        # kernel_prior = random.randint(0, 3) * 2 + 3
        # kernel_current = random.randint(0, 3) * 2 + 3
        # kernel_both = random.randint(0, 3) * 2 + 3

        # Air bronchogram
        air_broncho_choices = random.choices([(0, 0), (0, 1), (1, 0), (1, 1)], weights=[0.46, 0.04, 0.04, 0.46])[0]

        ###

        prior_boundary_seg = prior_boundary_seg.to(DEVICE)
        current_boundary_seg = current_boundary_seg.to(DEVICE)
        both_boundary_seg = both_boundary_seg.to(DEVICE)

        prior_opacs = Consolidation.get_balls(prior_boundary_seg, num_balls=num_nodules_prior, radius_range=radius_range_prior)
        current_opacs = Consolidation.get_balls(current_boundary_seg, num_balls=num_nodules_current, radius_range=radius_range_current)
        both_opacs = Consolidation.get_balls(both_boundary_seg, num_balls=num_nodules_both, radius_range=radius_range_both)

        if torch.any(prior_opacs != 0) or torch.any(current_opacs != 0) or torch.any(both_opacs != 0):
            combined_opacs = prior_opacs.clone()
            combined_opacs[current_opacs != 0.] = 2.
            combined_opacs[both_opacs != 0.] = 3.

            combined_opacs = Consolidation.random_deform(combined_opacs, deform_resolution=8, frequencies=def_freqs_1, intensities=def_ints_1).squeeze()
            if def_freqs_2 is not None:
                combined_opacs = Consolidation.random_deform(combined_opacs, deform_resolution=8, frequencies=def_freqs_2, intensities=def_ints_2).squeeze()

            combined_opacs = torch.round(combined_opacs)

            prior_opacs = torch.zeros_like(combined_opacs)
            current_opacs = torch.zeros_like(combined_opacs)
            both_opacs = torch.zeros_like(combined_opacs)

            if num_nodules_prior > 0:
                prior_opacs[combined_opacs == 1.] = 1.
            if num_nodules_current > 0:
                current_opacs[combined_opacs == 2.] = 1.
            if num_nodules_both > 0:
                both_opacs[combined_opacs == 3.] = 1.

            del combined_opacs
            combined_opacs = None

            torch.cuda.empty_cache()
            gc.collect()

            prior_opacs *= prior_boundary_seg
            current_opacs *= current_boundary_seg
            both_opacs *= both_boundary_seg

        # prior_opacs = torch.maximum(prior_opacs, both_opacs)
        # current_opacs = torch.maximum(current_opacs, both_opacs)

        # del both_opacs
        # both_opacs = None

        # torch.cuda.empty_cache()
        # gc.collect()

        intensity_map = Consolidation.get_intensity_map(lungs_seg, inv_freq, scale, shift)

        prior_boundary_seg = Consolidation.fill_borders_gap(prior, prior_boundary_seg, prior_boundary_seg, include_th=shift)
        current_boundary_seg = Consolidation.fill_borders_gap(current, current_boundary_seg, current_boundary_seg, include_th=shift)
        both_boundary_seg = Consolidation.fill_borders_gap(prior, both_boundary_seg, both_boundary_seg, include_th=shift)

        prior = Consolidation.add_intensity(prior, prior_opacs, intensity_map, prior_boundary_seg, avg_radius=prior_radius, add_or_set='set')
        prior = Consolidation.add_intensity(prior, both_opacs, intensity_map, both_boundary_seg, avg_radius=both_radius, add_or_set='set')
        current = Consolidation.add_intensity(current, current_opacs, intensity_map, current_boundary_seg, avg_radius=current_radius, add_or_set='set')
        current = Consolidation.add_intensity(current, both_opacs, intensity_map, both_boundary_seg, avg_radius=both_radius, add_or_set='set')

        bronchi_seg = bronchi_seg.to(DEVICE)
        if air_broncho_choices[0]:
            Consolidation.add_air_bronchogram(prior, bronchi_seg, orig_scan=orig_scan.to(DEVICE), boundary_seg=prior_boundary_seg)
        if air_broncho_choices[1]:
            Consolidation.add_air_bronchogram(current, bronchi_seg, orig_scan=orig_scan.to(DEVICE), boundary_seg=current_boundary_seg)

        ret_dict = {'scans': (prior, current), 'segs': segs}

        ### Logging ###
        if log_params:
            params_log = {'Entity': 'Consolidation', 'num_nodules_prior': num_nodules_prior, 'num_nodules_current': num_nodules_current, 'num_nodules_both': num_nodules_both,
                          'radius_range_prior': radius_range_prior, 'radius_range_current': radius_range_current, 'radius_range_both': radius_range_both,
                          'def_freqs_1': def_freqs_1, 'def_ints_1': def_ints_1, 'def_freqs_2': def_freqs_2, 'def_ints_2': def_ints_2, 'inv_freq': inv_freq,
                          'prior_radius': prior_radius, 'current_radius': current_radius, 'both_radius': both_radius, 'air_broncho_choices': air_broncho_choices,
                          **parts_params}
            ret_dict['params'] = params_log

        return ret_dict
