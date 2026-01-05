from torch import Tensor
import torch
import random
import gc

from Entity3D import Entity3D


class LungOpacity(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
        assert len(scans) == 2, "\'scans\' should contain only a prior and a current"
        assert len(segs) == 2, "\'segs\' should contain segmentations for both the prior and the current"
        assert 'lungs' in segs[0] and 'lungs' in segs[1], '\'LungOpacity\' requires lungs segmentations for both the prior and the current'
        assert torch.equal(segs[0]['lungs'], segs[1]['lungs']), "\'LungOpacity\' requires that prior and current have the same lung segmentations"

        orig_scan = scans[0].clone()
        prior = scans[0]
        current = scans[1]

        lungs_seg = segs[0]['lungs']
        boundary_seg = segs[0]['middle_right_lobe']
        # lungs_seg = segs[0]['lower_right_lobe']

        # num_nodules_prior = random.randint(1, 2)
        # num_nodules_current = random.randint(1, 2)
        # num_nodules_both = random.randint(1, 2)
        num_nodules_prior = 100
        num_nodules_current = 0
        num_nodules_both = 0
        #TODO: ADD RANDOMNESS

        lungs_h = LungOpacity.calc_lungs_height(lungs_seg)
        # radius_range = (int(0.08 * lungs_h), int(0.5 * lungs_h))
        radius_range = (int(0.01 * lungs_h), int(0.05 * lungs_h))

        prior_opacs = LungOpacity.get_balls(boundary_seg, num_balls=num_nodules_prior, radius_range=radius_range)
        current_opacs = LungOpacity.get_balls(boundary_seg, num_balls=num_nodules_current, radius_range=radius_range)
        both_opacs = LungOpacity.get_balls(boundary_seg, num_balls=num_nodules_both, radius_range=radius_range)

        if torch.any(prior_opacs != 0) or torch.any(current_opacs != 0) or torch.any(both_opacs != 0):
            combined_opacs = prior_opacs.clone()
            combined_opacs[current_opacs != 0.] = 2.
            combined_opacs[both_opacs != 0.] = 3.

            # combined_opacs = LungOpacity.random_deform(combined_opacs, deform_resolution=8, frequencies=((30, 5), (30, 5), (30, 5)), intensities=((0.1, 0.035), (0.1, 0.035), (0.1, 0.035))).squeeze()
            combined_opacs = LungOpacity.random_deform(combined_opacs, deform_resolution=8, frequencies=((4, 1), (4, 1), (4, 1)), intensities=((0.2, 0.05), (0.2, 0.05), (0.2, 0.05))).squeeze()
            # combined_opacs = LungOpacity.random_deform(combined_opacs, deform_resolution=8, frequencies=((8, 2), (8, 2), (8, 2)), intensities=((0.2, 0.035), (0.2, 0.035), (0.2, 0.035))).squeeze()

            # combined_opacs = LungOpacity.random_deform(combined_opacs, deform_resolution=8, frequencies=((14, 14), (14, 14), (14, 14)), intensities=((0.17, 0.12), (0.17, 0.12), (0.17, 0.12))).squeeze()
            # combined_opacs = LungOpacity.random_deform(combined_opacs, deform_resolution=8, frequencies=((5, 1), (5, 1), (5, 1)), intensities=((0.15, 0.07), (0.15, 0.07), (0.15, 0.07))).squeeze()

            combined_opacs = torch.round(combined_opacs)

            prior_opacs = torch.zeros_like(combined_opacs)
            current_opacs = torch.zeros_like(combined_opacs)
            both_opacs = torch.zeros_like(combined_opacs)
            prior_opacs[combined_opacs == 1.] = 1.
            current_opacs[combined_opacs == 2.] = 1.
            both_opacs[combined_opacs == 3.] = 1.

            del combined_opacs
            combined_opacs = None

            torch.cuda.empty_cache()
            gc.collect()

            prior_opacs *= boundary_seg
            current_opacs *= boundary_seg
            both_opacs *= boundary_seg

        prior_opacs = torch.maximum(prior_opacs, both_opacs)
        current_opacs = torch.maximum(current_opacs, both_opacs)

        del both_opacs
        both_opacs = None

        torch.cuda.empty_cache()
        gc.collect()

        #TODO: ADD RANDOMNESS
        inv_freq = 2
        # scale = 500
        # shift = 50
        # scale = 150
        # shift = 550
        scale = 150
        shift = -30.
        intensity_map = LungOpacity.get_intensity_map(boundary_seg, inv_freq, scale, shift)

        kernel = 5
        border_filled_seg = LungOpacity.fill_borders_gap(prior, boundary_seg, boundary_seg, include_th=shift)
        prior = LungOpacity.add_intensity(prior, prior_opacs, intensity_map, border_filled_seg, avg_kernel=kernel, add_or_set='add')
        current = LungOpacity.add_intensity(current, current_opacs, intensity_map, border_filled_seg, avg_kernel=kernel, add_or_set='add')

        # bronchi_seg = segs[0]['bronchi']
        # LungOpacity.add_air_bronchogram(prior, bronchi_seg, orig_scan=orig_scan, lungs_seg=lungs_seg)

        return [prior, current], segs
