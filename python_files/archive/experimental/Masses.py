from torch import Tensor
import torch
import random
from constants import DEVICE
import gc

from Entity3D import Entity3D


class Masses(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
        assert len(scans) == 2, "\'scans\' should contain only a prior and a current"
        assert len(segs) == 2, "\'segs\' should contain segmentations for both the prior and the current"
        assert 'lungs' in segs[0] and 'lungs' in segs[1], '\'Masses\' requires lungs segmentations for both the prior and the current'
        assert torch.equal(segs[0]['lungs'], segs[1]['lungs']), "\'Masses\' requires that prior and current have the same lung segmentations"

        prior = scans[0]
        current = scans[1]

        lungs_seg = segs[0]['lungs']

        num_nodules_prior = max(random.randint(1, 4), 0)
        num_nodules_current = max(random.randint(1, 4), 0)
        num_nodules_both = max(random.randint(1, 4), 0)

        # TODO: SHOULD NOT BE STATIC. Normalize by height of lung. should be arround [0.025 * h, 0.0715 * h]
        lungs_h = Masses.calc_lungs_height(lungs_seg)
        print(lungs_h) # TODO: MAKE SURE calc_lungs_height works properly
        radius_range = (int(0.025 * lungs_h), int(0.0715 * lungs_h))

        prior_masses = Masses.get_balls(lungs_seg, num_balls=num_nodules_prior, radius_range=radius_range)
        current_masses = Masses.get_balls(lungs_seg, num_balls=num_nodules_current, radius_range=radius_range)
        both_masses = Masses.get_balls(lungs_seg, num_balls=num_nodules_both, radius_range=radius_range)

        if torch.any(prior_masses != 0) or torch.any(current_masses != 0) or torch.any(both_masses != 0):
            combined_masses = prior_masses.clone()
            combined_masses[current_masses != 0.] = 2.
            combined_masses[both_masses != 0.] = 3.

            combined_masses = Masses.random_deform(combined_masses, deform_resolution=8, frequencies=((15, 15), (15, 15), (15, 15)), intensities=((0.05, 0.025), (0.05, 0.025), (0.05, 0.025))).squeeze()

            combined_masses = torch.round(combined_masses)

            prior_masses = torch.zeros_like(combined_masses)
            current_masses = torch.zeros_like(combined_masses)
            both_masses = torch.zeros_like(combined_masses)
            prior_masses[combined_masses == 1.] = 1.
            current_masses[combined_masses == 2.] = 1.
            both_masses[combined_masses == 3.] = 1.

            del combined_masses
            combined_masses = None

            torch.cuda.empty_cache()
            gc.collect()

            prior_masses *= lungs_seg
            current_masses *= lungs_seg
            both_masses *= lungs_seg

        prior_masses = torch.maximum(prior_masses, both_masses)
        current_masses = torch.maximum(current_masses, both_masses)

        del both_masses
        both_masses = None

        torch.cuda.empty_cache()
        gc.collect()

        inv_freq = random.randint(1, 3)
        intensity_map = Masses.get_intensity_map(lungs_seg, inv_freq, shift=180., scale=5.)

        prior = Masses.override_with_intensity(prior, prior_masses, intensity_map, avg_kernel=3)
        current = Masses.override_with_intensity(current, current_masses, intensity_map, avg_kernel=3)

        return prior, current