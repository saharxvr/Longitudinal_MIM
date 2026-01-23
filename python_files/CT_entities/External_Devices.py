import os

from torch import Tensor
import torch
import random
import gc
import operator
from typing import Any
import nibabel as nib
import math

from skimage.draw import line_nd
from skimage.morphology import ellipse
import numpy as np
from kornia.geometry import rotate3d

from constants import DEVICE
from Entity3D import Entity3D


class ExternalDevices(Entity3D):
    def __init__(self):
        Entity3D.__init__(self)

    @staticmethod
    def add_stickers(scan, lungs_seg, n=5, r_prior=None):
        stickers_map = torch.zeros_like(lungs_seg.cpu())

        dila_seg = ExternalDevices.binary_dilation(lungs_seg, ('ball', 1))
        seg_hull = torch.logical_and(dila_seg.bool(), ~(lungs_seg.bool()))
        seg_hull_coords = seg_hull.nonzero()

        seg_coords = lungs_seg.nonzero().T
        min_h, max_h = torch.min(seg_coords[0]), torch.max(seg_coords[0])
        min_w, max_w = torch.min(seg_coords[1]), torch.max(seg_coords[1])
        min_d, max_d = torch.min(seg_coords[2]), torch.max(seg_coords[2])
        entity_height = (max_h - min_h).item()
        entity_width = (max_w - min_w).item()
        entity_depth = (max_d - min_d).item()

        max_size_coef = 0.3
        max_h = int(entity_height * max_size_coef)
        max_w = int(entity_width * max_size_coef)
        max_dim = max(max_h, max_w)

        stickers_map = torch.nn.functional.pad(stickers_map, (max_dim, max_dim, max_dim, max_dim, max_dim, max_dim))
        padded_shape = stickers_map.shape

        cs = []

        # seg_hull_coords are in the unpadded coordinate system.
        # When writing into the padded stickers_map we must offset by +max_dim.
        seg_hull_coords_cpu = seg_hull_coords.detach().cpu()

        for _ in range(n):
            c = None
            sticker = None
            yaw = None
            pitch = None
            roll = None
            c_pad = None
            x0 = x1 = y0 = y1 = z0 = z1 = None
            # Retry sampling until the sticker fits inside the padded volume.
            for _try in range(40):
                c_candidate = seg_hull_coords_cpu[random.randint(0, seg_hull_coords_cpu.shape[0] - 1)]

                size_coef = random.random() * 0.008333 + 0.025
                d = random.randint(6, 10)

                h = int(entity_height * size_coef)
                w = int(entity_width * size_coef)
                sticker = torch.tensor(ellipse(w, h), dtype=stickers_map.dtype)
                sticker = sticker[..., None]
                sticker = sticker.repeat((1, 1, d))

                yaw = torch.tensor(random.random() * 30 - 15)
                pitch = torch.tensor(random.random() * 30 - 15)
                roll = torch.tensor(random.random() * 30 - 15)
                sticker = rotate3d(sticker[None, None, ...], yaw, pitch, roll).squeeze()

                c_pad = c_candidate + max_dim
                x0 = int(c_pad[0].item()) - (sticker.shape[0] // 2)
                x1 = x0 + sticker.shape[0]
                y0 = int(c_pad[1].item()) - (sticker.shape[1] // 2)
                y1 = y0 + sticker.shape[1]
                z0 = int(c_pad[2].item())
                z1 = z0 + sticker.shape[2]

                if x0 < 0 or y0 < 0 or z0 < 0:
                    continue
                if x1 > padded_shape[0] or y1 > padded_shape[1] or z1 > padded_shape[2]:
                    continue

                c = c_candidate
                # Keep the validated parameters
                c_pad = c_candidate + max_dim
                break

            if c is None:
                # Failed to place a valid sticker; skip this one.
                continue

            cs.append(c)

            # x0..z1 are already validated in the sampling loop

            if random.random() < 0.5:
                inner_mult = random.random() * 0.15 + 0.1
                inner_d = random.randint(12, 16)
                thickness = random.randint(2, 5) / 2

                inner_sticker = torch.tensor(ellipse(int(h * inner_mult), int(w * inner_mult)), dtype=stickers_map.dtype)
                inner_sticker = torch.nn.functional.pad(inner_sticker, (2, 2, 2, 2))
                dial_inner_sticker = inner_sticker.clone()[..., None]
                dial_inner_sticker = ExternalDevices.binary_dilation(dial_inner_sticker, ('ball', thickness))
                inner_sticker = dial_inner_sticker - inner_sticker
                inner_sticker = inner_sticker[..., None].repeat((1, 1, inner_d))
                inner_sticker = rotate3d(inner_sticker[None, None, ...], yaw, pitch, roll).squeeze()

                ix0 = int(c_pad[0].item()) - (inner_sticker.shape[0] // 2)
                ix1 = ix0 + inner_sticker.shape[0]
                iy0 = int(c_pad[1].item()) - (inner_sticker.shape[1] // 2)
                iy1 = iy0 + inner_sticker.shape[1]
                iz0 = int(c_pad[2].item())
                iz1 = iz0 + inner_sticker.shape[2]
                if ix0 >= 0 and iy0 >= 0 and iz0 >= 0 and ix1 <= padded_shape[0] and iy1 <= padded_shape[1] and iz1 <= padded_shape[2]:
                    stickers_map[ix0:ix1, iy0:iy1, iz0:iz1] = inner_sticker

            if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 <= padded_shape[0] and y1 <= padded_shape[1] and z1 <= padded_shape[2]:
                stickers_map[x0:x1, y0:y1, z0:z1] = sticker

        stickers_map = stickers_map[max_dim: -max_dim, max_dim: -max_dim, max_dim: -max_dim]

        avg_r = random.randint(4, 5) / 2
        stickers_map = ExternalDevices.average_pooling_3d(stickers_map, ('ball', avg_r))
        stickers_map *= 10000
        stickers_map = stickers_map.to(scan.device)
        scan[stickers_map > 0] = stickers_map[stickers_map > 0]
        if r_prior is not None:
            r_prior[stickers_map > 0] = stickers_map[stickers_map > 0]
            return scan, r_prior, cs

        return scan, cs

    @staticmethod
    def add_cables(scan, lungs_seg, p1s, num_cables=None, r_prior=None):
        assert (p1s is not None) or (num_cables is not None)

        edge = torch.zeros_like(lungs_seg)
        edge_depth = edge.shape[-1]
        edge[..., :, 0, edge_depth // 4: 3 * edge_depth // 4] = 1.
        edge[..., :, -1, edge_depth // 4: 3 * edge_depth // 4] = 1.
        edge[..., 0, :, edge_depth // 4: 3 * edge_depth // 4] = 1.
        edge[..., -1, :, edge_depth // 4: 3 * edge_depth // 4] = 1.
        edge_coords = edge.nonzero()

        lines_seg = np.zeros_like(lungs_seg.cpu().numpy())

        if p1s is not None:
            for p1 in p1s:
                p2 = edge_coords[random.randint(0, edge_coords.shape[0] - 1)]
                l = line_nd(p1.cpu(), p2.cpu())
                lines_seg[l] = 1.
        else:
            for i in range(num_cables):
                p1 = edge_coords[random.randint(0, edge_coords.shape[0] - 1)]
                p2 = edge_coords[random.randint(0, edge_coords.shape[0] - 1)]
                l = line_nd(p1.cpu(), p2.cpu())
                lines_seg[l] = 1.

        lines_seg = torch.tensor(lines_seg).to(DEVICE)

        deform_indic = random.random()
        if deform_indic < 0.35:
            lines_seg = ExternalDevices.random_deform(lines_seg, deform_resolution=8, frequencies=((20, 35), (20, 35), (20, 35)), intensities=((0.05, 0.125), (0.05, 0.125), (0.05, 0.025))).squeeze()
        elif deform_indic < 0.7:
            lines_seg = ExternalDevices.random_deform(lines_seg, deform_resolution=8, frequencies=((25, 55), (25, 55), (25, 55)), intensities=((0.15, 0.2), (0.15, 0.2), (0.05, 0.075))).squeeze()
        else:
            lines_seg = ExternalDevices.random_deform(lines_seg, deform_resolution=8, frequencies=((35, 20), (35, 20), (35, 20)), intensities=((0.125, 0.225), (0.125, 0.225), (0.05, 0.075))).squeeze()

        th = 0
        lines_seg[lines_seg > th] = 1.
        lines_seg[lines_seg <= th] = 0.
        # lines_seg = ILD.binary_closing(lines_seg, ('ball', 2))
        dilate_r = random.randint(8, 10) / 2
        lines_seg = ExternalDevices.binary_dilation(lines_seg, ('ball', dilate_r))
        lines_seg = ExternalDevices.binary_erosion(lines_seg, ('ball', 3.5))
        # lines_seg = ExternalDevices.binary_closing(lines_seg, ('ball', 1))
        lines_seg = ExternalDevices.average_pooling_3d(lines_seg, ('ball', 2.5))
        lines_seg *= 10000
        scan[lines_seg > 0] = lines_seg[lines_seg > 0]
        if r_prior is not None:
            r_prior[lines_seg > 0] = lines_seg[lines_seg > 0]
            return scan, r_prior, p1s

        return scan, p1s

    @staticmethod
    def add_device(scan, lungs_seg, dev_p, registrated_prior=None):
        dev = torch.tensor(np.transpose(nib.load(dev_p).get_fdata(), (2, 0, 1))).float().to(DEVICE)
        max_shape = max(dev.shape[0], dev.shape[1], dev.shape[2])
        max_shape_x2 = 2 * max_shape

        scan = torch.nn.functional.pad(scan, (max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2))
        lungs_seg = torch.nn.functional.pad(lungs_seg, (max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2))
        if registrated_prior is not None:
            registrated_prior = torch.nn.functional.pad(registrated_prior, (max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2, max_shape_x2))

        dila_seg = ExternalDevices.binary_dilation(lungs_seg, ('ball', 1))
        seg_hull = torch.logical_and(dila_seg.bool(), ~(lungs_seg.bool()))
        seg_hull_coords = seg_hull.nonzero()

        # seg_coords = lungs_seg.nonzero().T
        # min_h, max_h = torch.min(seg_coords[0]), torch.max(seg_coords[0])
        # min_w, max_w = torch.min(seg_coords[1]), torch.max(seg_coords[1])
        # min_d, max_d = torch.min(seg_coords[2]), torch.max(seg_coords[2])
        # entity_height = (max_h - min_h).item()
        # entity_width = (max_w - min_w).item()
        # entity_depth = (max_d - min_d).item()

        c = seg_hull_coords[random.randint(0, seg_hull_coords.shape[0] - 1)]

        if random.random() < 0.5:
            inv_freq = random.randint(1, 3)
            shift = 1 - random.random() * 0.3
            scale = (1 - shift) * 2
            int_map = ExternalDevices.get_intensity_map(dev, inv_freq=inv_freq, scale=scale, shift=shift)
            dev = dev * int_map

        if random.random() < 0.5:
            if 'pacemaker' in dev_p and random.random() < 0.5:
                coef_x = random.random() * 0.25 + 0.75
                coef_y = coef_x
                coef_z = coef_x
            else:
                coef_x = random.random() * 0.25 + 0.75
                coef_y = random.random() * 0.25 + 0.75
                coef_z = random.random() * 0.25 + 0.75
            downsample = torch.nn.Upsample(size=(int(dev.shape[0] * coef_x), int(dev.shape[1] * coef_y), int(dev.shape[2] * coef_z)))
            dev = downsample(dev[None, None, ...]).squeeze()

        dev = torch.nn.functional.pad(dev, (max_shape, max_shape, max_shape, max_shape, max_shape, max_shape))

        # yaw -> around depth axis
        # pitch -> around width axis
        # roll -> around height axis

        yaw = torch.tensor(random.random() * 360 - 180).to(DEVICE)
        pitch = torch.tensor(random.random() * 40 - 20).to(DEVICE)
        roll = torch.tensor(random.random() * 40 - 20).to(DEVICE)
        dev = rotate3d(dev[None, None, ...], yaw, pitch, roll).squeeze()

        port_x_min, port_x_max = c[0] - dev.shape[0] // 2, c[0] + dev.shape[0] - dev.shape[0] // 2
        port_y_min, port_y_max = c[1] - dev.shape[1] // 2, c[1] + dev.shape[1] - dev.shape[1] // 2
        port_z_min, port_z_max = c[2] - dev.shape[2] // 2, c[2] + dev.shape[2] - dev.shape[2] // 2

        dev_copy = dev.clone()

        dev_coords = (dev != 0).nonzero()
        wire_coord = dev_coords[random.randint(0, dev_coords.shape[0] - 1)]

        dev[dev == 0] = scan[port_x_min: port_x_max, port_y_min: port_y_max, port_z_min: port_z_max][dev == 0]
        scan[port_x_min: port_x_max, port_y_min: port_y_max, port_z_min: port_z_max] = dev
        scan = scan[max_shape_x2: -max_shape_x2, max_shape_x2: -max_shape_x2, max_shape_x2: -max_shape_x2]

        if registrated_prior is not None:
            dev_copy[dev_copy == 0] = registrated_prior[port_x_min: port_x_max, port_y_min: port_y_max, port_z_min: port_z_max][dev_copy == 0]
            registrated_prior[port_x_min: port_x_max, port_y_min: port_y_max, port_z_min: port_z_max] = dev_copy
            registrated_prior = registrated_prior[max_shape_x2: -max_shape_x2, max_shape_x2: -max_shape_x2, max_shape_x2: -max_shape_x2]

        wire_coord = torch.tensor([
            min((wire_coord[0] + port_x_min - max_shape_x2).item(), scan.shape[0] - 1),
            min((wire_coord[1] + port_y_min - max_shape_x2).item(), scan.shape[1] - 1),
            min((wire_coord[2] + port_z_min - max_shape_x2).item(), scan.shape[2] - 1)
        ], dtype=torch.float32)

        if registrated_prior is not None:
            return scan, [wire_coord], registrated_prior
        return scan, [wire_coord]

    @staticmethod
    def add_to_CT_pair(scans: list[Tensor], segs: list[dict[str, Tensor]], *args, **kwargs) -> dict[str, Any]:
        if 'log_params' in kwargs:
            log_params = kwargs['log_params']
        else:
            log_params = False

        prior = scans[0]
        current = scans[1]
        registrated_prior = kwargs['registrated_prior']

        lungs_seg = segs[0]['lungs']

        devices_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/MedicalDevices'

        choices = random.choice([(1, 0), (0, 1), (1, 1)])

        if choices[0] == 1:
            add_wires = 0
            add_stickers = 0
            add_devices = 0
            flags = [add_wires, add_stickers, add_devices]
            while sum(flags) == 0:
                add_wires = random.random() < 0.4
                add_stickers = random.random() < 0.4
                add_devices = random.random() < 0.3
                flags = [add_wires, add_stickers, add_devices]

            if add_wires:
                num_cables = int((random.random() ** 2.5) * 8 + 1)
                prior, _ = ExternalDevices.add_cables(prior, lungs_seg, p1s=None, num_cables=num_cables, r_prior=None)
            if add_stickers:
                num_stickers = int((random.random() ** 1.5) * 6 + 1)
                prior, p1s = ExternalDevices.add_stickers(prior, lungs_seg, n=num_stickers, r_prior=None)
                if random.random() < 0.85:
                    prior, _ = ExternalDevices.add_cables(prior, lungs_seg, p1s, r_prior=None)
            if add_devices:
                num_devices = math.ceil((random.random() ** 3) * 3)
                list_devs = os.listdir(devices_p)
                random.shuffle(list_devs)
                for i in range(num_devices):
                    dev_name = list_devs[i]
                    dev_p = f'{devices_p}/{dev_name}'
                    prior, dev_p1 = ExternalDevices.add_device(prior, lungs_seg, dev_p, registrated_prior=None)
                    if random.random() < 0.6:
                        prior, _ = ExternalDevices.add_cables(prior, lungs_seg, p1s=dev_p1, num_cables=None, r_prior=None)

        if choices[1] == 1:
            add_wires = 0
            add_stickers = 0
            add_devices = 0
            flags = [add_wires, add_stickers, add_devices]
            while sum(flags) == 0:
                add_wires = random.random() < 0.4
                add_stickers = random.random() < 0.4
                add_devices = random.random() < 0.3
                flags = [add_wires, add_stickers, add_devices]

            if add_wires:
                num_cables = int((random.random() ** 2.5) * 8 + 1)
                current, registrated_prior, _ = ExternalDevices.add_cables(current, lungs_seg, p1s=None, num_cables=num_cables, r_prior=registrated_prior)
            if add_stickers:
                num_stickers = int((random.random() ** 1.5) * 6 + 1)
                current, registrated_prior, p1s = ExternalDevices.add_stickers(current, lungs_seg, n=num_stickers, r_prior=registrated_prior)
                if random.random() < 0.85:
                    current, registrated_prior, _ = ExternalDevices.add_cables(current, lungs_seg, p1s, r_prior=registrated_prior)
            if add_devices:
                num_devices = math.ceil((random.random() ** 3) * 3)
                list_devs = os.listdir(devices_p)
                random.shuffle(list_devs)
                for i in range(num_devices):
                    dev_name = list_devs[i]
                    dev_p = f'{devices_p}/{dev_name}'
                    current, dev_p1, registrated_prior = ExternalDevices.add_device(current, lungs_seg, dev_p, registrated_prior=registrated_prior)
                    if random.random() < 0.6:
                        current, registrated_prior, _ = ExternalDevices.add_cables(current, lungs_seg, p1s=dev_p1, num_cables=None, r_prior=registrated_prior)

        ret_dict = {'scans': (prior, current), 'segs': segs, 'registrated_prior': registrated_prior}

        if log_params:
            params_log = {}
            ret_dict['params'] = params_log

        return ret_dict
