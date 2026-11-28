"""Render supervoxel-change scenes to DRR pairs with exact per-change signed GT maps.

Mirrors the existing DRR pipeline (python_files/CT_entities/DRR_generator.py):
rotate the cropped CT, crop to the rotated lung bounding box (z/x, ext=7), ray-sum along
the AP axis after clamping to -1000 HU, resize to 512x512, and store raw projections
(the dataset applies min-max normalization at load time). The GT map of a change is its
own projected HU delta divided by the lung-footprint maximum of the current-frame base
projection; this is the linearization of ``calculate_diff_map`` (same 0.015 and 50-pixel
cleanup), so prior/current images and per-change GT are consistent by construction.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

OUTPUT_SIZE = 512
DEFAULT_ROTATION_PARAMS = (17.5, 37.5, 0.0, 1.75)
GT_THRESHOLD = 0.015
MIN_COMPONENT = 50
EIGHT_CONNECTED = np.ones((3, 3), int)


def sample_angles(rng, params=DEFAULT_ROTATION_PARAMS):
    """Same distribution as CT_Rotations.get_random_rotation_angles (degrees)."""
    per_axis, max_sum, min_sum, exponent = params
    if per_axis == 0:
        return (0.0, 0.0, 0.0)
    for _ in range(10000):
        angles = [(rng.random() ** exponent) * (1 if rng.random() < 0.5 else -1) * per_axis
                  for _ in range(3)]
        total = sum(abs(angle) for angle in angles)
        if min_sum <= total <= max_sum:
            return tuple(float(angle) for angle in angles)
    raise ValueError("Could not sample rotation angles within the requested sum range")


def _rotation_matrix(angles):
    a, b, c = (math.radians(angle) for angle in angles)
    rz = torch.tensor([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = torch.tensor([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rx = torch.tensor([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return (rz @ ry @ rx).float()


def _theta(shape, angles):
    """affine_grid theta [3, 4] of a rotation about the volume center in voxel units (anisotropic-safe)."""
    depth, height, width = shape
    rotation = _rotation_matrix(angles)  # acts on (z, x, y) voxel offsets
    half = torch.tensor([depth / 2, height / 2, width / 2])
    inverse = rotation.T
    # affine_grid coordinates are ordered (x=width, y=height, z=depth) and normalized.
    order = [2, 1, 0]
    theta = torch.zeros(3, 4)
    for row_out, axis_out in enumerate(order):
        for column, axis_in in enumerate(order):
            theta[row_out, column] = inverse[axis_out, axis_in] * half[axis_in] / half[axis_out]
    return theta


def sampling_grid(shape, angles, device):
    """Full-volume grid (reference implementation; FrameRenderer builds the grid in chunks)."""
    depth, height, width = shape
    return F.affine_grid(_theta(shape, angles)[None].to(device), (1, 1, depth, height, width), align_corners=False)


def rotate(volume, grid, padding="zeros"):
    tensor = torch.as_tensor(volume, dtype=torch.float32, device=grid.device)[None, None]
    return F.grid_sample(tensor, grid, mode="bilinear", padding_mode=padding,
                         align_corners=False)[0, 0]


def lung_crop(rotated_lungs, ext=7):
    coordinates = torch.nonzero(rotated_lungs > 0.5)
    if not len(coordinates):
        raise ValueError("Rotated lung mask is empty")
    low, high = coordinates.min(0).values, coordinates.max(0).values
    return (slice(max(int(low[0]) - ext, 0), min(int(high[0]) + ext, rotated_lungs.shape[0])),
            slice(max(int(low[1]) - ext, 0), min(int(high[1]) + ext, rotated_lungs.shape[1])))


def project(rotated, crop, offset=0.0):
    image = (rotated[crop] + offset).sum(-1)
    return _resize(image)


def _resize(image):
    return F.interpolate(image[None, None], size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear",
                         antialias=True, align_corners=False)[0, 0]


def _normalized(size, start, stop, device):
    """align_corners=False normalized coordinates of voxel indices start..stop-1."""
    return (2.0 * torch.arange(start, stop, device=device, dtype=torch.float32) + 1.0) / size - 1.0


CHUNK_VOXELS = 1 << 22  # output voxels sampled at once (grid ~50 MB)


def gpu_memory_gb(device):
    """Peak torch allocation since the last reset plus memory held by cupy's pools (gryds); None on CPU."""
    import sys

    if torch.device(device).type != "cuda":
        return None
    total = torch.cuda.max_memory_allocated(device)
    if "cupy" in sys.modules:
        total += sys.modules["cupy"].get_default_memory_pool().total_bytes()
    return total / 2 ** 30


def release_gpu(device):
    """Return cached torch / cupy blocks to the driver so concurrent workers can use them. gc first:
    arrays in reference cycles (e.g. cupy memory shared with torch by the device insertion) are only
    freed by the cyclic collector."""
    import gc
    import sys

    if torch.device(device).type != "cuda":
        return
    gc.collect()
    if "cupy" in sys.modules:
        sys.modules["cupy"].get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def gpu_held_gb(device):
    """GPU memory this process still holds (torch reserved + cupy pool), GB; None on CPU."""
    import sys

    if torch.device(device).type != "cuda":
        return None
    held = torch.cuda.memory_reserved(device)
    if "cupy" in sys.modules:
        held += sys.modules["cupy"].get_default_memory_pool().total_bytes()
    return held / 2 ** 30


class FrameRenderer:
    """Renders several volumes with one rotation and one lung crop (one camera frame).

    Equivalent to rotating the whole volume with ``sampling_grid`` and ray-summing the lung crop,
    but the grid is generated in depth chunks inside the crop only, and a sparse change
    (VolumeDelta) is sampled from its own box over the output block it can reach. Peak memory is
    one chunk instead of a full-volume grid, and changes cost time in proportion to their size.
    """

    def __init__(self, lungs, angles, device="cpu", lungs_tensor=None):
        self.device = torch.device(device)
        self.shape = tuple(int(size) for size in lungs.shape)
        theta = _theta(self.shape, angles)
        self.theta = theta.to(self.device)
        # 3x3 inverse on the CPU: a GPU inverse needs a cuSOLVER handle, which fails under memory pressure.
        self.inverse = torch.linalg.inv(theta[:, :3]).to(self.device)
        self.angles = tuple(angles)
        volume = (lungs_tensor if lungs_tensor is not None
                  else torch.as_tensor(np.asarray(lungs, np.float32), device=self.device))
        depth, height, width = self.shape
        hit, footprint = [], []
        for rows in self._chunks(0, depth, height * width):
            sampled = self._sample(volume, (rows, (0, height), (0, width)), "zeros")
            hit.append((sampled > 0.5).any(-1))
            footprint.append(sampled.amax(-1))
        coordinates = torch.nonzero(torch.cat(hit))
        if not len(coordinates):
            raise ValueError("Rotated lung mask is empty")
        low, high = coordinates.min(0).values, coordinates.max(0).values
        self.crop = (slice(max(int(low[0]) - 7, 0), min(int(high[0]) + 7, depth)),
                     slice(max(int(low[1]) - 7, 0), min(int(high[1]) + 7, height)))
        self.footprint = _resize(torch.cat(footprint)[self.crop])

    @staticmethod
    def _chunks(start, stop, row_voxels):
        rows = max(1, CHUNK_VOXELS // max(1, row_voxels))
        return [(first, min(stop, first + rows)) for first in range(start, stop, rows)]

    def _grid(self, block, box=None):
        """Input sampling coordinates [dz, dx, dy, 3] of an output block (optionally in box coordinates)."""
        (z0, z1), (x0, x1), (y0, y1) = block
        depth, height, width = self.shape
        gz = _normalized(depth, z0, z1, self.device)[:, None, None]
        gx = _normalized(height, x0, x1, self.device)[None, :, None]
        gy = _normalized(width, y0, y1, self.device)[None, None, :]
        theta = self.theta
        components = [theta[c, 0] * gy + theta[c, 1] * gx + theta[c, 2] * gz + theta[c, 3] for c in range(3)]
        if box is not None:  # component c samples array axis 2 - c; re-normalize to the box extent
            for c, item in enumerate(reversed(box)):
                size = self.shape[2 - c]
                components[c] = ((components[c] + 1.0) * size - 2.0 * item.start) / (item.stop - item.start) - 1.0
        return torch.stack(torch.broadcast_tensors(*components), -1)

    def _sample(self, volume, block, padding, box=None):
        return F.grid_sample(volume[None, None], self._grid(block, box)[None], mode="bilinear",
                             padding_mode=padding, align_corners=False)[0, 0]

    def _ray_sum(self, volume, block, padding, box=None):
        """AP ray-sum [dz, dx] of an output block, in depth chunks."""
        (z0, z1), rows_x, rows_y = block
        parts = [self._sample(volume, (rows, rows_x, rows_y), padding, box).sum(-1)
                 for rows in self._chunks(z0, z1, (rows_x[1] - rows_x[0]) * (rows_y[1] - rows_y[0]))]
        return torch.cat(parts)

    def _crop_block(self):
        return ((self.crop[0].start, self.crop[0].stop), (self.crop[1].start, self.crop[1].stop), (0, self.shape[2]))

    def _reach(self, box):
        """Output block (inside the crop) whose samples can fall inside box (one voxel margin)."""
        corners = []
        for z in (box[0].start - 1, box[0].stop):
            for x in (box[1].start - 1, box[1].stop):
                for y in (box[2].start - 1, box[2].stop):
                    corners.append([(2.0 * index + 1.0) / size - 1.0
                                    for index, size in zip((y, x, z), self.shape[::-1])])
        corners = torch.tensor(corners, dtype=torch.float32, device=self.device)
        output = (corners - self.theta[:, 3]) @ self.inverse.T  # normalized (y, x, z)
        indices = ((output + 1.0) * torch.tensor(self.shape[::-1], device=self.device) - 1.0) / 2.0
        low, high = indices.min(0).values.floor().int() - 1, indices.max(0).values.ceil().int() + 2
        crop = self._crop_block()
        block = []
        for axis, (first, last) in zip((0, 1, 2), crop):
            component = 2 - axis
            block.append((max(first, int(low[component])), min(last, int(high[component]))))
        return block if all(start < stop for start, stop in block) else None

    def _as_volume(self, volume):
        if torch.is_tensor(volume):
            return volume.to(self.device, torch.float32)
        return torch.as_tensor(np.asarray(volume, np.float32), device=self.device)

    def base(self, volume):
        """Clamped CT -> ray-sum of (HU + 1000), reflection padding as the existing generator."""
        shifted = torch.clamp_min(self._as_volume(volume), -1000.0)
        shifted += 1000.0
        return _resize(self._ray_sum(shifted, self._crop_block(), "reflection"))

    def delta(self, volume):
        """Projection of a change: a VolumeDelta (sampled from its box) or a full volume."""
        if not hasattr(volume, "box"):
            return _resize(self._ray_sum(self._as_volume(volume), self._crop_block(), "zeros"))
        crop = self._crop_block()
        image = torch.zeros(crop[0][1] - crop[0][0], crop[1][1] - crop[1][0], device=self.device)
        block = self._reach(volume.box)
        if block is not None:
            values = torch.as_tensor(np.asarray(volume.values, np.float32), device=self.device)
            image[block[0][0] - crop[0][0]:block[0][1] - crop[0][0],
                  block[1][0] - crop[1][0]:block[1][1] - crop[1][0]] = self._ray_sum(values, block, "zeros",
                                                                                     volume.box)
        return _resize(image)


def clean_map(signed, threshold=GT_THRESHOLD, min_component=MIN_COMPONENT):
    signed = np.where(np.abs(signed) < threshold, 0.0, signed).astype(np.float32)
    labels, count = ndi.label(signed != 0, EIGHT_CONNECTED)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes > min_component
        keep[0] = False
        signed[~keep[labels]] = 0.0
    return np.clip(signed, -1.0, 1.0)


def change_maps(delta_projections, base_projection, footprint, signs):
    scale = float((base_projection * footprint).max())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid projection scale")
    maps = [clean_map(sign * projection / scale) for projection, sign in zip(delta_projections, signs)]
    return maps, scale


def device_volume(ctx, name, device):
    """ctx.ct ('ct') or the lung mask ('lungs') as a float32 tensor, cached on ctx per device
    (drop_device_volumes frees it between scenes)."""
    cache = ctx.__dict__.setdefault("_device_volumes", {})
    key = (name, str(device))
    if key not in cache:
        if name == "ct":
            cache[key] = torch.as_tensor(np.asarray(ctx.ct, np.float32), device=device)
        else:  # upload the boolean mask (4x smaller) and convert on the device
            cache[key] = torch.from_numpy(np.ascontiguousarray(ctx.segs["lungs"])).to(device).float()
    return cache[key]


def drop_device_volumes(ctx):
    ctx.__dict__.pop("_device_volumes", None)


def render_scene(ctx, changes, rng, device="cpu", rotation_params=DEFAULT_ROTATION_PARAMS,
                 devices_hook: Optional[Callable] = None, prior_nuisance=None, angles=None):
    """Render one scene; ``changes`` = [(ChangeSpec, mode), ...] with a supervoxel_tasks timepoint mode
    (appear / resolve / progress / regress / stable; 'current' / 'prior' are aliases of appear / resolve).

    Returns raw projections in the DRR [z, x] frame plus per-change signed maps
    GT = P(current delta - prior delta) / scale (stable changes give an all-zero map).
    ``prior_nuisance`` is a VolumeDelta applied only to the prior render (never in GT).
    ``devices_hook(base tensor, rng) -> (prior tensor, current tensor, names)`` adds device nuisance.
    """
    from data_generation.supervoxel_tasks import timepoint_deltas

    current_angles, prior_angles = angles or (sample_angles(rng, rotation_params),
                                              sample_angles(rng, rotation_params))
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    ct = device_volume(ctx, "ct", device)
    lungs = device_volume(ctx, "lungs", device)
    base_prior = base_current = ct
    device_names = []
    if devices_hook is not None:
        base_prior, base_current, device_names = devices_hook(ct, rng)
    seconds = dict(devices=time.time() - started)
    gpu_peak = dict(devices=gpu_memory_gb(device))
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    frame = FrameRenderer(ctx.segs["lungs"], current_angles, device, lungs_tensor=lungs)
    base_projection = frame.base(base_current)
    del base_current
    current = base_projection.clone()
    registered_prior = base_projection.clone()
    differences = []
    pairs = [timepoint_deltas(change, mode) for change, mode in changes]
    for prior_delta, current_delta in pairs:
        current_projection = frame.delta(current_delta) if current_delta is not None else None
        if prior_delta is current_delta:
            prior_projection = current_projection
        else:
            prior_projection = frame.delta(prior_delta) if prior_delta is not None else None
        difference = torch.zeros_like(base_projection)
        if current_projection is not None:
            current += current_projection
            difference += current_projection
        if prior_projection is not None:
            registered_prior += prior_projection
            difference -= prior_projection
        differences.append(difference.cpu().numpy())
    maps, scale = change_maps(differences, base_projection.cpu().numpy(), frame.footprint.cpu().numpy(),
                              [1.0] * len(differences))
    prior_volume = torch.as_tensor(base_prior, dtype=torch.float32, device=device).clone()
    del base_prior
    for delta in [prior for prior, _ in pairs if prior is not None] + (
            [prior_nuisance] if prior_nuisance is not None else []):
        prior_volume[delta.box] += torch.as_tensor(np.asarray(delta.values, np.float32), device=device)
    prior = FrameRenderer(ctx.segs["lungs"], prior_angles, device, lungs_tensor=lungs).base(prior_volume)
    del prior_volume
    seconds["render"] = time.time() - started
    gpu_peak["render"] = gpu_memory_gb(device)
    return dict(prior=prior.cpu().numpy(), current=current.cpu().numpy(),
                registered_prior=registered_prior.cpu().numpy(), maps=maps, scale=scale,
                current_angles=list(current_angles), prior_angles=list(prior_angles),
                device_names=list(device_names), seconds=seconds, gpu_peak_gb=gpu_peak)


# ---------------------------------------------------------------------------- coarse labels
def fit_ellipses(signed, min_pixels=MIN_COMPONENT, inflate=1.15, pad=2.0, threshold=0.0):
    """Annotation-tool style ellipses (cx=row, cy=col, rx/ry radii, angle deg) per component."""
    from skimage.measure import regionprops

    ellipses = []
    for label, mask in (("Appearance", signed > threshold), ("Disappearance", signed < -threshold)):
        components, _ = ndi.label(mask, EIGHT_CONNECTED)
        for region in regionprops(components):
            if region.area <= min_pixels:
                continue
            row, column = region.centroid
            major = max(region.axis_major_length, 1.0) / 2 * inflate + pad
            minor = max(region.axis_minor_length, 1.0) / 2 * inflate + pad
            ellipses.append(dict(cx=float(row), cy=float(column), rx=float(major), ry=float(minor),
                                 angle=float(np.rad2deg(region.orientation)), label=label,
                                 comment="synthetic"))
    return ellipses


def rasterize_ellipses(ellipses, shape):
    """Same call as Observer_Variability_sq_crop.load_labels_map: returns (positive, negative)."""
    from skimage.draw import ellipse

    positive, negative = np.zeros(shape, bool), np.zeros(shape, bool)
    for item in ellipses:
        rows, columns = ellipse(item["cx"], item["cy"], item["rx"], item["ry"], shape=shape,
                                rotation=np.deg2rad(item["angle"]))
        if item["label"] == "Appearance":
            positive[rows, columns] = True
        elif item["label"] == "Disappearance":
            negative[rows, columns] = True
    return positive, negative


def realized_statistics(signed):
    magnitude = np.abs(signed)
    return dict(area_001=int((magnitude > 0.001).sum()), area_005=int((magnitude > 0.05).sum()),
                positive_area=int((signed > 0.001).sum()), negative_area=int((signed < -0.001).sum()),
                max_abs=float(magnitude.max()) if magnitude.size else 0.0)


def gt_sign(statistics):
    if statistics["positive_area"] and statistics["negative_area"]:
        smaller = min(statistics["positive_area"], statistics["negative_area"])
        if smaller / max(statistics["positive_area"], statistics["negative_area"]) > 0.1:
            return "mixed"
    if statistics["positive_area"] >= statistics["negative_area"] and statistics["positive_area"]:
        return "positive"
    if statistics["negative_area"]:
        return "negative"
    return "none"
