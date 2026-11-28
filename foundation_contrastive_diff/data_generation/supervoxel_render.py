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


def sampling_grid(shape, angles, device):
    """Grid for a rotation about the volume center in voxel units (anisotropic-safe)."""
    depth, height, width = shape
    rotation = _rotation_matrix(angles)  # acts on (z, x, y) voxel offsets
    half = torch.tensor([depth / 2, height / 2, width / 2])
    inverse = rotation.T
    # affine_grid coordinates are ordered (x=width, y=height, z=depth) and normalized.
    order = [2, 1, 0]
    theta = torch.zeros(1, 3, 4)
    for row_out, axis_out in enumerate(order):
        for column, axis_in in enumerate(order):
            theta[0, row_out, column] = inverse[axis_out, axis_in] * half[axis_in] / half[axis_out]
    return F.affine_grid(theta.to(device), (1, 1, depth, height, width), align_corners=False)


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
    return F.interpolate(image[None, None], size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear",
                         antialias=True, align_corners=False)[0, 0]


class FrameRenderer:
    """Renders several volumes with one rotation and one lung crop (one camera frame)."""

    def __init__(self, lungs, angles, device="cpu"):
        self.grid = sampling_grid(lungs.shape, angles, device)
        rotated_lungs = rotate(lungs.astype(np.float32), self.grid)
        self.crop = lung_crop(rotated_lungs)
        footprint = rotated_lungs[self.crop].amax(-1)
        self.footprint = F.interpolate(footprint[None, None], size=(OUTPUT_SIZE, OUTPUT_SIZE),
                                       mode="bilinear", antialias=True, align_corners=False)[0, 0]
        self.angles = tuple(angles)

    def base(self, volume):
        """Clamped CT -> ray-sum of (HU + 1000), reflection padding as the existing generator."""
        rotated = rotate(np.maximum(volume, -1000.0) + 1000.0, self.grid, "reflection")
        return project(rotated, self.crop)

    def delta(self, volume):
        return project(rotate(volume, self.grid, "zeros"), self.crop)


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


def render_scene(ctx, changes, rng, device="cpu", rotation_params=DEFAULT_ROTATION_PARAMS,
                 devices_hook: Optional[Callable] = None, prior_nuisance=None, angles=None):
    """Render one scene; ``changes`` = [(ChangeSpec, mode), ...] with a supervoxel_tasks timepoint mode
    (appear / resolve / progress / regress / stable; 'current' / 'prior' are aliases of appear / resolve).

    Returns raw projections in the DRR [z, x] frame plus per-change signed maps
    GT = P(current delta - prior delta) / scale (stable changes give an all-zero map).
    ``prior_nuisance`` is a VolumeDelta applied only to the prior render (never in GT).
    ``devices_hook(base) -> (prior_volume, current_volume)`` adds device nuisance.
    """
    from data_generation.supervoxel_tasks import timepoint_deltas

    current_angles, prior_angles = angles or (sample_angles(rng, rotation_params),
                                              sample_angles(rng, rotation_params))
    base_prior = base_current = ctx.ct
    device_names = []
    if devices_hook is not None:
        base_prior, base_current, device_names = devices_hook(ctx.ct)
    frame = FrameRenderer(ctx.segs["lungs"], current_angles, device)
    base_projection = frame.base(base_current)
    current = base_projection.clone()
    registered_prior = base_projection.clone()
    differences = []
    pairs = [timepoint_deltas(change, mode) for change, mode in changes]
    for prior_delta, current_delta in pairs:
        current_projection = frame.delta(current_delta.full(ctx.shape)) if current_delta is not None else None
        if prior_delta is current_delta:
            prior_projection = current_projection
        else:
            prior_projection = frame.delta(prior_delta.full(ctx.shape)) if prior_delta is not None else None
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
    prior_volume = np.array(base_prior, np.float32, copy=True)
    for prior_delta, _ in pairs:
        if prior_delta is not None:
            prior_volume[prior_delta.box] += prior_delta.values
    if prior_nuisance is not None:
        prior_volume[prior_nuisance.box] += prior_nuisance.values
    prior = FrameRenderer(ctx.segs["lungs"], prior_angles, device).base(prior_volume)
    return dict(prior=prior.cpu().numpy(), current=current.cpu().numpy(),
                registered_prior=registered_prior.cpu().numpy(), maps=maps, scale=scale,
                current_angles=list(current_angles), prior_angles=list(prior_angles),
                device_names=list(device_names))


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
