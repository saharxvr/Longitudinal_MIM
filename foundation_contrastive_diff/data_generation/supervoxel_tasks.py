"""Synthetic thoracic change tasks from CT segmentations (few-shot training corpus).

Every change is an additive HU delta on the (cropped, clamped) CT. Because the DRR is a
linear ray-sum after rotation (see supervoxel_render.py), each change can be projected on
its own, so every scene has an exact per-change signed GT map.

All sizes, distances, smoothing widths and rotations are in MILLIMETRES using the CT voxel
spacing (array axes z, x, y). Real CTs are strongly anisotropic (e.g. 0.35 x 0.35 x 1.5 mm),
so voxel-unit sizes would stretch every change along z by a different factor per CT.

Compartments come only from the existing segmentations (lungs, five lobes, heart, bronchi,
lung vessels) plus HU rules (body, bone, subcutaneous fat, mediastinum, sub-diaphragm,
chest wall, pleural rim, fissures, hila). No TotalSegmentator.

Tiers (see FEW_SHOT_DIFF_PLAN.md section 4.1):
    T1      generic supervoxel/noise regions x generic mechanisms, per compartment
    T2      anatomy-specific families that are not validation pathologies
    T2near  families that resemble validation pathologies (ablation only)
Validation pathologies (consolidation, effusion, pneumothorax, fluid overload) are never
generated here. Nuisance (inspiration depth) never enters any GT map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

HU_MIN, HU_MAX = -1000.0, 3000.0
HU_AIR, HU_LUNG, HU_FAT, HU_FLUID, HU_SOFT, HU_BONE = -1000.0, -850.0, -100.0, 15.0, 45.0, 500.0
TISSUES = dict(air=HU_AIR, fat=HU_FAT, fluid=HU_FLUID, soft=HU_SOFT, bone=HU_BONE)
LOBES = ("upper_right_lobe", "middle_right_lobe", "lower_right_lobe",
         "upper_left_lobe", "lower_left_lobe")
T1_COMPARTMENTS = ("lungs", "mediastinum", "heart", "chest_wall", "sub_diaphragm")
T1_MECHANISMS = ("density", "replace", "grow", "shrink", "displace", "texture", "cutpaste")
VALIDATION_FAMILIES = ("consolidation", "pleural_effusion", "pneumothorax", "fluid_overload")
ISOTROPIC = np.ones(3)


# ---------------------------------------------------------------------------- geometry (mm)
def _vector(value):
    return np.broadcast_to(np.asarray(value, np.float64), (3,)).astype(np.float64)


def box_around(center, radii, shape, margin=0):
    """Box around center with per-axis voxel radii and margin."""
    radii, margin = _vector(radii), _vector(margin)
    return tuple(slice(max(0, int(c - r - m)), min(s, int(c + r + m) + 1))
                 for c, r, m, s in zip(center, radii, margin, shape))


def mask_box(mask, margin, shape=None):
    """Bounding box of a boolean mask with a per-axis voxel margin; None if empty."""
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return None
    shape = mask.shape if shape is None else shape
    margin = _vector(margin).astype(int)
    low, high = coordinates.min(0), coordinates.max(0) + 1
    return tuple(slice(max(0, int(a) - m), min(s, int(b) + m)) for a, b, m, s in zip(low, high, margin, shape))


def box_grid(box):
    return np.ogrid[tuple(slice(item.start, item.stop) for item in box)]


def ellipsoid(box, center, radii_mm, spacing=ISOTROPIC, rotation=None):
    """Ellipsoid with radii in mm, optionally rotated in physical space."""
    spacing = _vector(spacing)
    z, x, y = box_grid(box)
    offsets = [(z - center[0]) * spacing[0], (x - center[1]) * spacing[1], (y - center[2]) * spacing[2]]
    if rotation is not None:
        offsets = [sum(rotation[row, column] * offsets[column] for column in range(3)) for row in range(3)]
    return sum((offset / max(radius, 1e-3)) ** 2 for offset, radius in zip(offsets, _vector(radii_mm))) <= 1


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def dilate(mask, radius_mm, spacing=ISOTROPIC):
    if radius_mm <= 0 or not mask.any():
        return mask.copy()
    return ndi.distance_transform_edt(~mask, sampling=_vector(spacing)) <= radius_mm


def erode(mask, radius_mm, spacing=ISOTROPIC):
    if radius_mm <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(mask, sampling=_vector(spacing)) > radius_mm


def smooth_noise(shape, sigma_mm, rng, spacing=ISOTROPIC):
    sigma = _vector(sigma_mm) / _vector(spacing)
    field_ = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return field_ / (field_.std() + 1e-6)


def feather(mask, sigma_mm, spacing=ISOTROPIC):
    if np.all(_vector(sigma_mm) <= 0):
        return mask.astype(np.float32)
    return ndi.gaussian_filter(mask.astype(np.float32), _vector(sigma_mm) / _vector(spacing))


# ---------------------------------------------------------------------------- deltas
@dataclass
class VolumeDelta:
    """Sparse additive HU change: values live inside ``box`` of the full volume."""

    box: Tuple[slice, slice, slice]
    values: np.ndarray

    def support(self, threshold=1.0):
        return np.abs(self.values) > threshold

    def full(self, shape):
        volume = np.zeros(shape, np.float32)
        volume[self.box] = self.values
        return volume


@dataclass
class ChangeSpec:
    family: str
    tier: str
    compartment: str
    effect: str  # denser | lighter | mixed (HU direction in the modified volume)
    delta: VolumeDelta
    params: Dict = field(default_factory=dict)
    partial: Optional[VolumeDelta] = field(default=None, repr=False)  # smaller version (progress/regress)
    replay: Optional[tuple] = field(default=None, repr=False)  # (rng state, donor) of the successful draw


def bounded_delta(base, box, target):
    """Delta that moves base[box] to target while respecting the projection HU bounds."""
    values = np.clip(target, HU_MIN, HU_MAX) - base[box]
    return VolumeDelta(box, values.astype(np.float32))


def effect_of(values, threshold=1.0):
    positive, negative = float(values[values > threshold].sum()), float(-values[values < -threshold].sum())
    total = positive + negative
    if total <= 0:
        return None
    if positive / total > 0.9:
        return "denser"
    if negative / total > 0.9:
        return "lighter"
    return "mixed"


# ---------------------------------------------------------------------------- CT context
class CTContext:
    """Cropped CT (HU >= -1000) with voxel spacing (z, x, y mm), segmentations and compartments.

    Compartments live on a grid of about ``grid_mm`` and supervoxels on about ``supervoxel_mm``
    (per-axis integer downsampling); ``downsample`` / ``supervoxel_downsample`` force isotropic
    integer factors instead (tests). ``scale`` is the mean lung extent in mm.
    ``min_visibility`` = (peak |GT|, GT area in DRR pixels) a change must reach in an
    approximate AP projection, else it is rejected (None disables the gate).
    """

    def __init__(self, ct, segs, spacing=(1.0, 1.0, 1.0), grid_mm=3.0, supervoxel_mm=5.0, supervoxels=None,
                 n_supervoxels=1500, seed=0, downsample=None, supervoxel_downsample=None,
                 min_visibility=(0.035, 80.0)):
        self.min_visibility = min_visibility
        self.ct = np.clip(np.asarray(ct, np.float32), HU_MIN, HU_MAX)
        self.shape = self.ct.shape
        self.spacing = _vector(spacing)
        self.ds = (np.full(3, int(downsample)) if downsample
                   else np.maximum(1, np.round(grid_mm / self.spacing)).astype(int))
        self.sv_ds = (np.full(3, int(supervoxel_downsample)) if supervoxel_downsample
                      else np.maximum(1, np.round(supervoxel_mm / self.spacing)).astype(int))
        self.ds_spacing = self.spacing * self.ds
        self.voxel_mm3 = float(np.prod(self.spacing))
        self.segs = {name: np.asarray(mask) > 0.5 for name, mask in segs.items()}
        if "lungs" not in self.segs:
            lobes = [self.segs[name] for name in LOBES if name in self.segs]
            if not lobes:
                raise ValueError("Need a lungs segmentation or lobe segmentations")
            self.segs["lungs"] = np.logical_or.reduce(lobes)
        if int(self.segs["lungs"].sum()) < 1000:
            raise ValueError("Lung segmentation is too small")
        self._geometry()
        self.compartments = self._compartments()
        expected = tuple((size + factor - 1) // factor for size, factor in zip(self.shape, self.sv_ds))
        if supervoxels is not None and tuple(supervoxels.shape) != expected:
            supervoxels = None  # stale cache from another grid
        self.supervoxels = (supervoxels if supervoxels is not None
                            else compute_supervoxels(self.ct, self.compartments["body"], self.ds, self.sv_ds,
                                                     self.spacing, n_supervoxels, seed))
        self.adjacency = supervoxel_adjacency(self.supervoxels)

    # unit helpers
    def vox(self, mm):
        """Per-axis voxel lengths of a physical length (mm)."""
        return _vector(mm) / self.spacing

    def _down(self, mask):
        return mask[::self.ds[0], ::self.ds[1], ::self.ds[2]]

    def _index(self, box, factors):
        factors = _vector(factors).astype(int)
        sizes = [(size + factor - 1) // factor for size, factor in zip(self.shape, factors)]
        return np.ix_(*[np.minimum(np.arange(item.start, item.stop) // factor, size - 1)
                        for item, factor, size in zip(box, factors, sizes)])

    def mask(self, name, box):
        """Full-resolution boolean compartment within box."""
        if name in self.segs:
            return self.segs[name][box]
        return self.compartments[name][self._index(box, self.ds)]

    def labels(self, box):
        return self.supervoxels[self._index(box, self.sv_ds)]

    def full_box(self):
        return tuple(slice(0, size) for size in self.shape)

    def available(self, name):
        if name in self.segs:
            return int(self.segs[name].sum()) > 50
        return name in self.compartments and int(self.compartments[name].sum()) > 20

    def visibility(self, delta):
        """Approximate DRR effect of a delta (AP projection, no rotation): peak |GT| and GT area (px)."""
        projected = delta.values.astype(np.float64).sum(axis=2) / self.ap_scale
        peak = float(np.abs(projected).max()) if projected.size else 0.0
        area = float((np.abs(projected) > 0.015).sum() * self.drr_pixels_per_cell)
        return peak, area

    def dome_columns(self, lung):
        """(x, y) columns whose lowest lung voxel is on the diaphragm (bottom 40% of that lung)."""
        z = np.nonzero(lung.any((1, 2)))[0]
        if not len(z):
            return np.zeros(lung.shape[1:], bool)
        columns = lung.any(0)
        if self.inferior > 0:
            base = lung.shape[0] - 1 - np.argmax(lung[::-1], 0)
            return columns & (base >= z[0] + 0.6 * (z[-1] - z[0]))
        base = np.argmax(lung, 0)
        return columns & (base <= z[-1] - 0.6 * (z[-1] - z[0]))

    def full_points(self, mask_ds):
        """Full-resolution voxel coordinates of a compartment-grid mask (cell centres)."""
        return np.argwhere(mask_ds) * self.ds + self.ds // 2

    def _geometry(self):
        lungs = self.segs["lungs"]
        area = lungs.sum((1, 2)).astype(np.float64)
        z = np.nonzero(area)[0]
        self.z0, self.z1 = int(z[0]), int(z[-1]) + 1
        coordinates = np.argwhere(lungs[::self.ds[0], ::self.ds[1], ::self.ds[2]]) * self.ds
        self.lung_center = coordinates.mean(0)
        # Head/foot: the heart lies inferior to the lung centre and the trachea superior. Lung
        # area profiles are NOT reliable (diaphragm domes make the lowest slices small).
        self.inferior, self.inferior_rule = 1, "default (+z after load_volume flip)"
        heart = self.segs.get("heart")
        airways = self.segs.get("bronchi")
        if heart is not None and heart.sum() > 1000:
            heart_z = np.argwhere(heart[::self.ds[0], ::self.ds[1], ::self.ds[2]])[:, 0].mean() * self.ds[0]
            self.inferior, self.inferior_rule = (1 if heart_z > self.lung_center[0] else -1), "heart centroid"
        elif airways is not None and (airways & ~lungs).sum() > 100:
            trachea_z = np.argwhere((airways & ~lungs)[::self.ds[0], ::self.ds[1], ::self.ds[2]])[:, 0].mean() * self.ds[0]
            self.inferior, self.inferior_rule = (1 if trachea_z < self.lung_center[0] else -1), "trachea centroid"
        self.lung_extent_mm = (coordinates.max(0) - coordinates.min(0) + self.ds) * self.spacing
        self.lung_height_mm = (self.z1 - self.z0) * self.spacing[0]
        self.scale = float(np.mean(self.lung_extent_mm))
        # Anterior-posterior projection without rotation (visibility pre-check), same units as the DRR.
        self.ap_base = self.ct.sum(axis=2, dtype=np.float64) + 1000.0 * self.shape[2]
        self.ap_footprint = lungs.any(axis=2)
        self.ap_scale = float((self.ap_base * self.ap_footprint).max())
        x = np.nonzero(self.ap_footprint.any(0))[0]
        self.drr_pixels_per_cell = (512.0 / (self.z1 - self.z0 + 14)) * (512.0 / (x[-1] - x[0] + 15))
        right = self.segs.get("lung_right")
        left = self.segs.get("lung_left")
        if right is None and all(name in self.segs for name in LOBES[:3]):
            right = np.logical_or.reduce([self.segs[name] for name in LOBES[:3]])
        if left is None and all(name in self.segs for name in LOBES[3:]):
            left = np.logical_or.reduce([self.segs[name] for name in LOBES[3:]])
        if right is None or left is None or not right.any() or not left.any():
            labels, count = ndi.label(self._down(lungs))
            if count < 2:
                raise ValueError("Cannot separate left and right lungs")
            sizes = ndi.sum(np.ones_like(labels), labels, range(1, count + 1))
            first, second = np.argsort(sizes)[-2:] + 1
            up = self._index(self.full_box(), self.ds)
            right, left = labels[up] == first, labels[up] == second
        self.segs["lung_right"], self.segs["lung_left"] = right, left

    def lung_z_fraction(self, fraction):
        """Full-resolution z index at a fraction from apex (0) to base (1)."""
        if self.inferior > 0:
            return int(self.z0 + fraction * (self.z1 - self.z0))
        return int(self.z1 - 1 - fraction * (self.z1 - self.z0))

    def _compartments(self):
        ct = self._down(self.ct)
        segs = {name: self._down(mask) for name, mask in self.segs.items()}
        grid = self.ds_spacing
        lungs = segs["lungs"]
        body = ndi.binary_opening(ct > -500, iterations=1)
        labels, count = ndi.label(body)
        if count:
            sizes = ndi.sum(np.ones_like(labels), labels, range(1, count + 1))
            body = labels == (int(np.argmax(sizes)) + 1)
        body = np.stack([ndi.binary_fill_holes(plane) for plane in body]) | lungs
        heart = segs.get("heart", np.zeros_like(body))
        bone = body & (ct > 250) & ~lungs
        z0, z1 = self.z0 // self.ds[0], max(self.z0 // self.ds[0] + 1, self.z1 // self.ds[0])
        in_lung_z = np.zeros(body.shape, bool)
        in_lung_z[z0:z1] = True

        right, left = segs["lung_right"], segs["lung_left"]
        low, high = ((right, left) if np.argwhere(right)[:, 1].mean() < np.argwhere(left)[:, 1].mean()
                     else (left, right))
        mediastinum = np.zeros_like(body)
        for z in range(z0, z1):
            low_x, high_x = np.nonzero(low[z].any(1))[0], np.nonzero(high[z].any(1))[0]
            if not len(low_x) or not len(high_x):
                continue
            start, stop = low_x.max(), high_x.min()
            if stop - start < 2:
                continue
            lung_y = np.nonzero((low[z] | high[z]).any(0))[0]
            mediastinum[z, start:stop, lung_y.min():lung_y.max() + 1] = True
        mediastinum &= body & ~lungs & ~heart & ~bone & (ct > -300)

        distance_to_lungs = ndi.distance_transform_edt(~lungs, sampling=grid)
        dome = self.dome_columns(segs["lung_right"]) | self.dome_columns(segs["lung_left"])
        if self.inferior > 0:
            base_index = np.where(dome, lungs.shape[0] - 1 - np.argmax(lungs[::-1], 0), -1)
            depth = np.arange(lungs.shape[0])[:, None, None] - base_index[None]
        else:
            base_index = np.where(dome, np.argmax(lungs, 0), lungs.shape[0])
            depth = base_index[None] - np.arange(lungs.shape[0])[:, None, None]
        depth_mm = depth * grid[0]
        band_mm = max(2 * grid[0], 0.15 * self.lung_height_mm)
        sub_diaphragm = body & ~lungs & ~bone & dome[None] & (depth_mm > 0) & (depth_mm <= band_mm)
        # Chest wall only where it projects onto the lungs (lateral wall is cropped out of the DRR).
        projects_on_lungs = lungs.any(axis=2)[:, :, None]
        chest_wall = (body & ~lungs & ~mediastinum & ~heart & ~sub_diaphragm & in_lung_z & projects_on_lungs
                      & (distance_to_lungs <= max(2 * grid.max(), 0.12 * self.scale)))
        surface = body & ~erode(body, max(grid.max(), 0.04 * self.scale), grid)
        subcutaneous = chest_wall & surface & (ct > -190) & (ct < -30)
        rim = lungs & ~erode(lungs, max(grid.max(), 0.015 * self.scale), grid)
        fissures = np.zeros_like(body)
        lobes = [segs[name] for name in LOBES if name in segs]
        for index, lobe in enumerate(lobes):
            grown = dilate(lobe, 1.5 * grid.max(), grid)
            for other in lobes[index + 1:]:
                fissures |= grown & other
        z_range = sorted((self.lung_z_fraction(0.3) // self.ds[0], self.lung_z_fraction(0.7) // self.ds[0]))
        central = np.zeros_like(body)
        central[z_range[0]:z_range[1] + 1] = True
        near_mediastinum = dilate(mediastinum | heart, max(2 * grid.max(), 0.08 * self.scale), grid)
        hila = lungs & near_mediastinum & central
        tubes = segs.get("lung_vessels", np.zeros_like(body)) | segs.get("bronchi", np.zeros_like(body))
        if tubes.any():
            hila &= dilate(tubes, 2 * grid.max(), grid)
        return dict(body=body, bone=bone, mediastinum=mediastinum, sub_diaphragm=sub_diaphragm,
                    chest_wall=chest_wall, subcutaneous=subcutaneous, pleural_rim=rim,
                    fissures=fissures, hila=hila, lungs_ds=lungs, heart_ds=heart)


def compute_supervoxels(ct, body, compartment_ds, sv_ds, spacing, n_segments, seed=0):
    """3D SLIC on the windowed, downsampled CT with physical spacing (cache the result per CT)."""
    from skimage.segmentation import slic

    sv_ds, compartment_ds = _vector(sv_ds).astype(int), _vector(compartment_ds).astype(int)
    image = np.clip((ct[::sv_ds[0], ::sv_ds[1], ::sv_ds[2]] + 1000.0) / 1400.0, 0, 1)
    index = np.ix_(*[np.minimum(np.arange(size) * factor // grid, limit - 1)
                     for size, factor, grid, limit in zip(image.shape, sv_ds, compartment_ds, body.shape)])
    labels = slic(image.astype(np.float64), n_segments=n_segments, compactness=0.08, channel_axis=None,
                  mask=body[index], start_label=1, enforce_connectivity=True,
                  spacing=_vector(spacing) * sv_ds)
    return labels.astype(np.int32)


def supervoxel_adjacency(labels):
    pairs = []
    for axis in range(3):
        a = np.moveaxis(labels, axis, 0)[:-1].ravel()
        b = np.moveaxis(labels, axis, 0)[1:].ravel()
        keep = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.stack((a[keep], b[keep]), 1))
    pairs = np.unique(np.concatenate(pairs), axis=0) if pairs else np.zeros((0, 2), int)
    adjacency: Dict[int, set] = {}
    for first, second in pairs.tolist():
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    return adjacency


# ---------------------------------------------------------------------------- regions
def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _candidates(ctx, compartment):
    if compartment in ctx.segs:
        points = np.argwhere(ctx._down(ctx.segs[compartment])) * ctx.ds
    elif compartment in ctx.compartments:
        points = ctx.full_points(ctx.compartments[compartment])
    else:
        return None
    return points if len(points) else None


def _interface_candidates(ctx, compartment, width_mm=10.0):
    """Compartment points within width_mm of the lung/non-lung interface (where edits have contrast)."""
    cache = ctx.__dict__.setdefault("_interface_cache", {})
    if compartment not in cache:
        lungs, grid = ctx.compartments["lungs_ds"], ctx.ds_spacing
        if compartment == "lungs":
            region = lungs & ~erode(lungs, width_mm, grid)
        else:
            own = ctx._down(ctx.segs[compartment]) if compartment in ctx.segs else ctx.compartments[compartment]
            region = own & ~lungs & dilate(lungs, width_mm, grid)
        cache[compartment] = ctx.full_points(region) if region.any() else None
    return cache[compartment]


def sample_region(ctx: CTContext, compartment, rng, size_range=(0.03, 0.12), sampler=None, candidates=None):
    """Random connected supervoxel union (or smooth-noise blob) inside a compartment.

    The radius is a log-uniform fraction of the lung scale (mm); returns (box, mask) or None.
    """
    sampler = sampler or ("supervoxel" if rng.random() < 0.7 else "noise")
    radius = log_uniform(rng, *size_range) * ctx.scale
    candidates = _candidates(ctx, compartment) if candidates is None else candidates
    if candidates is None:
        return None
    center = candidates[rng.integers(len(candidates))]
    box = box_around(center, ctx.vox(radius * 1.5), ctx.shape, margin=2)
    inside = ctx.mask(compartment, box)
    if sampler == "noise":
        noise = smooth_noise(inside.shape, max(ctx.spacing.max(), radius / 3), rng, ctx.spacing)
        blob = (ellipsoid(box, center, radius, ctx.spacing, random_rotation(rng))
                & (noise > rng.uniform(-0.5, 0.5)))
        mask = blob & inside
    else:
        labels = ctx.labels(box)
        start = int(ctx.supervoxels[tuple(np.minimum(center // ctx.sv_ds, np.array(ctx.supervoxels.shape) - 1))])
        if start <= 0:
            return None
        target = (4.0 / 3.0) * np.pi * radius ** 3 / ctx.voxel_mm3
        sizes = np.bincount(labels.ravel(), minlength=max(int(labels.max()) + 1, start + 1))
        chosen, frontier, volume = {start}, [start], float(sizes[start])
        while frontier and volume < target:
            node = frontier.pop(int(rng.integers(len(frontier))))
            for neighbor in ctx.adjacency.get(node, ()):
                if neighbor in chosen or neighbor >= len(sizes) or not sizes[neighbor]:
                    continue
                chosen.add(neighbor)
                frontier.append(neighbor)
                volume += sizes[neighbor]
                if volume >= target:
                    break
        mask = np.isin(labels, list(chosen)) & inside
    if mask.any():
        mask = ndi.binary_closing(mask, iterations=1) & inside
    if mask.sum() * ctx.voxel_mm3 < 20.0:
        return None
    return box, mask


# ---------------------------------------------------------------------------- mechanisms
def op_density(ctx, box, mask, rng, sign=None, high=700.0):
    sign = sign or (1 if rng.random() < 0.5 else -1)
    soft = feather(mask, rng.uniform(1.0, 3.0), ctx.spacing)
    amplitude = rng.uniform(200, high)
    texture = 1 + 0.3 * smooth_noise(mask.shape, 2.0, rng, ctx.spacing)
    target = ctx.ct[box] + sign * amplitude * soft * texture
    return bounded_delta(ctx.ct, box, target), dict(sign=sign, amplitude=amplitude)


def op_replace(ctx, box, mask, rng, tissue=None):
    current = float(np.median(ctx.ct[box][mask]))
    options = [name for name, value in TISSUES.items() if abs(value - current) > 150]
    tissue = tissue or str(rng.choice(options))
    soft = feather(mask, rng.uniform(1.0, 2.0), ctx.spacing)
    value = TISSUES[tissue] + 15 * smooth_noise(mask.shape, 1.5, rng, ctx.spacing)
    target = ctx.ct[box] * (1 - soft) + value * soft
    return bounded_delta(ctx.ct, box, target), dict(tissue=tissue)


def _smooth_inside(ctx, target, region, box):
    smoothed = ndi.gaussian_filter(target, 1.0 / ctx.spacing)
    return smoothed * region + ctx.ct[box] * ~region


def _contrasting(ctx, box, region, value, contrast=150.0):
    """Voxels of region whose HU differs from value by more than contrast (bone excluded)."""
    return region & (np.abs(ctx.ct[box] - value) > contrast) & ~ctx.mask("bone", box)


def op_grow(ctx, box, mask, rng, compartment):
    """The region's tissue advances into adjacent tissue of different density (an interface moves)."""
    radius = rng.uniform(3.0, max(4.0, 0.05 * ctx.scale))
    value = float(np.median(ctx.ct[box][mask]))
    grown = _contrasting(ctx, box, dilate(mask, radius, ctx.spacing) & ctx.mask("body", box) & ~mask, value)
    if compartment == "lungs":
        grown &= ~ctx.mask("chest_wall", box)  # lung advancing into the wall would mimic pneumothorax
    if grown.sum() * ctx.voxel_mm3 < 20.0:
        return None, {}
    target = ctx.ct[box].copy()
    target[grown] = value + 15 * smooth_noise(mask.shape, 1.5, rng, ctx.spacing)[grown]
    return bounded_delta(ctx.ct, box, _smooth_inside(ctx, target, grown | mask, box)), dict(radius_mm=radius, fill=value)


def op_shrink(ctx, box, mask, rng, compartment):
    """Adjacent tissue of different density invades the region from its side."""
    radius = rng.uniform(3.0, max(4.0, 0.05 * ctx.scale))
    value = float(np.median(ctx.ct[box][mask]))
    ring = dilate(mask, radius + 2 * ctx.spacing.max(), ctx.spacing) & ~mask & ctx.mask("body", box)
    neighbor = _contrasting(ctx, box, ring, value)
    if neighbor.sum() * ctx.voxel_mm3 < 20.0:
        return None, {}
    removed = mask & dilate(neighbor, radius, ctx.spacing)
    if removed.sum() * ctx.voxel_mm3 < 20.0:
        return None, {}
    fill = float(np.median(ctx.ct[box][neighbor]))
    target = ctx.ct[box].copy()
    target[removed] = fill + 15 * smooth_noise(mask.shape, 1.5, rng, ctx.spacing)[removed]
    return bounded_delta(ctx.ct, box, target), dict(radius_mm=radius, fill=fill)


def displacement(ctx, box, values, weight, vector_mm):
    """Warp values by a smooth displacement (mm vector scaled by weight in [0, 1])."""
    coordinates = np.indices(weight.shape, dtype=np.float32)
    shift = _vector(vector_mm) / ctx.spacing
    moved = [coordinates[axis] - shift[axis] * weight for axis in range(3)]
    return ndi.map_coordinates(values, moved, order=1, mode="nearest")


def op_displace(ctx, box, mask, rng, compartment):
    """Smooth local push of tissue around the region (sampled at an interface, so it moves an edge)."""
    magnitude = rng.uniform(3.0, max(4.0, 0.05 * ctx.scale))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    weight = feather(dilate(mask, magnitude, ctx.spacing), magnitude, ctx.spacing)
    weight /= weight.max() + 1e-6
    warped = displacement(ctx, box, ctx.ct[box], weight, magnitude * direction)
    return bounded_delta(ctx.ct, box, warped), dict(magnitude_mm=magnitude, direction=direction.tolist())


def op_texture(ctx, box, mask, rng, kind=None):
    """Non-negative fine / reticular / nodular pattern (zero-mean patterns cancel along the ray)."""
    kind = kind or str(rng.choice(["fine", "reticular", "nodular"]))
    if kind == "fine":
        pattern = np.clip(smooth_noise(mask.shape, 1.0, rng, ctx.spacing), 0, None)
    elif kind == "reticular":
        pattern = (np.abs(smooth_noise(mask.shape, 2.0, rng, ctx.spacing)) < 0.3).astype(np.float32)
    else:
        dots = (rng.random(mask.shape) < 0.004 * ctx.voxel_mm3).astype(np.float32)
        pattern = ndi.gaussian_filter(dots, 1.5 / ctx.spacing)
        pattern /= pattern.max() + 1e-6
    amplitude = rng.uniform(300, 700)
    target = ctx.ct[box] + amplitude * pattern * feather(mask, 1.0, ctx.spacing)
    return bounded_delta(ctx.ct, box, target), dict(kind=kind, amplitude=amplitude)


def op_cutpaste(ctx, box, mask, rng, donor=None):
    """Paste HU content from a donor CT patch (another training CT) into the region."""
    if donor is None:
        return None, {}
    patch = donor[tuple(slice(0, min(a, b)) for a, b in zip(donor.shape, mask.shape))]
    if patch.shape != mask.shape:
        patch = np.pad(patch, [(0, a - b) for a, b in zip(mask.shape, patch.shape)], mode="edge")
    soft = feather(mask, 1.5, ctx.spacing)
    target = ctx.ct[box] * (1 - soft) + patch * soft
    return bounded_delta(ctx.ct, box, target), dict(donor_shape=list(donor.shape))


T1_SIZE = dict(default=(0.03, 0.18), texture=(0.06, 0.2))  # region radius, fraction of the lung scale
T1_AT_INTERFACE = ("grow", "shrink", "displace")
T1_FOCI, T1_FOCI_P = (1, 2, 3), (0.5, 0.3, 0.2)  # mean 1.7, as ICU annotations (1.7 regions each)
T1_DENSITY_MAX = dict(lungs=950.0)  # lung (-850 HU) can become fully soft tissue


def merge_deltas(parts):
    """One VolumeDelta over the union box of non-overlapping parts."""
    if len(parts) == 1:
        return parts[0]
    low = np.min([[item.start for item in part.box] for part in parts], 0)
    high = np.max([[item.stop for item in part.box] for part in parts], 0)
    values = np.zeros(high - low, np.float32)
    for part in parts:
        values[tuple(slice(item.start - a, item.stop - a) for item, a in zip(part.box, low))] += part.values
    return VolumeDelta(tuple(slice(int(a), int(b)) for a, b in zip(low, high)), values)


def _boxes_overlap(first, second, margin):
    return all(a.start < b.stop + m and b.start < a.stop + m for a, b, m in zip(first, second, margin))


def t1_change(ctx, rng, compartment, mechanism, donor=None, foci=None):
    """One T1 change: the same mechanism (and sign / tissue / texture kind) in 1-3 separate regions."""
    # Edits inside homogeneous tissue are invisible in projection: interface mechanisms (and
    # non-lung cut-paste) are seeded within 10 mm of the lung boundary.
    at_interface = mechanism in T1_AT_INTERFACE or (mechanism == "cutpaste" and compartment != "lungs")
    candidates = _interface_candidates(ctx, compartment) if at_interface else None
    if at_interface and candidates is None:
        return None
    foci = foci or int(rng.choice(T1_FOCI, p=T1_FOCI_P))
    shared = dict(sign=1 if rng.random() < 0.5 else -1, tissue=None, kind=None)
    margin = np.ceil(ctx.vox(3.0)).astype(int)
    parts, params = [], {}
    for _ in range(3 * foci):
        if len(parts) == foci:
            break
        region = sample_region(ctx, compartment, rng, T1_SIZE.get(mechanism, T1_SIZE["default"]),
                               candidates=candidates)
        if region is None:
            continue
        box, mask = region
        if any(_boxes_overlap(box, part.box, margin) for part in parts):
            continue
        delta, params_i = _t1_op(ctx, box, mask, rng, compartment, mechanism, donor, shared)
        if delta is None or effect_of(delta.values) is None:
            continue
        shared["tissue"] = shared["tissue"] or params_i.get("tissue")
        shared["kind"] = shared["kind"] or params_i.get("kind")
        parts.append(delta)
        params = params or params_i
    if not parts:
        return None
    return _finish(ctx, merge_deltas(parts), f"t1_{mechanism}", "T1", compartment, dict(params, foci=len(parts)))


def _t1_op(ctx, box, mask, rng, compartment, mechanism, donor, shared):
    if mechanism == "density":
        return op_density(ctx, box, mask, rng, shared["sign"], T1_DENSITY_MAX.get(compartment, 700.0))
    if mechanism == "replace":
        return op_replace(ctx, box, mask, rng, shared["tissue"])
    if mechanism == "grow":
        return op_grow(ctx, box, mask, rng, compartment)
    if mechanism == "shrink":
        return op_shrink(ctx, box, mask, rng, compartment)
    if mechanism == "displace":
        return op_displace(ctx, box, mask, rng, compartment)
    if mechanism == "texture":
        return op_texture(ctx, box, mask, rng, shared["kind"])
    if mechanism == "cutpaste":
        return op_cutpaste(ctx, box, mask, rng, donor)
    raise ValueError(f"Unknown T1 mechanism {mechanism}")


def _finish(ctx, delta, family, tier, compartment, params):
    if delta is None:
        return None
    effect = effect_of(delta.values)
    if effect is None or delta.support().sum() * ctx.voxel_mm3 < 20.0:
        return None
    peak, area = ctx.visibility(delta)
    if ctx.min_visibility is not None and (peak < ctx.min_visibility[0] or area < ctx.min_visibility[1]):
        return None
    params = dict(params, ap_peak=round(peak, 4), ap_area_px=int(area))
    return ChangeSpec(family, tier, compartment, effect, delta, params)


# ---------------------------------------------------------------------------- T2 helpers
def _point(ctx, points, rng):
    return points[rng.integers(len(points))] if points is not None and len(points) else None


def _compartment_point(ctx, name, rng, z_range=None, near_lungs_mm=None):
    mask = ctx.compartments[name]
    if z_range is not None:
        mask = mask.copy()
        z = np.arange(mask.shape[0]) * ctx.ds[0]
        mask[(z < z_range[0]) | (z > z_range[1])] = False
    if near_lungs_mm is not None:
        mask = mask & dilate(ctx.compartments["lungs_ds"], near_lungs_mm, ctx.ds_spacing)
    return _point(ctx, ctx.full_points(mask) if mask.any() else None, rng)


def _replace_in(ctx, box, region, value, rng, sigma_mm=1.0, noise=12.0):
    soft = feather(region, sigma_mm, ctx.spacing)
    filled = value + noise * smooth_noise(region.shape, 1.5, rng, ctx.spacing)
    return bounded_delta(ctx.ct, box, ctx.ct[box] * (1 - soft) + filled * soft)


def _ball(ctx, center, radii_mm, rng=None, region=None, rotate=False, scale=1.0):
    """Ellipsoid of radii_mm * scale; the box always fits the full radii (same box for any scale)."""
    radii_mm = _vector(radii_mm)
    box = box_around(center, ctx.vox(radii_mm.max()), ctx.shape, margin=2)
    ball = ellipsoid(box, center, radii_mm * scale, ctx.spacing, random_rotation(rng) if rotate and rng else None)
    if region is not None:
        ball &= ctx.mask(region, box)
    return box, ball


# ---------------------------------------------------------------------------- T2 families
def _scale_structure(ctx, name, scale, fill_out):
    """Scale a structure about its centroid (isotropic in mm); uncovered voxels take fill_out."""
    structure = ctx.segs.get(name)
    if structure is None or not structure.any():
        return None, {}
    box = mask_box(structure, ctx.vox(0.15 * ctx.scale) + 3)
    inside = structure[box]
    center = np.argwhere(inside[::2, ::2, ::2]).mean(0) * 2
    coordinates = np.indices(inside.shape, dtype=np.float32)
    source = [center[axis] + (coordinates[axis] - center[axis]) / scale for axis in range(3)]
    scaled = ndi.map_coordinates(inside.astype(np.float32), source, order=1) > 0.5
    scaled_ct = ndi.map_coordinates(ctx.ct[box], source, order=1, mode="nearest")
    body = ctx.mask("body", box)
    target = ctx.ct[box].copy()
    target[scaled & body] = scaled_ct[scaled & body]
    uncovered = inside & ~scaled
    target[uncovered] = fill_out(box, uncovered) if callable(fill_out) else fill_out
    return bounded_delta(ctx.ct, box, _smooth_inside(ctx, target, scaled | inside, box)), dict(scale=scale)


def heart_resize(ctx, rng, fraction=1.0):
    scale = rng.uniform(1.06, 1.25) if rng.random() < 0.6 else rng.uniform(0.82, 0.95)
    scale = 1.0 + fraction * (scale - 1.0)

    def surrounding(box, uncovered):
        heart = ctx.segs["heart"][box]
        ring = dilate(heart, 3.0, ctx.spacing) & ~heart
        return float(np.median(ctx.ct[box][ring])) if ring.any() else HU_LUNG

    delta, params = _scale_structure(ctx, "heart", scale, surrounding)
    return _finish(ctx, delta, "heart_resize", "T2", "heart", params)


def pericardial_shell(ctx, rng, fraction=1.0):
    heart = ctx.segs.get("heart")
    if heart is None or not heart.any():
        return None
    thickness = rng.uniform(3.0, max(4.0, 0.04 * ctx.scale))
    box = mask_box(heart, ctx.vox(thickness) + 3)
    shell = dilate(heart[box], thickness * fraction, ctx.spacing) & ~heart[box] & ctx.mask("body", box)
    keep = smooth_noise(shell.shape, 6.0, rng, ctx.spacing) > rng.uniform(-1.0, 0.0)
    delta = _replace_in(ctx, box, shell & keep, HU_FLUID, rng)
    return _finish(ctx, delta, "pericardial_shell", "T2", "heart", dict(thickness_mm=thickness * fraction))


def mediastinal_widening(ctx, rng, fraction=1.0):
    """Upper mediastinum widens laterally into the lungs (z-tapered, asymmetric between sides)."""
    z_lo, z_hi = sorted((ctx.lung_z_fraction(0.0), ctx.lung_z_fraction(0.5)))
    upper = ctx.compartments["mediastinum"].copy()
    z = np.arange(upper.shape[0]) * ctx.ds[0]
    upper[(z < z_lo) | (z > z_hi)] = False
    small = mask_box(upper, 0)
    if small is None:
        return None
    width = rng.uniform(5.0, max(6.0, 0.07 * ctx.scale))
    pad = np.ceil(ctx.vox(width)).astype(int) + 2
    box = tuple(slice(max(0, item.start * factor - (p if axis else 0)), min(size, item.stop * factor + (p if axis else 0)))
                for axis, (item, factor, p, size) in enumerate(zip(small, ctx.ds, pad, ctx.shape)))
    lungs = ctx.segs["lungs"][box]
    medi = upper[ctx._index(box, ctx.ds)] & ~lungs
    if not medi.any():
        return None
    # In-plane distance only (huge z sampling): every slice widens from its own mediastinal contour.
    distance = ndi.distance_transform_edt(~medi, sampling=(1e6, ctx.spacing[1], ctx.spacing[2]))
    t = (np.arange(box[0].start, box[0].stop) - z_lo) / max(1, z_hi - z_lo)
    profile = np.clip(np.exp(-((t - rng.uniform(0.3, 0.7)) / 0.35) ** 2), 0.2, 1.0)
    sides = rng.uniform(0.3, 1.0, size=2)
    sides[rng.integers(2)] = 1.0
    center_x = np.argwhere(medi)[:, 1].mean()
    side = np.where(np.arange(medi.shape[1]) < center_x, sides[0], sides[1])
    reach = profile[:, None, None] * side[None, :, None]
    full = lungs & (distance <= width * reach)
    grown = lungs & (distance <= width * fraction * reach)
    inner = mask_box(full, 2)
    if inner is None:
        return None
    absolute = tuple(slice(outer.start + item.start, outer.start + item.stop) for outer, item in zip(box, inner))
    delta = _replace_in(ctx, absolute, grown[inner], HU_SOFT, rng)
    return _finish(ctx, delta, "mediastinal_widening", "T2", "mediastinum",
                   dict(width_mm=width * fraction, side_weights=sides.tolist()))


def mediastinal_mass(ctx, rng, fraction=1.0):
    """Mass on the mediastinal border bulging into a lung (soft tissue replacing lung)."""
    center = _compartment_point(ctx, "mediastinum", rng,
                                sorted((ctx.lung_z_fraction(0.05), ctx.lung_z_fraction(0.6))), near_lungs_mm=6.0)
    if center is None:
        return None
    radii = [rng.uniform(0.05, 0.12) * ctx.scale for _ in range(3)]
    box, region = _ball(ctx, center, radii, rng, rotate=True, scale=fraction)
    region &= ctx.mask("body", box) & ~ctx.mask("bone", box)
    if fraction == 1.0 and (region & ctx.mask("lungs", box)).sum() < 0.2 * region.sum():
        return None
    delta = _replace_in(ctx, box, region, HU_SOFT, rng)
    return _finish(ctx, delta, "mediastinal_mass", "T2", "mediastinum", dict(radii_mm=radii))


def mediastinal_shift(ctx, rng, fraction=1.0):
    region = ctx.compartments["mediastinum"].copy()
    if "heart" in ctx.segs:
        region |= ctx.compartments["heart_ds"]
    if not region.any():
        return None
    full = region[ctx._index(ctx.full_box(), ctx.ds)]
    if "bronchi" in ctx.segs:
        full |= ctx.segs["bronchi"] & ~ctx.segs["lungs"]
    shift = rng.uniform(4.0, max(5.0, 0.05 * ctx.scale)) * (1 if rng.random() < 0.5 else -1)
    box = mask_box(full, ctx.vox(abs(shift)) + 4)
    weight = feather(dilate(full[box], abs(shift), ctx.spacing), abs(shift), ctx.spacing)
    weight /= weight.max() + 1e-6
    warped = displacement(ctx, box, ctx.ct[box], weight, (0.0, shift * fraction, 0.0))
    delta = bounded_delta(ctx.ct, box, warped)
    return _finish(ctx, delta, "mediastinal_shift", "T2", "mediastinum", dict(shift_mm=shift * fraction))


def _trachea(ctx):
    airways = ctx.segs.get("bronchi")
    if airways is None or not airways.any():
        return None
    labels, count = ndi.label(airways & ~ctx.segs["lungs"])
    if not count:
        return None
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, count + 1))
    return labels == int(np.argmax(sizes)) + 1


def tracheal_deviation(ctx, rng, fraction=1.0):
    trachea = _trachea(ctx)
    if trachea is None or trachea.sum() * ctx.voxel_mm3 < 100:
        return None
    shift_mm = rng.uniform(4.0, max(5.0, 0.05 * ctx.scale)) * (1 if rng.random() < 0.5 else -1)
    shift = int(round(shift_mm * fraction / ctx.spacing[1]))
    box = mask_box(trachea, ctx.vox(abs(shift_mm)) + 3)
    old = trachea[box]
    new = np.roll(old, shift, axis=1)
    target = ctx.ct[box].copy()
    target[old & ~new] = HU_SOFT
    target[new & ~old] = HU_AIR
    region = dilate(old | new, 1.0, ctx.spacing)
    delta = bounded_delta(ctx.ct, box, _smooth_inside(ctx, target, region, box))
    return _finish(ctx, delta, "tracheal_deviation", "T2", "airways", dict(shift_mm=shift_mm * fraction))


def _airway_region(ctx, rng, radius_fraction=(0.06, 0.15)):
    airways = ctx.segs.get("bronchi")
    if airways is None:
        return None
    center = _point(ctx, np.argwhere(ctx._down(airways & ctx.segs["lungs"])) * ctx.ds, rng)
    if center is None:
        return None
    radius = rng.uniform(*radius_fraction) * ctx.scale
    box, ball = _ball(ctx, center, radius)
    local = airways[box] & ball
    return (box, local) if local.sum() * ctx.voxel_mm3 > 20 else None


def bronchial_dilation(ctx, rng, fraction=1.0):
    region = _airway_region(ctx, rng)
    if region is None:
        return None
    box, airways = region
    width = rng.uniform(1.5, 4.0) * fraction
    grown = dilate(airways, width, ctx.spacing) & ~airways & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, grown, HU_AIR, rng, sigma_mm=0.7, noise=5)
    return _finish(ctx, delta, "bronchial_dilation", "T2", "airways", dict(width_mm=width))


def mucus_plug(ctx, rng):
    """Bronchi fill with mucus plus a peribronchial cuff (branching 'finger-in-glove' opacity)."""
    region = _airway_region(ctx, rng, (0.08, 0.18))
    if region is None:
        return None
    box, airways = region
    cuff = dilate(airways, rng.uniform(1.5, 3.5), ctx.spacing) & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, airways | cuff, HU_FLUID, rng, sigma_mm=0.7)
    return _finish(ctx, delta, "mucus_plug", "T2", "airways", {})


def hilar_mass(ctx, rng, fraction=1.0):
    center = _compartment_point(ctx, "hila", rng)
    if center is None:
        return None
    radii = [rng.uniform(0.03, 0.08) * ctx.scale for _ in range(3)]
    box, region = _ball(ctx, center, radii, rng, rotate=True, scale=fraction)
    delta = _replace_in(ctx, box, region & ctx.mask("body", box), HU_SOFT, rng)
    return _finish(ctx, delta, "hilar_mass", "T2", "hila", dict(radii_mm=radii))


def pa_dilation(ctx, rng, fraction=1.0):
    vessels = ctx.segs.get("lung_vessels")
    center = _compartment_point(ctx, "hila", rng)
    if vessels is None or center is None:
        return None
    radius = rng.uniform(0.06, 0.12) * ctx.scale
    box, ball = _ball(ctx, center, radius)
    local = vessels[box] & ball
    if local.sum() * ctx.voxel_mm3 < 20:
        return None
    width = rng.uniform(1.5, 4.0) * fraction
    grown = dilate(local, width, ctx.spacing) & ~local & ctx.mask("body", box)
    delta = _replace_in(ctx, box, grown, HU_SOFT, rng, sigma_mm=0.7)
    return _finish(ctx, delta, "pa_dilation", "T2", "hila", dict(width_mm=width))


def _base_depth(ctx, lung):
    """Signed z distance (mm) above (+) / below (-) the diaphragmatic lung base, dome columns only."""
    z = np.arange(lung.shape[0])[:, None, None]
    columns = ctx.dome_columns(lung)
    if ctx.inferior > 0:
        base = lung.shape[0] - 1 - np.argmax(lung[::-1], 0)
        depth = base[None] - z
    else:
        base = np.argmax(lung, 0)
        depth = z - base[None]
    return depth * ctx.spacing[0], columns


def diaphragm_shift(ctx, rng, fraction=1.0):
    """Unilateral dome elevation (lung base -> soft tissue) or depression (soft -> lung)."""
    side = "right" if rng.random() < 0.5 else "left"
    lung = ctx.segs[f"lung_{side}"]
    full_height = rng.uniform(0.03, 0.10) * ctx.lung_height_mm
    height = full_height * fraction
    elevate = rng.random() < 0.5
    box = mask_box(lung, ctx.vox(full_height) + 3)
    local = lung[box]
    depth, columns = _base_depth(ctx, local)
    if elevate:
        region = local & (depth >= 0) & (depth < height) & columns[None]
        delta = _replace_in(ctx, box, region, HU_SOFT, rng)
    else:
        region = (ctx.mask("body", box) & ~local & (depth < 0) & (depth >= -height) & columns[None]
                  & ~ctx.mask("bone", box) & ~ctx.mask("lungs", box))
        if "heart" in ctx.segs:
            region &= ~ctx.mask("heart", box)
        delta = _replace_in(ctx, box, region, HU_LUNG, rng, noise=25)
    return _finish(ctx, delta, "diaphragm_shift", "T2", "sub_diaphragm",
                   dict(side=side, elevate=elevate, height_mm=height))


def free_air(ctx, rng, fraction=1.0):
    """Air crescent under a dome, separated from the lung base by the diaphragm (gap_mm)."""
    side = "right" if rng.random() < 0.6 else "left"
    lung = ctx.segs[f"lung_{side}"]
    gap, full_thickness = rng.uniform(3.0, 6.0), rng.uniform(3.0, 10.0)
    thickness = full_thickness * fraction
    box = mask_box(lung, ctx.vox(gap + full_thickness) + 3)
    depth, columns = _base_depth(ctx, lung[box])
    allowed = ctx.mask("body", box) & ~ctx.mask("lungs", box) & ~ctx.mask("bone", box) & columns[None]
    if "heart" in ctx.segs:
        allowed &= ~ctx.mask("heart", box)
    region = allowed & (depth <= -gap) & (depth > -gap - thickness)
    inner = mask_box(allowed & (depth <= -gap) & (depth > -gap - full_thickness), 2)
    if inner is None:
        return None
    absolute = tuple(slice(outer.start + item.start, outer.start + item.stop) for outer, item in zip(box, inner))
    delta = _replace_in(ctx, absolute, region[inner], HU_AIR, rng, sigma_mm=0.7, noise=5)
    return _finish(ctx, delta, "free_air", "T2", "sub_diaphragm",
                   dict(side=side, gap_mm=gap, thickness_mm=thickness))


def gastric_bubble(ctx, rng, fraction=1.0):
    """Air pocket below a dome (mostly left), kept off the diaphragm so the dome line stays."""
    side = "left" if rng.random() < 0.8 else "right"
    dome = ctx.dome_columns(ctx._down(ctx.segs[f"lung_{side}"]))
    away = ~dilate(ctx.compartments["lungs_ds"], 6.0, ctx.ds_spacing)
    mask = ctx.compartments["sub_diaphragm"] & dome[None] & away
    center = _point(ctx, ctx.full_points(mask) if mask.any() else None, rng)
    if center is None:
        return None
    radii = [rng.uniform(0.03, 0.08) * ctx.scale for _ in range(3)]
    box, region = _ball(ctx, center, radii, region="sub_diaphragm", scale=fraction)
    region &= ~dilate(ctx.segs["lungs"][box], 4.0, ctx.spacing)
    delta = _replace_in(ctx, box, region, HU_AIR, rng, sigma_mm=1.0, noise=5)
    return _finish(ctx, delta, "gastric_bubble", "T2", "sub_diaphragm", dict(side=side, radii_mm=radii))


def subcutaneous_emphysema(ctx, rng):
    region = sample_region(ctx, "chest_wall", rng, (0.08, 0.2), sampler="noise")
    if region is None:
        return None
    box, mask = region
    streaks = np.abs(smooth_noise(mask.shape, 2.0, rng, ctx.spacing)) < rng.uniform(0.15, 0.35)
    delta = _replace_in(ctx, box, mask & streaks, -950.0, rng, sigma_mm=0.5, noise=20)
    return _finish(ctx, delta, "subcutaneous_emphysema", "T2", "chest_wall", {})


def chest_wall_thickening(ctx, rng, fraction=1.0):
    compartment = "subcutaneous" if ctx.available("subcutaneous") else "chest_wall"
    region = sample_region(ctx, compartment, rng, (0.05, 0.15))
    if region is None:
        return None
    box, mask = region
    body = ctx.mask("body", box)
    width = rng.uniform(6.0, max(8.0, 0.08 * ctx.scale)) * fraction  # en face soft tissue needs ~1 cm to show on AP
    grown = dilate(mask, width, ctx.spacing) & ~body
    delta = _replace_in(ctx, box, grown | (mask & ~body), HU_SOFT, rng)
    return _finish(ctx, delta, "chest_wall_thickening", "T2", "chest_wall", dict(width_mm=width))


def rib_lesion(ctx, rng, fraction=1.0):
    bone = ctx.compartments["bone"] & ctx.compartments["chest_wall"]
    center = _point(ctx, ctx.full_points(bone) if bone.any() else None, rng)
    if center is None:
        return None
    radius = rng.uniform(0.03, 0.06) * ctx.scale
    box, region = _ball(ctx, center, radius, region="bone", scale=fraction)
    lytic = rng.random() < 0.6
    delta = _replace_in(ctx, box, region, HU_FAT if lytic else rng.uniform(1000, 1500), rng, sigma_mm=0.7)
    return _finish(ctx, delta, "rib_lesion", "T2", "chest_wall", dict(lytic=lytic, radius_mm=radius))


def _lung_ball(ctx, rng, radius_range):
    if not hasattr(ctx, "_interior_lung"):
        interior = erode(ctx._down(ctx.segs["lungs"]), 2 * ctx.ds_spacing.max(), ctx.ds_spacing)
        ctx._interior_lung = ctx.full_points(interior)
    center = _point(ctx, ctx._interior_lung, rng)
    if center is None:
        return None
    return center, rng.uniform(*radius_range) * ctx.scale


def nodule(ctx, rng, fraction=1.0):
    found = _lung_ball(ctx, rng, (0.02, 0.07))
    if found is None:
        return None
    center, radius = found
    radii = [radius * rng.uniform(0.7, 1.3) for _ in range(3)]
    box, region = _ball(ctx, center, radii, rng, region="lungs", rotate=True, scale=fraction)
    region &= smooth_noise(region.shape, max(1.0, radius / 3), rng, ctx.spacing) > -1.2
    delta = _replace_in(ctx, box, region, HU_SOFT, rng)
    return _finish(ctx, delta, "nodule", "T2", "lungs", dict(radius_mm=radius * fraction))


def cavity(ctx, rng, fraction=1.0):
    found = _lung_ball(ctx, rng, (0.03, 0.07))
    if found is None:
        return None
    center, radius = found
    box, outer = _ball(ctx, center, radius, region="lungs", scale=fraction)
    inner = ellipsoid(box, center, radius * fraction * rng.uniform(0.4, 0.75), ctx.spacing)
    target = ctx.ct[box].copy()
    target[outer] = HU_SOFT
    target[inner & outer] = HU_AIR
    delta = bounded_delta(ctx.ct, box, _smooth_inside(ctx, target, dilate(outer, 1.0, ctx.spacing), box))
    return _finish(ctx, delta, "cavity", "T2", "lungs", dict(radius_mm=radius * fraction))


def bulla(ctx, rng, fraction=1.0):
    # Lucent interior and dense wall partly cancel along the ray, so bullae must be large to show.
    found = _lung_ball(ctx, rng, (0.04, 0.1))
    if found is None:
        return None
    center, radius = found
    box, outer = _ball(ctx, center, radius, region="lungs", scale=fraction)
    wall = outer & ~erode(outer, 1.0, ctx.spacing)
    target = ctx.ct[box].copy()
    target[outer] = HU_AIR
    target[wall] = 0.0
    return _finish(ctx, bounded_delta(ctx.ct, box, target), "bulla", "T2", "lungs",
                   dict(radius_mm=radius * fraction))


def calcified_dot(ctx, rng):
    found = _lung_ball(ctx, rng, (0.01, 0.02))
    if found is None:
        return None
    center, radius = found
    radius = float(np.clip(radius, 3.0, 6.0))
    box, region = _ball(ctx, center, radius, region="lungs")
    delta = _replace_in(ctx, box, region, rng.uniform(800, 1500), rng, sigma_mm=0.5)
    return _finish(ctx, delta, "calcified_dot", "T2", "lungs", dict(radius_mm=radius))


def lobar_hyperlucency(ctx, rng):
    lobes = [name for name in LOBES if name in ctx.segs and ctx.segs[name].sum() * ctx.voxel_mm3 > 1000]
    if not lobes:
        return None
    lobe = str(rng.choice(lobes))
    box = mask_box(ctx.segs[lobe], 2)
    amount = rng.uniform(80, 180)  # lung parenchyma is ~-850 HU, so the air clamp caps the effect
    target = ctx.ct[box] - amount * feather(ctx.segs[lobe][box], 3.0, ctx.spacing)
    return _finish(ctx, bounded_delta(ctx.ct, box, target), "lobar_hyperlucency", "T2", "lungs",
                   dict(lobe=lobe, amount=amount))


# ---------------------------------------------------------------------------- T2-near families
def lobar_collapse(ctx, rng, fraction=1.0):
    lobes = [name for name in LOBES if name in ctx.segs and ctx.segs[name].sum() * ctx.voxel_mm3 > 1000]
    hila = ctx.compartments["hila"]
    if not lobes or not hila.any():
        return None
    lobe = str(rng.choice(lobes))
    box = mask_box(ctx.segs[lobe], 2)
    region = ctx.segs[lobe][box]
    hilum = ctx.full_points(hila).mean(0) - np.array([item.start for item in box])
    z, x, y = box_grid(tuple(slice(0, item.stop - item.start) for item in box))
    distance = np.sqrt(((z - hilum[0]) * ctx.spacing[0]) ** 2 + ((x - hilum[1]) * ctx.spacing[1]) ** 2
                       + ((y - hilum[2]) * ctx.spacing[2]) ** 2)
    limit = np.quantile(distance[region], rng.uniform(0.25, 0.5) * fraction)
    collapsed = region & (distance <= limit)
    target = ctx.ct[box] + rng.uniform(400, 800) * feather(collapsed, 2.0, ctx.spacing)
    return _finish(ctx, bounded_delta(ctx.ct, box, target), "lobar_collapse", "T2near", "lungs", dict(lobe=lobe))


def fissural_fluid(ctx, rng):
    center = _compartment_point(ctx, "fissures", rng)
    if center is None:
        return None
    radius = rng.uniform(0.06, 0.15) * ctx.scale
    box, ball = _ball(ctx, center, (radius, radius, radius * 0.6))
    sheet = dilate(ctx.mask("fissures", box) & ball, rng.uniform(1.0, 3.0), ctx.spacing) & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, sheet, HU_FLUID, rng, sigma_mm=0.7)
    return _finish(ctx, delta, "fissural_fluid", "T2near", "fissures", {})


def pleural_thickening(ctx, rng):
    region = sample_region(ctx, "pleural_rim", rng, (0.05, 0.15), sampler="noise")
    if region is None:
        return None
    box, mask = region
    return _finish(ctx, _replace_in(ctx, box, mask, HU_SOFT, rng), "pleural_thickening", "T2near", "pleural_rim", {})


def pneumomediastinum(ctx, rng):
    mediastinum = ctx.compartments["mediastinum"]
    if not mediastinum.any():
        return None
    near_lungs = dilate(ctx._down(ctx.segs["lungs"]), 2 * ctx.ds_spacing.max(), ctx.ds_spacing)
    border_ds = mediastinum & near_lungs
    small = mask_box(border_ds, 1)
    if small is None:
        return None
    box = tuple(slice(item.start * factor, min(size, item.stop * factor))
                for item, factor, size in zip(small, ctx.ds, ctx.shape))
    border = ctx.mask("mediastinum", box) & dilate(ctx.segs["lungs"][box], 2 * ctx.ds_spacing.max(), ctx.spacing)
    streaks = np.abs(smooth_noise(border.shape, 1.5, rng, ctx.spacing)) < rng.uniform(0.2, 0.4)
    delta = _replace_in(ctx, box, border & streaks, -950.0, rng, sigma_mm=0.5)
    return _finish(ctx, delta, "pneumomediastinum", "T2near", "mediastinum", {})


def bronchial_wall_thickening(ctx, rng):
    region = _airway_region(ctx, rng, (0.1, 0.25))
    if region is None:
        return None
    box, airways = region
    wall = dilate(airways, rng.uniform(1.0, 2.5), ctx.spacing) & ~airways & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, wall, HU_SOFT, rng, sigma_mm=0.7)
    return _finish(ctx, delta, "bronchial_wall_thickening", "T2near", "airways", {})


def vessel_change(ctx, rng):
    vessels = ctx.segs.get("lung_vessels")
    if vessels is None:
        return None
    center = _point(ctx, np.argwhere(ctx._down(vessels & ctx.segs["lungs"])) * ctx.ds, rng)
    if center is None:
        return None
    radius = rng.uniform(0.08, 0.2) * ctx.scale
    box, ball = _ball(ctx, center, radius)
    local = vessels[box] & ball
    if local.sum() * ctx.voxel_mm3 < 20:
        return None
    if rng.random() < 0.5:
        region, value = dilate(local, rng.uniform(1.0, 2.5), ctx.spacing) & ~local & ctx.mask("lungs", box), HU_SOFT
    else:
        region, value = local, HU_LUNG
    delta = _replace_in(ctx, box, region, value, rng, sigma_mm=0.7)
    return _finish(ctx, delta, "vessel_change", "T2near", "lungs", {})


# ---------------------------------------------------------------------------- nuisance
def inspiration_nuisance(ctx, rng):
    """Bilateral breathing-depth change: both domes move and lung density shifts together."""
    height = rng.uniform(0.02, 0.06) * ctx.lung_height_mm
    deeper = rng.random() < 0.5
    box = mask_box(ctx.segs["lungs"], ctx.vox(height) + 3)
    target = ctx.ct[box].copy()
    target[ctx.segs["lungs"][box]] -= (1 if deeper else -1) * rng.uniform(15, 50)
    body = ctx.mask("body", box)
    for side in ("right", "left"):
        lung = ctx.segs[f"lung_{side}"][box]
        depth, columns = _base_depth(ctx, lung)
        if deeper:
            target[~lung & (depth < 0) & (depth >= -height) & columns[None] & body] = HU_LUNG
        else:
            target[lung & (depth >= 0) & (depth < height) & columns[None]] = HU_SOFT
    return bounded_delta(ctx.ct, box, target), dict(height_mm=height, deeper=deeper)


# ---------------------------------------------------------------------------- catalog
def _t1(compartment, mechanism):
    return lambda ctx, rng, donor=None: t1_change(ctx, rng, compartment, mechanism, donor)


FAMILIES: Dict[str, dict] = {}
for _compartment in T1_COMPARTMENTS:
    for _mechanism in T1_MECHANISMS:
        FAMILIES[f"t1_{_mechanism}@{_compartment}"] = dict(
            tier="T1", compartment=_compartment, fn=_t1(_compartment, _mechanism),
            requires=("heart",) if _compartment == "heart" else (), analogue="none (primitive)")

for _name, _fn, _tier, _compartment, _requires, _analogue in (
        ("heart_resize", heart_resize, "T2", "heart", ("heart",), "cardiomegaly / heart size change"),
        ("pericardial_shell", pericardial_shell, "T2", "heart", ("heart",), "pericardial effusion"),
        ("mediastinal_widening", mediastinal_widening, "T2", "mediastinum", (), "mediastinal widening"),
        ("mediastinal_mass", mediastinal_mass, "T2", "mediastinum", (), "mediastinal mass"),
        ("mediastinal_shift", mediastinal_shift, "T2", "mediastinum", (), "mediastinal shift"),
        ("tracheal_deviation", tracheal_deviation, "T2", "airways", ("bronchi",), "tracheal deviation"),
        ("bronchial_dilation", bronchial_dilation, "T2", "airways", ("bronchi",), "bronchiectasis"),
        ("mucus_plug", mucus_plug, "T2", "airways", ("bronchi",), "mucus plugging"),
        ("hilar_mass", hilar_mass, "T2", "hila", (), "hilar lymphadenopathy"),
        ("pa_dilation", pa_dilation, "T2", "hila", ("lung_vessels",), "pulmonary artery enlargement"),
        ("diaphragm_shift", diaphragm_shift, "T2", "sub_diaphragm", (), "hemidiaphragm elevation"),
        ("free_air", free_air, "T2", "sub_diaphragm", (), "subdiaphragmatic free air"),
        ("gastric_bubble", gastric_bubble, "T2", "sub_diaphragm", (), "gastric bubble / air pocket"),
        ("subcutaneous_emphysema", subcutaneous_emphysema, "T2", "chest_wall", (), "subcutaneous emphysema"),
        ("chest_wall_thickening", chest_wall_thickening, "T2", "chest_wall", (), "chest wall hematoma"),
        ("rib_lesion", rib_lesion, "T2", "chest_wall", (), "rib lesion"),
        ("nodule", nodule, "T2", "lungs", (), "nodule / mass"),
        ("cavity", cavity, "T2", "lungs", (), "cavitation"),
        ("bulla", bulla, "T2", "lungs", (), "bulla / cyst"),
        ("calcified_dot", calcified_dot, "T2", "lungs", (), "calcified granuloma"),
        ("lobar_hyperlucency", lobar_hyperlucency, "T2", "lungs", (), "air trapping / oligemia"),
        ("lobar_collapse", lobar_collapse, "T2near", "lungs", (), "atelectasis / collapse"),
        ("fissural_fluid", fissural_fluid, "T2near", "fissures", (), "fissural fluid"),
        ("pleural_thickening", pleural_thickening, "T2near", "pleural_rim", (), "pleural thickening"),
        ("pneumomediastinum", pneumomediastinum, "T2near", "mediastinum", (), "pneumomediastinum"),
        ("bronchial_wall_thickening", bronchial_wall_thickening, "T2near", "airways", ("bronchi",),
         "peribronchial cuffing"),
        ("vessel_change", vessel_change, "T2near", "lungs", ("lung_vessels",), "vascular change")):
    FAMILIES[_name] = dict(tier=_tier, compartment=_compartment, fn=_fn, requires=_requires,
                           analogue=_analogue)

# How the smaller timepoint of a progressing / regressing change is built:
#   param     replay the same random draws with the size / shift / scale multiplied by fraction
#   intensity the same edit at fraction strength
#   spatial   the same edit restricted to part of each focus (a lesion spreading from a seed)
PARAM_PROGRESSION = ("heart_resize", "pericardial_shell", "mediastinal_widening", "mediastinal_mass",
                     "mediastinal_shift", "tracheal_deviation", "bronchial_dilation", "hilar_mass", "pa_dilation",
                     "diaphragm_shift", "free_air", "gastric_bubble", "chest_wall_thickening", "rib_lesion",
                     "nodule", "cavity", "bulla", "lobar_collapse")
for _name, _spec in FAMILIES.items():
    if _name in PARAM_PROGRESSION:
        _spec["progression"] = "param"
    elif _name in ("calcified_dot", "lobar_hyperlucency"):
        _spec["progression"] = "intensity"
    elif _name.startswith("t1_displace") or _name in ("subcutaneous_emphysema", "pneumomediastinum", "vessel_change"):
        _spec["progression"] = "spatial"  # a weaker warp would ghost (double edges)
    else:
        _spec["progression"] = "spatial_or_intensity"

# Timepoint of each change. appear: current only; resolve: prior only; progress / regress: in both,
# smaller in the prior / current; stable: identical in both (GT 0, never an episode target).
TIMEPOINT_MODES = dict(appear=0.30, resolve=0.30, progress=0.125, regress=0.125, stable=0.15)
MODE_ALIASES = dict(current="appear", prior="resolve")
PROGRESSION_FRACTION = (0.35, 0.75)


def family_names(tiers=("T1", "T2")):
    return sorted(name for name, spec in FAMILIES.items() if spec["tier"] in tiers)


def sample_change(ctx, family, rng, donor=None, attempts=10) -> Optional[ChangeSpec]:
    """``donor`` is a CT patch or a callable rng -> patch (a fresh donor per attempt)."""
    spec = FAMILIES[family]
    if any(not ctx.available(name) for name in spec["requires"]):
        return None
    for _ in range(attempts):
        if family.startswith("t1_"):
            patch = donor(rng) if callable(donor) else donor
            state = rng.bit_generator.state
            change = spec["fn"](ctx, rng, patch)
        else:
            patch, state = None, rng.bit_generator.state
            change = spec["fn"](ctx, rng)
        if change is not None:
            change.family = family
            change.replay = (state, patch)
            return change
    return None


def negate(delta):
    return VolumeDelta(delta.box, -delta.values)


def timepoint_deltas(change, mode):
    """(prior delta, current delta) of a change in a timepoint mode; None = absent."""
    mode = MODE_ALIASES.get(mode, mode)
    if mode in ("progress", "regress") and change.partial is None:
        raise ValueError(f"{change.family}: mode {mode} needs a partial delta")
    return dict(appear=(None, change.delta), resolve=(change.delta, None), progress=(change.partial, change.delta),
                regress=(change.delta, change.partial), stable=(change.delta, change.delta))[mode]


def spatial_partial(ctx, delta, rng, fraction):
    """The same edit on part of each focus: a ball around a random seed holding ~fraction of the
    focus volume; with several foci, each non-first focus is absent with probability 0.3 (new lesion)."""
    support = delta.support(5.0)
    step = np.maximum(1, np.round(2.0 / ctx.spacing)).astype(int)
    coarse = ndi.binary_dilation(support[::step[0], ::step[1], ::step[2]], iterations=2)
    labels, count = ndi.label(coarse)
    if not count:
        return None
    up = np.ix_(*[np.minimum(np.arange(size) // factor, limit - 1)
                  for size, factor, limit in zip(support.shape, step, labels.shape)])
    full_labels = np.where(support, labels[up], 0)
    z, x, y = (np.arange(size, dtype=np.float32) * np.float32(spacing)
               for size, spacing in zip(support.shape, ctx.spacing))
    keep = np.zeros(support.shape, bool)
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, count + 1))
    for rank, label in enumerate(np.argsort(sizes)[::-1][:6] + 1):
        if rank > 0 and rng.random() < 0.3:
            continue
        component = full_labels == label
        points = np.argwhere(component)
        if not len(points):
            continue
        seed = points[rng.integers(len(points))]
        distance = np.sqrt((z - z[seed[0]])[:, None, None] ** 2 + (x - x[seed[1]])[None, :, None] ** 2
                           + (y - y[seed[2]])[None, None, :] ** 2)
        limit = np.quantile(distance[component], fraction)
        keep |= component & (distance <= limit)
    if not keep.any():
        return None
    weight = np.clip(feather(keep, 1.5, ctx.spacing) * 2.0, 0.0, 1.0)
    return VolumeDelta(delta.box, (delta.values * weight).astype(np.float32))


def make_partial(ctx, change, rng, fraction):
    """Smaller version of a change (see PROGRESSION_FRACTION); returns (delta or None, method)."""
    method = FAMILIES[change.family]["progression"]
    if method == "spatial_or_intensity":
        method = "spatial" if rng.random() < 0.5 else "intensity"
    if method == "intensity":
        return VolumeDelta(change.delta.box, (change.delta.values * fraction).astype(np.float32)), method
    if method == "spatial":
        return spatial_partial(ctx, change.delta, rng, fraction), method
    state, _ = change.replay
    replay = np.random.default_rng()
    replay.bit_generator.state = state
    saved, ctx.min_visibility = ctx.min_visibility, None  # only the difference has to be visible
    try:
        partial = FAMILIES[change.family]["fn"](ctx, replay, fraction=fraction)
    finally:
        ctx.min_visibility = saved
    return (partial.delta if partial is not None else None), method


def _visible(ctx, delta):
    if ctx.min_visibility is None:
        return bool(delta.support().any())
    peak, area = ctx.visibility(delta)
    return peak >= ctx.min_visibility[0] and area >= ctx.min_visibility[1]


def smaller_version(ctx, change, rng):
    """Smaller timepoint for progress / regress: a random fraction, then the smallest fraction
    (largest difference). Returns (partial or None, method, fraction, visible difference)."""
    partial, method, fraction = None, FAMILIES[change.family]["progression"], PROGRESSION_FRACTION[0]
    for fraction in (float(rng.uniform(*PROGRESSION_FRACTION)), PROGRESSION_FRACTION[0]):
        partial, method = make_partial(ctx, change, rng, fraction)
        if partial is not None and _visible(ctx, merge_deltas([change.delta, negate(partial)])):
            return partial, method, fraction, True
    return partial, method, fraction, False


def assign_timepoint(ctx, change, rng, modes=None):
    """Sample a timepoint mode; progress / regress fall back to appear / resolve when no visibly
    different smaller version exists. Sets change.partial and records fraction / method."""
    modes = TIMEPOINT_MODES if modes is None else modes
    names = list(modes)
    probabilities = np.array([modes[name] for name in names], np.float64)
    mode = names[int(rng.choice(len(names), p=probabilities / probabilities.sum()))]
    if mode in ("progress", "regress"):
        partial, method, fraction, visible = smaller_version(ctx, change, rng)
        if visible:
            change.partial = partial
            change.params.update(fraction=round(fraction, 3), progression=method)
            return mode
        mode = "appear" if mode == "progress" else "resolve"
    return mode


def compose_scene(ctx, rng, tiers=("T1", "T2"), n_changes=(1, 3), margin_mm=3.0,
                  donor_fn: Optional[Callable] = None, family_weights=None, attempts=40, modes=None):
    """Sample voxel-disjoint changes (>= margin_mm apart) of distinct families; each gets a timepoint
    mode (TIMEPOINT_MODES). Returns [(ChangeSpec, mode), ...]."""
    names = [name for name in family_names(tiers)
             if all(ctx.available(item) for item in FAMILIES[name]["requires"])]
    if not names:
        return []
    weights = np.array([1.0 if family_weights is None else family_weights.get(name, 1.0) for name in names])
    target = int(rng.integers(n_changes[0], n_changes[1] + 1))
    occupied = np.zeros(ctx.shape, bool)
    changes: List[Tuple[ChangeSpec, str]] = []
    used = set()
    for _ in range(attempts):
        if len(changes) >= target:
            break
        probabilities = weights * np.array([name not in used for name in names])
        if probabilities.sum() <= 0:
            break
        family = names[int(rng.choice(len(names), p=probabilities / probabilities.sum()))]
        donor = donor_fn if donor_fn is not None and "cutpaste" in family else None
        change = sample_change(ctx, family, rng, donor)
        if change is None:
            continue
        if (dilate_box(ctx, occupied, change.delta.box, margin_mm) & change.delta.support()).any():
            continue
        mode = assign_timepoint(ctx, change, rng, modes)
        if change.partial is not None and mode in ("progress", "regress") and (
                dilate_box(ctx, occupied, change.partial.box, margin_mm) & change.partial.support()).any():
            change.partial, mode = None, ("appear" if mode == "progress" else "resolve")
        for part in (change.delta, change.partial) if mode in ("progress", "regress") else (change.delta,):
            occupied[part.box] |= part.support()
        used.add(family)
        changes.append((change, mode))
    return changes


def dilate_box(ctx, mask, box, margin_mm):
    """mask dilated by margin_mm, evaluated inside box."""
    margin = np.ceil(ctx.vox(margin_mm)).astype(int)
    grown = tuple(slice(max(0, item.start - m), min(size, item.stop + m))
                  for item, m, size in zip(box, margin, mask.shape))
    local = dilate(mask[grown], margin_mm, ctx.spacing)
    offset = tuple(slice(item.start - outer.start, item.stop - outer.start) for item, outer in zip(box, grown))
    return local[offset]
