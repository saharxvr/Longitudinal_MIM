"""Synthetic thoracic change tasks from CT segmentations (few-shot training corpus).

Every change is an additive HU delta on the (cropped, clamped) CT. Because the DRR is a
linear ray-sum after rotation (see supervoxel_render.py), each change can be projected on
its own, so every scene has an exact per-change signed GT map.

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


# ---------------------------------------------------------------------------- geometry
def box_around(center, radii, shape, margin=0):
    return tuple(slice(max(0, int(c - r - margin)), min(s, int(c + r + margin) + 1))
                 for c, r, s in zip(center, radii, shape))


def mask_box(mask, margin, shape=None):
    """Bounding box (tuple of slices) of a boolean mask with a margin; None if empty."""
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return None
    shape = mask.shape if shape is None else shape
    low, high = coordinates.min(0), coordinates.max(0) + 1
    return tuple(slice(max(0, int(a) - margin), min(s, int(b) + margin))
                 for a, b, s in zip(low, high, shape))


def box_grid(box):
    return np.ogrid[tuple(slice(item.start, item.stop) for item in box)]


def ellipsoid(box, center, radii, rotation=None):
    z, x, y = box_grid(box)
    offsets = [z - center[0], x - center[1], y - center[2]]
    if rotation is not None:
        offsets = [sum(rotation[row, column] * offsets[column] for column in range(3))
                   for row in range(3)]
    return sum((offset / max(radius, 1e-3)) ** 2 for offset, radius in zip(offsets, radii)) <= 1


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def dilate(mask, radius):
    if radius <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(~mask) <= radius


def erode(mask, radius):
    if radius <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(mask) > radius


def smooth_noise(shape, sigma, rng):
    field_ = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return field_ / (field_.std() + 1e-6)


def feather(mask, sigma):
    if sigma <= 0:
        return mask.astype(np.float32)
    return ndi.gaussian_filter(mask.astype(np.float32), sigma)


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

    def footprint(self, shape, margin=0):
        mask = np.zeros(shape, bool)
        mask[self.box] = self.support()
        return dilate(mask, margin) if margin else mask


@dataclass
class ChangeSpec:
    family: str
    tier: str
    compartment: str
    effect: str  # denser | lighter | mixed (HU direction in the modified volume)
    delta: VolumeDelta
    params: Dict = field(default_factory=dict)


def bounded_delta(base, box, target):
    """Delta that moves base[box] to target while respecting the projection HU bounds."""
    current = base[box]
    values = np.clip(target, HU_MIN, HU_MAX) - current
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
    """Cropped CT (HU >= -1000), full-resolution segmentations and downsampled compartments."""

    def __init__(self, ct, segs, downsample=2, supervoxel_downsample=4, supervoxels=None,
                 n_supervoxels=1500, seed=0):
        self.ct = np.clip(np.asarray(ct, np.float32), HU_MIN, HU_MAX)
        self.shape = self.ct.shape
        self.ds = int(downsample)
        self.sv_ds = int(supervoxel_downsample)
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
        self.supervoxels = (supervoxels if supervoxels is not None
                            else compute_supervoxels(self.ct, self.compartments["body"],
                                                     self.ds, self.sv_ds, n_supervoxels, seed))
        self.adjacency = supervoxel_adjacency(self.supervoxels)

    # downsampled helpers
    def _down(self, mask):
        return mask[::self.ds, ::self.ds, ::self.ds]

    def _index(self, box, factor):
        return np.ix_(*[np.minimum(np.arange(item.start, item.stop) // factor, size - 1)
                        for item, size in zip(box, self._ds_shape(factor))])

    def _ds_shape(self, factor):
        return tuple((size + factor - 1) // factor for size in self.shape)

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

    def _geometry(self):
        lungs = self.segs["lungs"]
        area = lungs.sum((1, 2)).astype(np.float64)
        z = np.nonzero(area)[0]
        self.z0, self.z1 = int(z[0]), int(z[-1]) + 1
        height = self.z1 - self.z0
        band = max(1, height // 6)
        top, bottom = area[self.z0:self.z0 + band].mean(), area[self.z1 - band:self.z1].mean()
        # Lung bases are wide and apices narrow: +1 means inferior is increasing z.
        self.inferior = 1 if bottom >= top else -1
        coordinates = np.argwhere(lungs)
        self.lung_center = coordinates.mean(0)
        self.lung_extent = coordinates.max(0) - coordinates.min(0) + 1
        self.scale = float(np.mean(self.lung_extent))
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
            up = np.ix_(*[np.minimum(np.arange(size) // self.ds, ds_size - 1)
                          for size, ds_size in zip(self.shape, labels.shape)])
            right, left = labels[up] == first, labels[up] == second
        self.segs["lung_right"], self.segs["lung_left"] = right, left
        self.side_center = {name: np.argwhere(mask[::4, ::4, ::4]).mean(0) * 4
                            for name, mask in (("right", right), ("left", left))}

    def lung_z_fraction(self, fraction):
        """z index at a fraction from apex (0) to base (1)."""
        if self.inferior > 0:
            return int(self.z0 + fraction * (self.z1 - self.z0))
        return int(self.z1 - 1 - fraction * (self.z1 - self.z0))

    def _compartments(self):
        ct = self._down(self.ct)
        segs = {name: self._down(mask) for name, mask in self.segs.items()}
        lungs = segs["lungs"]
        body = ct > -500
        body = ndi.binary_opening(body, iterations=1)
        labels, count = ndi.label(body)
        if count:
            sizes = ndi.sum(np.ones_like(labels), labels, range(1, count + 1))
            body = labels == (int(np.argmax(sizes)) + 1)
        body = np.stack([ndi.binary_fill_holes(plane) for plane in body]) | lungs
        heart = segs.get("heart", np.zeros_like(body))
        bone = body & (ct > 250) & ~lungs
        z0, z1 = self.z0 // self.ds, max(self.z0 // self.ds + 1, self.z1 // self.ds)
        in_lung_z = np.zeros(body.shape, bool)
        in_lung_z[z0:z1] = True

        right, left = segs["lung_right"], segs["lung_left"]
        x_right = np.argwhere(right)[:, 1].mean()
        x_left = np.argwhere(left)[:, 1].mean()
        low, high = (right, left) if x_right < x_left else (left, right)
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

        distance_to_lungs = ndi.distance_transform_edt(~lungs)
        wall_radius = max(2.0, 0.12 * self.scale / self.ds)
        lung_columns = lungs.any(0)
        if self.inferior > 0:
            base_index = np.where(lung_columns, lungs.shape[0] - 1 - np.argmax(lungs[::-1], 0), -1)
            depth = np.arange(lungs.shape[0])[:, None, None] - base_index[None]
        else:
            base_index = np.where(lung_columns, np.argmax(lungs, 0), lungs.shape[0])
            depth = base_index[None] - np.arange(lungs.shape[0])[:, None, None]
        band = max(2.0, 0.15 * (self.z1 - self.z0) / self.ds)
        sub_diaphragm = body & ~lungs & ~bone & lung_columns[None] & (depth > 0) & (depth <= band)
        chest_wall = (body & ~lungs & ~mediastinum & ~heart & ~sub_diaphragm & in_lung_z
                      & (distance_to_lungs <= wall_radius))
        surface = body & ~erode(body, max(1.0, 0.04 * self.scale / self.ds))
        subcutaneous = chest_wall & surface & (ct > -190) & (ct < -30)
        rim = lungs & ~erode(lungs, max(1.0, 0.015 * self.scale / self.ds))
        fissures = np.zeros_like(body)
        lobes = [segs[name] for name in LOBES if name in segs]
        for index, lobe in enumerate(lobes):
            grown = dilate(lobe, 1.5)
            for other in lobes[index + 1:]:
                fissures |= grown & other
        z_range = sorted((self.lung_z_fraction(0.3) // self.ds, self.lung_z_fraction(0.7) // self.ds))
        central = np.zeros_like(body)
        central[z_range[0]:z_range[1] + 1] = True
        near_mediastinum = dilate(mediastinum | heart, max(2.0, 0.08 * self.scale / self.ds))
        hila = lungs & near_mediastinum & central
        tubes = segs.get("lung_vessels", np.zeros_like(body)) | segs.get("bronchi", np.zeros_like(body))
        if tubes.any():
            hila &= dilate(tubes, 2)
        return dict(body=body, bone=bone, mediastinum=mediastinum, sub_diaphragm=sub_diaphragm,
                    chest_wall=chest_wall, subcutaneous=subcutaneous, pleural_rim=rim,
                    fissures=fissures, hila=hila, lungs_ds=lungs, heart_ds=heart)


def compute_supervoxels(ct, body, compartment_ds, sv_ds, n_segments, seed=0):
    """3D SLIC on the windowed, downsampled CT (computed once per CT; cache the result)."""
    from skimage.segmentation import slic

    image = np.clip((ct[::sv_ds, ::sv_ds, ::sv_ds] + 1000.0) / 1400.0, 0, 1)
    factor = sv_ds // compartment_ds if sv_ds % compartment_ds == 0 else 1
    mask = body[::factor, ::factor, ::factor] if factor > 1 else body
    mask = mask[:image.shape[0], :image.shape[1], :image.shape[2]]
    if mask.shape != image.shape:
        mask = np.pad(mask, [(0, a - b) for a, b in zip(image.shape, mask.shape)])
    labels = slic(image.astype(np.float64), n_segments=n_segments, compactness=0.08,
                  channel_axis=None, mask=mask, start_label=1, enforce_connectivity=True)
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


def sample_region(ctx: CTContext, compartment, rng, size_range=(0.02, 0.12), sampler=None):
    """Random connected supervoxel union (or smooth-noise blob) inside a compartment.

    size is a log-uniform radius fraction of the lung scale; returns (box, mask) or None.
    """
    region_mask = ctx.segs[compartment] if compartment in ctx.segs else None
    sampler = sampler or ("supervoxel" if rng.random() < 0.7 else "noise")
    radius = log_uniform(rng, *size_range) * ctx.scale
    candidates = _candidate_voxels(ctx, compartment, region_mask)
    if candidates is None or not len(candidates):
        return None
    center = candidates[rng.integers(len(candidates))]
    box = box_around(center, [radius * 1.5] * 3, ctx.shape, margin=2)
    inside = ctx.mask(compartment, box)
    if sampler == "noise":
        noise = smooth_noise(inside.shape, max(1.0, radius / 3), rng)
        blob = ellipsoid(box, center, [radius] * 3, random_rotation(rng)) & (noise > rng.uniform(-0.5, 0.5))
        mask = blob & inside
    else:
        labels = ctx.labels(box)
        start = int(ctx.supervoxels[tuple(np.minimum(center // ctx.sv_ds, np.array(ctx.supervoxels.shape) - 1))])
        if start <= 0:
            return None
        target = (4.0 / 3.0) * np.pi * radius ** 3 / ctx.sv_ds ** 3
        chosen, frontier, volume = {start}, [start], 0.0
        sizes = np.bincount(labels.ravel(), minlength=max(labels.max() + 1, start + 1))
        volume += sizes[start] / ctx.sv_ds ** 3 if start < len(sizes) else 0
        while frontier and volume < target:
            node = frontier.pop(int(rng.integers(len(frontier))))
            for neighbor in ctx.adjacency.get(node, ()):
                if neighbor in chosen or neighbor >= len(sizes) or not sizes[neighbor]:
                    continue
                chosen.add(neighbor)
                frontier.append(neighbor)
                volume += sizes[neighbor] / ctx.sv_ds ** 3
                if volume >= target:
                    break
        mask = np.isin(labels, list(chosen)) & inside
    mask = ndi.binary_closing(mask, iterations=1) & inside if mask.any() else mask
    if int(mask.sum()) < 20:
        return None
    return box, mask


def _candidate_voxels(ctx, compartment, region_mask):
    if region_mask is not None:
        coordinates = np.argwhere(region_mask[::4, ::4, ::4]) * 4
        return coordinates if len(coordinates) else None
    mask = ctx.compartments.get(compartment)
    if mask is None:
        return None
    coordinates = np.argwhere(mask) * ctx.ds
    return coordinates if len(coordinates) else None


# ---------------------------------------------------------------------------- mechanisms
def op_density(ctx, box, mask, rng, sign=None):
    sign = sign or (1 if rng.random() < 0.5 else -1)
    soft = feather(mask, rng.uniform(0.5, 2.0))
    amplitude = rng.uniform(150, 600)
    texture = 1 + 0.3 * smooth_noise(mask.shape, 1.5, rng)
    target = ctx.ct[box] + sign * amplitude * soft * texture
    return bounded_delta(ctx.ct, box, target), dict(sign=sign, amplitude=amplitude)


def op_replace(ctx, box, mask, rng, tissue=None):
    current = float(np.median(ctx.ct[box][mask]))
    options = [name for name, value in TISSUES.items() if abs(value - current) > 150]
    tissue = tissue or str(rng.choice(options))
    soft = feather(mask, rng.uniform(0.5, 1.5))
    value = TISSUES[tissue] + 15 * smooth_noise(mask.shape, 1.0, rng)
    target = ctx.ct[box] * (1 - soft) + value * soft
    return bounded_delta(ctx.ct, box, target), dict(tissue=tissue)


def op_grow(ctx, box, mask, rng, compartment):
    radius = rng.uniform(1.5, max(2.0, 0.03 * ctx.scale))
    allowed = ctx.mask("body", box)
    grown = dilate(mask, radius) & allowed & ~mask
    if not grown.any():
        return None, {}
    value = float(np.median(ctx.ct[box][mask]))
    target = ctx.ct[box].copy()
    target[grown] = value + 15 * smooth_noise(mask.shape, 1.0, rng)[grown]
    target = ndi.gaussian_filter(target, 0.7) * (grown | mask) + ctx.ct[box] * ~(grown | mask)
    return bounded_delta(ctx.ct, box, target), dict(radius=radius, fill=value)


def op_shrink(ctx, box, mask, rng, compartment):
    radius = rng.uniform(1.5, max(2.0, 0.03 * ctx.scale))
    removed = mask & ~erode(mask, radius)
    surroundings = dilate(mask, radius + 2) & ~mask & ctx.mask("body", box)
    if not removed.any() or not surroundings.any():
        return None, {}
    value = float(np.median(ctx.ct[box][surroundings]))
    target = ctx.ct[box].copy()
    target[removed] = value + 15 * smooth_noise(mask.shape, 1.0, rng)[removed]
    return bounded_delta(ctx.ct, box, target), dict(radius=radius, fill=value)


def op_displace(ctx, box, mask, rng, compartment):
    """Smooth local displacement (push) of tissue around the region."""
    magnitude = rng.uniform(1.5, max(2.0, 0.04 * ctx.scale))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    weight = feather(dilate(mask, magnitude), magnitude)
    weight /= weight.max() + 1e-6
    coordinates = np.indices(mask.shape, dtype=np.float32)
    moved = [coordinates[axis] - magnitude * direction[axis] * weight for axis in range(3)]
    warped = ndi.map_coordinates(ctx.ct[box], moved, order=1, mode="nearest")
    return bounded_delta(ctx.ct, box, warped), dict(magnitude=magnitude, direction=direction.tolist())


def op_texture(ctx, box, mask, rng):
    kind = str(rng.choice(["fine", "reticular", "nodular"]))
    if kind == "fine":
        pattern = smooth_noise(mask.shape, 0.8, rng)
    elif kind == "reticular":
        pattern = np.abs(smooth_noise(mask.shape, 1.5, rng)) < 0.25
        pattern = pattern.astype(np.float32) * 2 - 0.3
    else:
        pattern = (rng.random(mask.shape) < 0.01).astype(np.float32)
        pattern = ndi.gaussian_filter(pattern, 1.0) * 30
    amplitude = rng.uniform(100, 350)
    target = ctx.ct[box] + amplitude * pattern * feather(mask, 1.0)
    return bounded_delta(ctx.ct, box, target), dict(kind=kind, amplitude=amplitude)


def op_cutpaste(ctx, box, mask, rng, donor=None):
    """Paste HU content from a donor CT patch (another training CT) into the region."""
    if donor is None:
        return None, {}
    patch = donor[tuple(slice(0, min(a, b)) for a, b in zip(donor.shape, mask.shape))]
    if patch.shape != mask.shape:
        patch = np.pad(patch, [(0, a - b) for a, b in zip(mask.shape, patch.shape)], mode="edge")
    soft = feather(mask, 1.0)
    target = ctx.ct[box] * (1 - soft) + patch * soft
    return bounded_delta(ctx.ct, box, target), dict(donor_shape=list(donor.shape))


def t1_change(ctx, rng, compartment, mechanism, donor=None):
    region = sample_region(ctx, compartment, rng)
    if region is None:
        return None
    box, mask = region
    if mechanism == "density":
        delta, params = op_density(ctx, box, mask, rng)
    elif mechanism == "replace":
        delta, params = op_replace(ctx, box, mask, rng)
    elif mechanism == "grow":
        delta, params = op_grow(ctx, box, mask, rng, compartment)
    elif mechanism == "shrink":
        delta, params = op_shrink(ctx, box, mask, rng, compartment)
    elif mechanism == "displace":
        delta, params = op_displace(ctx, box, mask, rng, compartment)
    elif mechanism == "texture":
        delta, params = op_texture(ctx, box, mask, rng)
    elif mechanism == "cutpaste":
        delta, params = op_cutpaste(ctx, box, mask, rng, donor)
    else:
        raise ValueError(f"Unknown T1 mechanism {mechanism}")
    return _finish(delta, f"t1_{mechanism}", "T1", compartment, params)


def _finish(delta, family, tier, compartment, params):
    if delta is None:
        return None
    effect = effect_of(delta.values)
    if effect is None or int(delta.support().sum()) < 20:
        return None
    return ChangeSpec(family, tier, compartment, effect, delta, params)


# ---------------------------------------------------------------------------- T2 families
def _point_in(ctx, mask, rng, step=2):
    coordinates = np.argwhere(mask[::step, ::step, ::step]) * step
    return coordinates[rng.integers(len(coordinates))] if len(coordinates) else None


def _replace_in(ctx, box, region, value, rng, sigma=0.8, noise=12.0):
    target = ctx.ct[box].copy()
    soft = feather(region, sigma)
    filled = value + noise * smooth_noise(region.shape, 1.0, rng)
    target = target * (1 - soft) + filled * soft
    return bounded_delta(ctx.ct, box, target)


def _scale_structure(ctx, rng, name, scale, fill_out):
    """Scale a structure about its centroid; uncovered voxels take fill_out (a HU or callable)."""
    structure = ctx.segs.get(name) if name in ctx.segs else None
    if structure is None or not structure.any():
        return None, {}
    box = mask_box(structure, int(0.15 * ctx.scale) + 3)
    inside = structure[box]
    center = np.argwhere(inside).mean(0)
    coordinates = np.indices(inside.shape, dtype=np.float32)
    source = [center[axis] + (coordinates[axis] - center[axis]) / scale for axis in range(3)]
    scaled = ndi.map_coordinates(inside.astype(np.float32), source, order=1) > 0.5
    scaled_ct = ndi.map_coordinates(ctx.ct[box], source, order=1, mode="nearest")
    body = ctx.mask("body", box)
    target = ctx.ct[box].copy()
    target[scaled & body] = scaled_ct[scaled & body]
    uncovered = inside & ~scaled
    target[uncovered] = fill_out(box, uncovered) if callable(fill_out) else fill_out
    return bounded_delta(ctx.ct, box, ndi.gaussian_filter(target, 0.5) * (scaled | inside)
                         + ctx.ct[box] * ~(scaled | inside)), dict(scale=scale)


def heart_resize(ctx, rng):
    scale = rng.uniform(1.06, 1.25) if rng.random() < 0.6 else rng.uniform(0.82, 0.95)

    def surrounding(box, uncovered):
        ring = dilate(ctx.segs["heart"][box], 3) & ~ctx.segs["heart"][box]
        return float(np.median(ctx.ct[box][ring])) if ring.any() else HU_LUNG

    delta, params = _scale_structure(ctx, rng, "heart", scale, surrounding)
    return _finish(delta, "heart_resize", "T2", "heart", params)


def pericardial_shell(ctx, rng):
    heart = ctx.segs.get("heart")
    if heart is None or not heart.any():
        return None
    thickness = rng.uniform(1.5, max(2.0, 0.03 * ctx.scale))
    box = mask_box(heart, int(thickness) + 3)
    shell = dilate(heart[box], thickness) & ~heart[box] & ctx.mask("body", box)
    keep = smooth_noise(shell.shape, 3.0, rng) > rng.uniform(-1.0, 0.0)
    delta = _replace_in(ctx, box, shell & keep, HU_FLUID, rng)
    return _finish(delta, "pericardial_shell", "T2", "heart", dict(thickness=thickness))


def mediastinal_widening(ctx, rng):
    mediastinum = ctx.compartments["mediastinum"]
    if not mediastinum.any():
        return None
    radius = rng.uniform(2.0, max(3.0, 0.05 * ctx.scale))
    upper = sorted((ctx.lung_z_fraction(0.0), ctx.lung_z_fraction(0.5)))
    box = (slice(max(0, upper[0]), upper[1] + 1),) + tuple(slice(0, size) for size in ctx.shape[1:])
    region = ctx.mask("mediastinum", box)
    if not region.any():
        return None
    grown = np.zeros_like(region)
    shift = int(round(radius))
    for offset in range(1, shift + 1):
        grown[:, offset:] |= region[:, :-offset]
        grown[:, :-offset] |= region[:, offset:]
    grown &= ctx.mask("body", box) & ~region & ~ctx.mask("bone", box)
    delta = _replace_in(ctx, box, grown, HU_SOFT, rng)
    return _finish(delta, "mediastinal_widening", "T2", "mediastinum", dict(radius=radius))


def mediastinal_mass(ctx, rng):
    mediastinum = ctx.compartments["mediastinum"]
    upper = sorted((ctx.lung_z_fraction(0.05), ctx.lung_z_fraction(0.45)))
    candidates = mediastinum.copy()
    z = np.arange(candidates.shape[0]) * ctx.ds
    candidates[(z < upper[0]) | (z > upper[1])] = False
    center = _point_in(ctx, candidates, rng, 1)
    if center is None:
        return None
    center = center * ctx.ds
    radii = [rng.uniform(0.04, 0.1) * ctx.scale for _ in range(3)]
    box = box_around(center, radii, ctx.shape, 2)
    region = ellipsoid(box, center, radii, random_rotation(rng)) & ctx.mask("body", box) & ~ctx.mask("bone", box)
    delta = _replace_in(ctx, box, region, HU_SOFT, rng)
    return _finish(delta, "mediastinal_mass", "T2", "mediastinum", dict(radii=radii))


def mediastinal_shift(ctx, rng):
    region = ctx.compartments["mediastinum"].copy()
    if "heart" in ctx.segs:
        region |= ctx.compartments["heart_ds"]
    if not region.any():
        return None
    full = np.zeros(ctx.shape, bool)
    up = ctx._index(ctx.full_box(), ctx.ds)
    full = region[up]
    if "bronchi" in ctx.segs:
        full |= ctx.segs["bronchi"] & ~dilate(ctx.segs["lungs"], 0)
    shift = rng.uniform(2.0, max(3.0, 0.05 * ctx.scale)) * (1 if rng.random() < 0.5 else -1)
    box = mask_box(full, int(abs(shift)) + 4)
    weight = feather(dilate(full[box], abs(shift)), abs(shift))
    weight /= weight.max() + 1e-6
    coordinates = np.indices(weight.shape, dtype=np.float32)
    coordinates[1] -= shift * weight
    warped = ndi.map_coordinates(ctx.ct[box], coordinates, order=1, mode="nearest")
    delta = bounded_delta(ctx.ct, box, warped)
    return _finish(delta, "mediastinal_shift", "T2", "mediastinum", dict(shift=shift))


def _trachea(ctx):
    airways = ctx.segs.get("bronchi")
    if airways is None or not airways.any():
        return None
    outside = airways & ~ctx.segs["lungs"]
    labels, count = ndi.label(outside)
    if not count:
        return None
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, count + 1))
    return labels == int(np.argmax(sizes)) + 1


def tracheal_deviation(ctx, rng):
    trachea = _trachea(ctx)
    if trachea is None or trachea.sum() < 50:
        return None
    shift = int(round(rng.uniform(2.0, max(3.0, 0.04 * ctx.scale)))) * (1 if rng.random() < 0.5 else -1)
    box = mask_box(trachea, abs(shift) + 3)
    old = trachea[box]
    new = np.roll(old, shift, axis=1)
    target = ctx.ct[box].copy()
    target[old & ~new] = HU_SOFT
    target[new & ~old] = HU_AIR
    delta = bounded_delta(ctx.ct, box, ndi.gaussian_filter(target, 0.6) * dilate(old | new, 1)
                          + ctx.ct[box] * ~dilate(old | new, 1))
    return _finish(delta, "tracheal_deviation", "T2", "airways", dict(shift=shift))


def _airway_region(ctx, rng, radius_fraction=(0.06, 0.15)):
    airways = ctx.segs.get("bronchi")
    if airways is None:
        return None
    peripheral = airways & ctx.segs["lungs"]
    center = _point_in(ctx, peripheral, rng)
    if center is None:
        return None
    radius = rng.uniform(*radius_fraction) * ctx.scale
    box = box_around(center, [radius] * 3, ctx.shape, 2)
    local = airways[box] & ellipsoid(box, center, [radius] * 3)
    return (box, local) if local.sum() > 10 else None


def bronchial_dilation(ctx, rng):
    region = _airway_region(ctx, rng)
    if region is None:
        return None
    box, airways = region
    grown = dilate(airways, rng.uniform(1.0, 2.5)) & ~airways & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, grown, HU_AIR, rng, sigma=0.5, noise=5)
    return _finish(delta, "bronchial_dilation", "T2", "airways", {})


def mucus_plug(ctx, rng):
    region = _airway_region(ctx, rng, (0.04, 0.1))
    if region is None:
        return None
    box, airways = region
    delta = _replace_in(ctx, box, airways, HU_FLUID, rng, sigma=0.5)
    return _finish(delta, "mucus_plug", "T2", "airways", {})


def hilar_mass(ctx, rng):
    center = _point_in(ctx, ctx.compartments["hila"], rng, 1)
    if center is None:
        return None
    center = center * ctx.ds
    radii = [rng.uniform(0.03, 0.08) * ctx.scale for _ in range(3)]
    box = box_around(center, radii, ctx.shape, 2)
    region = ellipsoid(box, center, radii, random_rotation(rng)) & ctx.mask("body", box)
    delta = _replace_in(ctx, box, region, HU_SOFT, rng)
    return _finish(delta, "hilar_mass", "T2", "hila", dict(radii=radii))


def pa_dilation(ctx, rng):
    vessels = ctx.segs.get("lung_vessels")
    if vessels is None:
        return None
    hila = ctx.compartments["hila"]
    center = _point_in(ctx, hila, rng, 1)
    if center is None:
        return None
    center = center * ctx.ds
    radius = rng.uniform(0.06, 0.12) * ctx.scale
    box = box_around(center, [radius] * 3, ctx.shape, 3)
    local = vessels[box] & ellipsoid(box, center, [radius] * 3)
    if local.sum() < 10:
        return None
    grown = dilate(local, rng.uniform(1.0, 2.5)) & ~local & ctx.mask("body", box)
    delta = _replace_in(ctx, box, grown, HU_SOFT, rng, sigma=0.5)
    return _finish(delta, "pa_dilation", "T2", "hila", {})


def _side(ctx, rng):
    return "right" if rng.random() < 0.5 else "left"


def diaphragm_shift(ctx, rng):
    """Unilateral dome elevation (lung base -> soft tissue) or depression (soft -> lung)."""
    side = _side(ctx, rng)
    lung = ctx.segs[f"lung_{side}"]
    height = rng.uniform(0.03, 0.10) * (ctx.z1 - ctx.z0)
    elevate = rng.random() < 0.5
    box = mask_box(lung, int(height) + 3)
    local = lung[box]
    z = np.arange(local.shape[0])[:, None, None]
    columns = local.any(0)
    if ctx.inferior > 0:
        base = local.shape[0] - 1 - np.argmax(local[::-1], 0)
        depth = base[None] - z
    else:
        base = np.argmax(local, 0)
        depth = z - base[None]
    body = ctx.mask("body", box)
    if elevate:
        region = local & (depth >= 0) & (depth < height) & columns[None]
        delta = _replace_in(ctx, box, region, HU_SOFT, rng)
    else:
        region = body & ~local & (depth < 0) & (depth >= -height) & columns[None] & ~ctx.mask("bone", box)
        delta = _replace_in(ctx, box, region, HU_LUNG, rng, noise=25)
    return _finish(delta, "diaphragm_shift", "T2", "sub_diaphragm",
                   dict(side=side, elevate=elevate, height=height))


def free_air(ctx, rng):
    side = _side(ctx, rng)
    sub = ctx.compartments["sub_diaphragm"]
    box = ctx.full_box()
    lung = ctx.segs[f"lung_{side}"]
    columns = lung.any(0)
    region = ctx.mask("sub_diaphragm", box) & columns[None]
    distance = ndi.distance_transform_edt(~lung[::ctx.ds, ::ctx.ds, ::ctx.ds]) * ctx.ds
    thickness = rng.uniform(1.0, 3.0)
    near = (distance <= thickness + 2)[ctx._index(box, ctx.ds)]
    crescent = region & near
    if not sub.any() or crescent.sum() < 20:
        return None
    small = mask_box(crescent, 2)
    delta = _replace_in(ctx, small, crescent[small], HU_AIR, rng, sigma=0.5, noise=5)
    return _finish(delta, "free_air", "T2", "sub_diaphragm", dict(side=side))


def gastric_bubble(ctx, rng):
    sub = ctx.compartments["sub_diaphragm"]
    center = _point_in(ctx, sub, rng, 1)
    if center is None:
        return None
    center = center * ctx.ds
    radii = [rng.uniform(0.03, 0.08) * ctx.scale for _ in range(3)]
    box = box_around(center, radii, ctx.shape, 2)
    region = ellipsoid(box, center, radii) & ctx.mask("sub_diaphragm", box)
    delta = _replace_in(ctx, box, region, HU_AIR, rng, sigma=0.7, noise=5)
    return _finish(delta, "gastric_bubble", "T2", "sub_diaphragm", dict(radii=radii))


def subcutaneous_emphysema(ctx, rng):
    region = sample_region(ctx, "chest_wall", rng, (0.08, 0.2), sampler="noise")
    if region is None:
        return None
    box, mask = region
    streaks = np.abs(smooth_noise(mask.shape, 1.2, rng)) < rng.uniform(0.15, 0.35)
    delta = _replace_in(ctx, box, mask & streaks, -950.0, rng, sigma=0.4, noise=20)
    return _finish(delta, "subcutaneous_emphysema", "T2", "chest_wall", {})


def chest_wall_thickening(ctx, rng):
    compartment = "subcutaneous" if ctx.available("subcutaneous") else "chest_wall"
    region = sample_region(ctx, compartment, rng, (0.05, 0.15))
    if region is None:
        return None
    box, mask = region
    body = ctx.mask("body", box)
    grown = dilate(mask, rng.uniform(2.0, max(3.0, 0.04 * ctx.scale))) & ~body
    delta = _replace_in(ctx, box, grown | (mask & ~body), HU_SOFT, rng)
    return _finish(delta, "chest_wall_thickening", "T2", "chest_wall", {})


def rib_lesion(ctx, rng):
    bone = ctx.compartments["bone"] & ctx.compartments["chest_wall"]
    center = _point_in(ctx, bone, rng, 1)
    if center is None:
        return None
    center = center * ctx.ds
    radius = rng.uniform(0.015, 0.04) * ctx.scale
    box = box_around(center, [radius] * 3, ctx.shape, 2)
    region = ellipsoid(box, center, [radius] * 3) & ctx.mask("bone", box)
    lytic = rng.random() < 0.6
    delta = _replace_in(ctx, box, region, HU_SOFT if lytic else 900.0, rng, sigma=0.5)
    return _finish(delta, "rib_lesion", "T2", "chest_wall", dict(lytic=lytic))


def _lung_ball(ctx, rng, radius_range):
    if not hasattr(ctx, "_interior_lung"):
        ctx._interior_lung = np.argwhere(erode(ctx.segs["lungs"][::2, ::2, ::2], 2)) * 2
    if not len(ctx._interior_lung):
        return None
    center = ctx._interior_lung[rng.integers(len(ctx._interior_lung))]
    radius = rng.uniform(*radius_range) * ctx.scale
    box = box_around(center, [radius] * 3, ctx.shape, 2)
    return box, center, radius


def nodule(ctx, rng):
    found = _lung_ball(ctx, rng, (0.015, 0.06))
    if found is None:
        return None
    box, center, radius = found
    radii = [radius * rng.uniform(0.7, 1.3) for _ in range(3)]
    noise = smooth_noise(tuple(item.stop - item.start for item in box), 1.5, rng)
    region = ellipsoid(box, center, radii) & (noise > -1.2) & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, region, HU_SOFT, rng)
    return _finish(delta, "nodule", "T2", "lungs", dict(radius=radius))


def cavity(ctx, rng):
    found = _lung_ball(ctx, rng, (0.03, 0.07))
    if found is None:
        return None
    box, center, radius = found
    outer = ellipsoid(box, center, [radius] * 3) & ctx.mask("lungs", box)
    inner = ellipsoid(box, center, [radius * rng.uniform(0.4, 0.75)] * 3)
    target = ctx.ct[box].copy()
    target[outer] = HU_SOFT
    target[inner & outer] = HU_AIR
    delta = bounded_delta(ctx.ct, box, ndi.gaussian_filter(target, 0.6) * dilate(outer, 1)
                          + ctx.ct[box] * ~dilate(outer, 1))
    return _finish(delta, "cavity", "T2", "lungs", dict(radius=radius))


def bulla(ctx, rng):
    found = _lung_ball(ctx, rng, (0.03, 0.09))
    if found is None:
        return None
    box, center, radius = found
    outer = ellipsoid(box, center, [radius] * 3) & ctx.mask("lungs", box)
    wall = outer & ~erode(outer, 1.0)
    target = ctx.ct[box].copy()
    target[outer] = HU_AIR
    target[wall] = 0.0
    delta = bounded_delta(ctx.ct, box, target)
    return _finish(delta, "bulla", "T2", "lungs", dict(radius=radius))


def calcified_dot(ctx, rng):
    found = _lung_ball(ctx, rng, (0.006, 0.015))
    if found is None:
        return None
    box, center, radius = found
    region = ellipsoid(box, center, [max(radius, 1.5)] * 3) & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, region, rng.uniform(300, 900), rng, sigma=0.4)
    return _finish(delta, "calcified_dot", "T2", "lungs", {})


def lobar_hyperlucency(ctx, rng):
    lobes = [name for name in LOBES if name in ctx.segs and ctx.segs[name].sum() > 100]
    if not lobes:
        return None
    lobe = str(rng.choice(lobes))
    box = mask_box(ctx.segs[lobe], 2)
    region = ctx.segs[lobe][box]
    amount = rng.uniform(40, 140)
    target = ctx.ct[box] - amount * feather(region, 1.5)
    delta = bounded_delta(ctx.ct, box, target)
    return _finish(delta, "lobar_hyperlucency", "T2", "lungs", dict(lobe=lobe, amount=amount))


# ---------------------------------------------------------------------------- T2-near families
def lobar_collapse(ctx, rng):
    lobes = [name for name in LOBES if name in ctx.segs and ctx.segs[name].sum() > 100]
    hila = ctx.compartments["hila"]
    if not lobes or not hila.any():
        return None
    lobe = str(rng.choice(lobes))
    box = mask_box(ctx.segs[lobe], 2)
    region = ctx.segs[lobe][box]
    hilum = np.argwhere(hila).mean(0) * ctx.ds - np.array([item.start for item in box])
    z, x, y = np.indices(region.shape, dtype=np.float32)
    distance = np.sqrt((z - hilum[0]) ** 2 + (x - hilum[1]) ** 2 + (y - hilum[2]) ** 2)
    limit = np.quantile(distance[region], rng.uniform(0.25, 0.5))
    collapsed = region & (distance <= limit)
    delta = bounded_delta(ctx.ct, box, ctx.ct[box] + rng.uniform(400, 800) * feather(collapsed, 1.0))
    return _finish(delta, "lobar_collapse", "T2near", "lungs", dict(lobe=lobe))


def fissural_fluid(ctx, rng):
    fissures = ctx.compartments["fissures"]
    center = _point_in(ctx, fissures, rng, 1)
    if center is None:
        return None
    center = center * ctx.ds
    radius = rng.uniform(0.06, 0.15) * ctx.scale
    box = box_around(center, [radius] * 3, ctx.shape, 2)
    sheet = ctx.mask("fissures", box) & ellipsoid(box, center, [radius, radius, radius * 0.6])
    sheet = dilate(sheet, rng.uniform(0.5, 1.5)) & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, sheet, HU_FLUID, rng, sigma=0.5)
    return _finish(delta, "fissural_fluid", "T2near", "fissures", {})


def pleural_thickening(ctx, rng):
    region = sample_region(ctx, "pleural_rim", rng, (0.05, 0.15), sampler="noise")
    if region is None:
        return None
    box, mask = region
    delta = _replace_in(ctx, box, mask, HU_SOFT, rng)
    return _finish(delta, "pleural_thickening", "T2near", "pleural_rim", {})


def pneumomediastinum(ctx, rng):
    mediastinum = ctx.compartments["mediastinum"]
    if not mediastinum.any():
        return None
    box = ctx.full_box()
    region = ctx.mask("mediastinum", box)
    border = region & dilate(ctx.segs["lungs"][::ctx.ds, ::ctx.ds, ::ctx.ds], 2)[ctx._index(box, ctx.ds)]
    small = mask_box(border, 2)
    if small is None:
        return None
    streaks = np.abs(smooth_noise(border[small].shape, 1.0, rng)) < rng.uniform(0.2, 0.4)
    delta = _replace_in(ctx, small, border[small] & streaks, -950.0, rng, sigma=0.4)
    return _finish(delta, "pneumomediastinum", "T2near", "mediastinum", {})


def bronchial_wall_thickening(ctx, rng):
    region = _airway_region(ctx, rng, (0.1, 0.25))
    if region is None:
        return None
    box, airways = region
    wall = dilate(airways, rng.uniform(1.0, 2.0)) & ~airways & ctx.mask("lungs", box)
    delta = _replace_in(ctx, box, wall, HU_SOFT, rng, sigma=0.5)
    return _finish(delta, "bronchial_wall_thickening", "T2near", "airways", {})


def vessel_change(ctx, rng):
    vessels = ctx.segs.get("lung_vessels")
    if vessels is None:
        return None
    center = _point_in(ctx, vessels & ctx.segs["lungs"], rng)
    if center is None:
        return None
    radius = rng.uniform(0.08, 0.2) * ctx.scale
    box = box_around(center, [radius] * 3, ctx.shape, 3)
    local = vessels[box] & ellipsoid(box, center, [radius] * 3)
    if local.sum() < 10:
        return None
    if rng.random() < 0.5:
        region, value = dilate(local, rng.uniform(0.8, 2.0)) & ~local & ctx.mask("lungs", box), HU_SOFT
    else:
        region, value = local, HU_LUNG
    delta = _replace_in(ctx, box, region, value, rng, sigma=0.5)
    return _finish(delta, "vessel_change", "T2near", "lungs", {})


# ---------------------------------------------------------------------------- nuisance
def inspiration_nuisance(ctx, rng):
    """Bilateral breathing-depth change: both domes move and lung density shifts together."""
    height = rng.uniform(0.02, 0.06) * (ctx.z1 - ctx.z0)
    deeper = rng.random() < 0.5
    box = mask_box(ctx.segs["lungs"], int(height) + 3)
    lungs = ctx.segs["lungs"][box]
    target = ctx.ct[box].copy()
    target[lungs] -= (1 if deeper else -1) * rng.uniform(15, 50)
    for side in ("right", "left"):
        lung = ctx.segs[f"lung_{side}"][box]
        z = np.arange(lung.shape[0])[:, None, None]
        if ctx.inferior > 0:
            depth = (lung.shape[0] - 1 - np.argmax(lung[::-1], 0))[None] - z
        else:
            depth = z - np.argmax(lung, 0)[None]
        columns = lung.any(0)[None]
        if deeper:
            region = ~lung & (depth < 0) & (depth >= -height) & columns & ctx.mask("body", box)
            target[region] = HU_LUNG
        else:
            region = lung & (depth >= 0) & (depth < height) & columns
            target[region] = HU_SOFT
    return bounded_delta(ctx.ct, box, target), dict(height=height, deeper=deeper)


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


def family_names(tiers=("T1", "T2")):
    return sorted(name for name, spec in FAMILIES.items() if spec["tier"] in tiers)


def sample_change(ctx, family, rng, donor=None, attempts=6) -> Optional[ChangeSpec]:
    spec = FAMILIES[family]
    if any(not ctx.available(name) for name in spec["requires"]):
        return None
    for _ in range(attempts):
        change = spec["fn"](ctx, rng, donor) if family.startswith("t1_") else spec["fn"](ctx, rng)
        if change is not None:
            change.family = family
            return change
    return None


def compose_scene(ctx, rng, tiers=("T1", "T2"), n_changes=(1, 3), margin=3,
                  donor_fn: Optional[Callable] = None, family_weights=None, attempts=40):
    """Sample 0..n voxel-disjoint changes of distinct families; each assigned to prior or current."""
    names = [name for name in family_names(tiers)
             if all(ctx.available(item) for item in FAMILIES[name]["requires"])]
    if not names:
        return []
    weights = np.array([1.0 if family_weights is None else family_weights.get(name, 1.0)
                        for name in names], np.float64)
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
        donor = donor_fn(rng) if donor_fn is not None and "cutpaste" in family else None
        change = sample_change(ctx, family, rng, donor)
        if change is None:
            continue
        footprint = np.zeros(ctx.shape, bool)
        footprint[change.delta.box] = change.delta.support()
        if (occupied & footprint).any() or (dilate_box(occupied, change.delta.box, margin) & footprint[change.delta.box]).any():
            continue
        occupied |= footprint
        used.add(family)
        changes.append((change, "current" if rng.random() < 0.5 else "prior"))
    return changes


def dilate_box(mask, box, margin):
    grown = tuple(slice(max(0, item.start - margin), min(size, item.stop + margin))
                  for item, size in zip(box, mask.shape))
    local = dilate(mask[grown], margin) if mask[grown].any() else mask[grown]
    offset = tuple(slice(item.start - outer.start, item.start - outer.start + (item.stop - item.start))
                   for item, outer in zip(box, grown))
    return local[offset]
