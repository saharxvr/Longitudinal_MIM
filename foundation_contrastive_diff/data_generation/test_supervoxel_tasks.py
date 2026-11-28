"""Supervoxel change generator on a small synthetic chest phantom (CPU, seconds)."""

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from data_generation import audit_fewshot_inputs as audit
from data_generation import generate_supervoxel_corpus as corpus
from data_generation import supervoxel_render as render
from data_generation import supervoxel_tasks as tasks


def phantom(shape=(48, 64, 56)):
    """Air, body, two base-wide lungs (5 lobes), heart, spine, fat shell, airways, vessels."""
    z, x, y = np.indices(shape, dtype=np.float32)
    depth, width, ap = shape
    ct = np.full(shape, -1000.0, np.float32)
    body = ((x - width / 2) / (0.45 * width)) ** 2 + ((y - ap / 2) / (0.42 * ap)) ** 2 <= 1
    body &= (z > 2) & (z < depth - 2)
    ct[body] = 40.0
    surface = body & ~(((x - width / 2) / (0.40 * width)) ** 2 + ((y - ap / 2) / (0.37 * ap)) ** 2 <= 1)
    ct[surface] = -100.0
    segs = {}
    lungs = np.zeros(shape, bool)
    for side, center in (("right", 0.30 * width), ("left", 0.70 * width)):
        widen = 0.5 + 0.5 * np.clip((z - 6) / (depth - 14), 0, 1)  # bases (high z) are wider
        lung = (((x - center) / (0.14 * width * widen)) ** 2 + ((y - ap / 2) / (0.28 * ap)) ** 2 <= 1)
        lung &= (z > 6) & (z < depth - 8)
        lungs |= lung
        segs[f"lung_{side}"] = lung
    ct[lungs] = -850.0
    zl = np.nonzero(lungs.any((1, 2)))[0]
    third = (zl[-1] - zl[0]) / 3
    right, left = segs.pop("lung_right"), segs.pop("lung_left")
    segs["upper_right_lobe"] = right & (z < zl[0] + third)
    segs["middle_right_lobe"] = right & (z >= zl[0] + third) & (z < zl[0] + 2 * third)
    segs["lower_right_lobe"] = right & (z >= zl[0] + 2 * third)
    segs["upper_left_lobe"] = left & (z < zl[0] + 1.5 * third)
    segs["lower_left_lobe"] = left & (z >= zl[0] + 1.5 * third)
    heart = (((x - width / 2) / (0.12 * width)) ** 2 + ((y - 0.45 * ap) / (0.18 * ap)) ** 2
             + ((z - 0.62 * depth) / (0.18 * depth)) ** 2 <= 1) & ~lungs
    ct[heart] = 45.0
    segs["heart"] = heart
    spine = (((x - width / 2) / 4) ** 2 + ((y - 0.8 * ap) / 4) ** 2 <= 1) & body
    ct[spine] = 600.0
    trachea = (((x - width / 2) / 2.2) ** 2 + ((y - 0.45 * ap) / 2.2) ** 2 <= 1) & (z > 3) & (z < 18)
    branches = np.zeros(shape, bool)
    for center in (0.30 * width, 0.70 * width):
        branches |= ((((x - center) / 1.6) ** 2 + ((y - 0.45 * ap) / 1.6) ** 2) <= 1) & (z >= 14) & (z < 30)
    branches |= (np.abs(z - 16) <= 1) & (np.abs(y - 0.45 * ap) <= 1.5) & (np.abs(x - width / 2) < 0.22 * width)
    airways = (trachea | branches) & body
    ct[airways] = -1000.0
    segs["bronchi"] = airways
    vessels = np.zeros(shape, bool)
    for center in (0.27 * width, 0.33 * width, 0.67 * width, 0.73 * width):
        vessels |= ((((x - center) / 1.2) ** 2 + ((y - 0.5 * ap) / 1.2) ** 2) <= 1) & (z > 12) & (z < 36)
    vessels &= lungs
    ct[vessels] = 40.0
    segs["lung_vessels"] = vessels
    return ct, segs


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ct, segs = phantom()
        cls.ctx = tasks.CTContext(ct, segs, downsample=2, supervoxel_downsample=2,
                                  n_supervoxels=300, seed=0)

    def test_context_geometry_and_compartments(self):
        ctx = self.ctx
        self.assertEqual(ctx.inferior, 1)
        for name in ("body", "mediastinum", "sub_diaphragm", "chest_wall", "hila", "bone", "fissures"):
            with self.subTest(name=name):
                self.assertGreater(int(ctx.compartments[name].sum()), 0)
        lungs = ctx.segs["lungs"][::2, ::2, ::2]
        for name in ("mediastinum", "sub_diaphragm", "chest_wall"):
            self.assertFalse((ctx.compartments[name] & lungs).any())
        self.assertGreater(len(np.unique(ctx.supervoxels)), 50)
        self.assertGreater(len(ctx.adjacency), 50)

    def test_families_produce_bounded_nonzero_deltas(self):
        produced = {}
        for family in sorted(tasks.FAMILIES):
            if "cutpaste" in family:
                continue
            for seed in range(6):
                change = tasks.sample_change(self.ctx, family, np.random.default_rng(seed))
                if change is not None:
                    produced[family] = change
                    break
        for family in ("heart_resize", "nodule", "cavity", "bulla", "diaphragm_shift",
                       "mediastinal_widening", "mediastinal_mass", "tracheal_deviation",
                       "lobar_hyperlucency", "lobar_collapse", "t1_density@lungs",
                       "t1_replace@mediastinum", "t1_grow@lungs", "t1_displace@heart"):
            self.assertIn(family, produced)
        self.assertGreater(len(produced), 0.75 * (len(tasks.FAMILIES) - len(tasks.T1_COMPARTMENTS)))
        for family, change in produced.items():
            with self.subTest(family=family):
                modified = self.ctx.ct[change.delta.box] + change.delta.values
                self.assertGreaterEqual(float(modified.min()), tasks.HU_MIN - 1e-3)
                self.assertLessEqual(float(modified.max()), tasks.HU_MAX + 1e-3)
                self.assertIn(change.effect, ("denser", "lighter", "mixed"))
                self.assertEqual(change.tier, tasks.FAMILIES[family]["tier"])
                self.assertNotIn(change.family, tasks.VALIDATION_FAMILIES)

    def test_cutpaste_uses_donor_patch(self):
        donor = np.full((40, 40, 40), 300.0, np.float32)
        change = tasks.sample_change(self.ctx, "t1_cutpaste@lungs", np.random.default_rng(1), donor)
        self.assertIsNotNone(change)
        self.assertEqual(change.effect, "denser")

    def test_projection_is_linear_and_signs_follow_timepoint(self):
        rng = np.random.default_rng(3)
        center = np.argwhere(self.ctx.segs["lung_right"]).mean(0).round().astype(int)
        box = tasks.box_around(center, [5, 5, 5], self.ctx.shape)
        ball = tasks.ellipsoid(box, center, [5, 5, 5]) & self.ctx.segs["lungs"][box]
        delta = tasks.bounded_delta(self.ctx.ct, box, np.where(ball, 45.0, self.ctx.ct[box]))
        change = tasks.ChangeSpec("nodule", "T2", "lungs", "denser", delta)
        frame = render.FrameRenderer(self.ctx.segs["lungs"], render.sample_angles(rng), "cpu")
        delta = change.delta.full(self.ctx.shape)
        combined = frame.base(self.ctx.ct + delta)
        separate = frame.base(self.ctx.ct) + frame.delta(delta)
        self.assertLess(float((combined - separate).abs().max()), 1e-2 * float(frame.delta(delta).abs().max()) + 1e-3)
        for timepoint, sign in (("appear", 1), ("resolve", -1)):
            scene = render.render_scene(self.ctx, [(change, timepoint)], np.random.default_rng(4))
            values = scene["maps"][0]
            self.assertGreater(int((np.sign(values) == sign).sum()), 50)
            self.assertEqual(int((np.sign(values) == -sign).sum()), 0)
            self.assertEqual(scene["prior"].shape, (512, 512))
        moved = scene["registered_prior"] - scene["current"]
        self.assertGreater(float(np.abs(moved).max()), 0)

    def test_chunked_box_renderer_matches_full_grid_reference(self):
        angles = (9.0, -12.0, 5.0)
        lungs = self.ctx.segs["lungs"]
        grid = render.sampling_grid(lungs.shape, angles, "cpu")
        rotated = render.rotate(lungs.astype(np.float32), grid)
        crop = render.lung_crop(rotated)
        change = tasks.sample_change(self.ctx, "nodule", np.random.default_rng(2))
        reference_base = render.project(render.rotate(np.maximum(self.ctx.ct, -1000.0) + 1000.0, grid, "reflection"),
                                        crop)
        reference_delta = render.project(render.rotate(change.delta.full(self.ctx.shape), grid, "zeros"), crop)
        with unittest.mock.patch.object(render, "CHUNK_VOXELS", 5000):
            frame = render.FrameRenderer(lungs, angles)
            self.assertEqual(frame.crop, crop)
            np.testing.assert_allclose(frame.footprint.numpy(), render._resize(rotated[crop].amax(-1)).numpy(),
                                       atol=1e-5)
            np.testing.assert_allclose(frame.base(self.ctx.ct).numpy(), reference_base.numpy(), rtol=1e-4)
            sparse = frame.delta(change.delta).numpy()
        scale = float(np.abs(reference_delta.numpy()).max())
        self.assertGreater(scale, 0)
        np.testing.assert_allclose(sparse, reference_delta.numpy(), atol=1e-4 * scale)

    def test_device_bank_overlays_and_oom_retry(self):
        import torch

        base = torch.zeros(4, 5, 6)
        devices = base.clone()
        devices[1, 2, 3] = 900.0
        bank = corpus.DeviceBank.__new__(corpus.DeviceBank)
        bank.probability, bank.pairs = 1.0, [[corpus.DeviceBank._sparse(devices, base), None]]
        self.assertEqual(bank.pairs[0][0][0].tolist(), [1 * 30 + 2 * 6 + 3])
        sides = set()
        for seed in range(20):
            prior, current, names = bank(base, np.random.default_rng(seed))
            self.assertEqual(names, ["ExternalDevices"])
            self.assertEqual(float(base.abs().sum()), 0.0)  # the shared base volume is never modified
            sides.add((float(prior[1, 2, 3]), float(current[1, 2, 3])))
        self.assertEqual(sides, {(900.0, 0.0), (0.0, 900.0)})
        bank.probability = 0.0
        self.assertIs(bank(base, np.random.default_rng(0))[0], base)
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return "done"

        with unittest.mock.patch.object(corpus.time, "sleep"):
            self.assertEqual(corpus.with_oom_retry(flaky, "test", "cpu"), "done")
        self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            corpus.with_oom_retry(lambda: (_ for _ in ()).throw(ValueError("other")), "test", "cpu")

    def test_progress_regress_and_stable_modes(self):
        change = tasks.sample_change(self.ctx, "nodule", np.random.default_rng(2))
        self.assertIsNotNone(change)
        partial, method = tasks.make_partial(self.ctx, change, np.random.default_rng(0), 0.5)
        self.assertEqual(method, "param")
        # Replaying the same draws with half the radius gives a smaller lesion at the same place.
        self.assertLess(int(partial.support().sum()), int(change.delta.support().sum()))
        self.assertEqual(int((partial.support() & ~tasks.dilate(change.delta.support(), 1.0)).sum()), 0)
        change.partial = partial
        angles = (render.sample_angles(np.random.default_rng(8)), render.sample_angles(np.random.default_rng(9)))
        maps = {mode: render.render_scene(self.ctx, [(change, mode)], np.random.default_rng(4), angles=angles)
                for mode in ("appear", "progress", "regress", "stable")}
        progress = maps["progress"]["maps"][0]
        np.testing.assert_allclose(maps["regress"]["maps"][0], -progress, atol=1e-6)
        self.assertGreater(int((progress > 0).sum()), 50)
        self.assertLess(float(progress.sum()), float(maps["appear"]["maps"][0].sum()))
        stable = maps["stable"]
        self.assertFalse(np.any(stable["maps"][0]))
        np.testing.assert_allclose(stable["current"], stable["registered_prior"])
        self.assertGreater(float(np.abs(stable["current"] - maps["appear"]["registered_prior"]).max()), 0)
        from data_generation import preview_supervoxel_families as preview
        fast = preview.ModeFrames(self.ctx, angles).modes(change)
        for mode in ("progress", "stable"):
            np.testing.assert_allclose(fast[mode]["gt"], maps[mode]["maps"][0], atol=1e-4)
            np.testing.assert_allclose(fast[mode]["current"], maps[mode]["current"], rtol=1e-4)
            np.testing.assert_allclose(fast[mode]["prior"], maps[mode]["prior"], rtol=1e-3)
        for name in ("t1_density@lungs", "lobar_hyperlucency"):
            other = tasks.sample_change(self.ctx, name, np.random.default_rng(1))
            smaller, how = tasks.make_partial(self.ctx, other, np.random.default_rng(1), 0.5)
            self.assertIn(how, ("spatial", "intensity"))
            self.assertLess(float(np.abs(smaller.values).sum()), float(np.abs(other.delta.values).sum()))

    def test_scene_changes_are_disjoint_distinct_families(self):
        changes = tasks.compose_scene(self.ctx, np.random.default_rng(7), n_changes=(3, 3))
        self.assertGreaterEqual(len(changes), 2)
        self.assertEqual(len({change.family for change, _ in changes}), len(changes))
        occupied = np.zeros(self.ctx.shape, int)
        for change, timepoint in changes:
            self.assertIn(timepoint, tasks.TIMEPOINT_MODES)
            self.assertEqual(change.partial is not None, timepoint in ("progress", "regress"))
            occupied[change.delta.box] += change.delta.support()
        self.assertLessEqual(int(occupied.max()), 1)
        tiers = {change.tier for change, _ in changes}
        self.assertTrue(tiers <= {"T1", "T2"})
        rng, seen = np.random.default_rng(0), set()
        for _ in range(60):
            change = tasks.sample_change(self.ctx, "heart_resize", rng)
            seen.add(tasks.assign_timepoint(self.ctx, change, rng))
        self.assertEqual(seen, set(tasks.TIMEPOINT_MODES))

    def test_inspiration_nuisance_never_enters_gt(self):
        rng = np.random.default_rng(5)
        nuisance, params = tasks.inspiration_nuisance(self.ctx, rng)
        scene = render.render_scene(self.ctx, [], rng, prior_nuisance=nuisance)
        self.assertEqual(scene["maps"], [])
        np.testing.assert_allclose(scene["current"], scene["registered_prior"])

    def test_ellipse_labels_round_trip_and_small_components_removed(self):
        from skimage.draw import ellipse

        signed = np.zeros((128, 128), np.float32)
        rows, columns = ellipse(60, 50, 30, 9, shape=signed.shape, rotation=np.deg2rad(35))
        signed[rows, columns] = 0.2
        signed[100:104, 100:104] = -0.3
        cleaned = render.clean_map(signed)
        self.assertEqual(int((cleaned < 0).sum()), 0)
        fitted = render.fit_ellipses(cleaned, inflate=1.0, pad=0.0)
        self.assertEqual([item["label"] for item in fitted], ["Appearance"])
        positive, negative = render.rasterize_ellipses(fitted, signed.shape)
        reference = cleaned > 0
        self.assertGreater((positive & reference).sum() / (positive | reference).sum(), 0.95)
        self.assertFalse(negative.any())
        coarse = render.rasterize_ellipses(render.fit_ellipses(cleaned), signed.shape)[0]
        self.assertTrue((coarse | ~reference).all())

    def test_angle_sampler_respects_limits(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            angles = render.sample_angles(rng)
            self.assertTrue(all(abs(angle) <= 17.5 for angle in angles))
            self.assertLessEqual(sum(abs(angle) for angle in angles), 37.5)

    def test_sizes_are_physical_on_anisotropic_grids(self):
        spacing = (2.0, 0.5, 0.5)
        shape = (40, 120, 120)
        box = tuple(slice(0, size) for size in shape)
        ball = tasks.ellipsoid(box, (20, 60, 60), 10.0, spacing)
        extents_mm = [(np.ptp(np.nonzero(ball.any(axis=tuple(a for a in range(3) if a != axis)))[0]) + 1) * s
                      for axis, s in enumerate(spacing)]
        self.assertTrue(np.allclose(extents_mm, 20.0, atol=2.0), extents_mm)
        point = np.zeros(shape, bool)
        point[20, 60, 60] = True
        grown = tasks.dilate(point, 6.0, spacing)
        self.assertEqual(int(grown[:, 60, 60].sum()), 7)
        self.assertEqual(int(grown[20, :, 60].sum()), 25)
        ct, segs = phantom()
        stretched = tasks.CTContext(np.repeat(ct, 2, axis=1), {k: np.repeat(v, 2, axis=1) for k, v in segs.items()},
                                    spacing=(1.0, 0.5, 1.0), n_supervoxels=300)
        self.assertEqual(stretched.ds.tolist(), [3, 6, 3])
        self.assertAlmostEqual(stretched.scale, self.ctx.scale, delta=0.1 * self.ctx.scale)

    def test_orientation_from_heart_and_diaphragm_dome_columns(self):
        ct, segs = phantom()
        flipped = tasks.CTContext(ct[::-1].copy(), {k: v[::-1].copy() for k, v in segs.items()},
                                  downsample=2, supervoxel_downsample=2, n_supervoxels=300)
        self.assertEqual(flipped.inferior_rule, "heart centroid")
        self.assertEqual(flipped.inferior, -1)
        for ctx in (self.ctx, flipped):
            below = np.argwhere(ctx.compartments["sub_diaphragm"])[:, 0].mean() * ctx.ds[0]
            self.assertGreater(ctx.inferior * (below - ctx.lung_center[0]), 0)
        lung = np.zeros((50, 10, 4), bool)
        lung[10:40, :5] = True
        lung[10:20, 5:] = True  # columns ending high (e.g. over the heart) are not diaphragm
        np.testing.assert_array_equal(self.ctx.dome_columns(lung)[:, 0], [True] * 5 + [False] * 5)

    def test_multifocal_t1_shares_direction_across_separate_regions(self):
        parts = [tasks.VolumeDelta((slice(0, 2), slice(0, 2), slice(0, 2)), np.ones((2, 2, 2), np.float32)),
                 tasks.VolumeDelta((slice(5, 7), slice(3, 4), slice(0, 1)), np.full((2, 1, 1), -2, np.float32))]
        merged = tasks.merge_deltas(parts)
        self.assertEqual(merged.values.shape, (7, 4, 2))
        np.testing.assert_array_equal(merged.full((8, 8, 8)), parts[0].full((8, 8, 8)) + parts[1].full((8, 8, 8)))
        found = []
        for seed in range(12):
            change = tasks.t1_change(self.ctx, np.random.default_rng(seed), "lungs", "density", foci=3)
            if change is not None and change.params["foci"] >= 2:
                found.append(change)
        self.assertTrue(found)
        for change in found:
            self.assertIn(change.effect, ("denser", "lighter"))
            self.assertGreaterEqual(ndi.label(change.delta.support())[1], 2)

    def test_widening_stays_in_lungs_and_faint_changes_are_rejected(self):
        change = tasks.sample_change(self.ctx, "mediastinal_widening", np.random.default_rng(0))
        self.assertIsNotNone(change)
        lungs = tasks.dilate(self.ctx.segs["lungs"][change.delta.box], 2.0)
        self.assertFalse((change.delta.support(50.0) & ~lungs).any())
        box = (slice(20, 24), slice(15, 19), slice(26, 30))
        faint = tasks.VolumeDelta(box, np.full((4, 4, 4), 5.0, np.float32))
        self.assertIsNone(tasks._finish(self.ctx, faint, "x", "T1", "lungs", {}))
        self.ctx.min_visibility = None
        try:
            self.assertIsNotNone(tasks._finish(self.ctx, faint, "x", "T1", "lungs", {}))
        finally:
            self.ctx.min_visibility = (0.035, 80.0)

    def test_corpus_scene_files_manifest_and_train_only_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = corpus.parse_args(["--source_manifest", "x", "--output", folder,
                                      "--empty_scene_prob", "0", "--inspiration_prob", "1",
                                      "--changes_per_scene", "2", "2"])
            donors = corpus.DonorBank()
            donors.add(self.ctx)
            record = corpus.generate_scene(self.ctx, "caseA", "patientA", 3, args, np.eye(4), None,
                                           donors, root / "caseA" / "scene0003")
            again = corpus.generate_scene(self.ctx, "caseA", "patientA", 3, args, np.eye(4), None,
                                          donors, root / "caseA" / "scene0003")
            self.assertEqual(record["changes"], again["changes"])
            maps = np.load(root / "caseA" / "scene0003" / "changes.npz")["maps"]
            prior = np.squeeze(nib.load(root / "caseA" / "scene0003" / "prior.nii.gz").get_fdata())
            self.assertEqual(maps.shape, (record["num_changes"], 512, 512))
            self.assertEqual(prior.shape, (512, 512))
            ellipses = json.loads((root / "caseA" / "scene0003" / "ellipses.json").read_text())
            self.assertEqual(len(ellipses), record["num_changes"])
            self.assertIsNotNone(record["nuisance"]["inspiration"])
            summary = corpus.build_manifest(root)
            self.assertEqual(summary["scenes"], 1)
            line = json.loads((root / "manifest_scenes.jsonl").read_text().splitlines()[0])
            self.assertEqual(line["scene_dir"], "caseA/scene0003")
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps(item) for item in (
                dict(case="a", ct_source_path="a.nii.gz", split="train", patient_key="p1"),
                dict(case="b", ct_source_path="b.nii.gz", split="val", patient_key="p2"))) + "\n")
            self.assertEqual(corpus.train_cases(source), [("a", "a.nii.gz", "p1")])
            source.write_text(source.read_text() + json.dumps(
                dict(case="c", ct_source_path="c.nii.gz", split="train", patient_key="p2")) + "\n")
            with self.assertRaises(ValueError):
                corpus.train_cases(source)

    def test_disk_loading_crop_and_audit_on_saved_phantom(self):
        ct, segs = phantom()

        def save(volume, path):
            data = np.transpose(np.flip(volume, 0), (1, 2, 0)).astype(np.float32)
            nib.save(nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4)), str(path))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save(ct, root / "caseX.nii.gz")
            templates = {}
            for name, mask in segs.items():
                save(mask.astype(np.float32), root / f"caseX_{name}.nii.gz")
                templates[name] = str(root / f"{{case}}_{name}.nii.gz")
            templates["lungs"] = str(root / "{case}_missing_lungs.nii.gz")
            loaded, loaded_segs, _, missing = corpus.load_case("caseX", str(root / "caseX.nii.gz"), templates)
            self.assertEqual(missing, ["lungs"])
            np.testing.assert_array_equal(loaded_segs["heart"].sum(), segs["heart"].sum())
            self.assertLessEqual(loaded.shape[0], ct.shape[0])
            coverage = audit.segmentation_coverage([("caseX", "", "p")], templates)
            self.assertEqual(coverage["with_all_lobes"], 1)
            original = tasks.CTContext
            with unittest.mock.patch.object(tasks, "CTContext", side_effect=lambda c, s, **k: original(
                    c, s, downsample=2, supervoxel_downsample=2, n_supervoxels=300)):
                report = audit.sample_ct_checks([("caseX", str(root / "caseX.nii.gz"), "p")], templates, 1, "cpu")
            self.assertNotIn("error", report[0])
            self.assertLess(report[0]["linearity_relative_error"], 0.02)
            self.assertGreater(report[0]["family_success"], 0)

    def test_export_bundle_and_local_preview_gallery(self):
        from data_generation import export_preview_cases as export
        from data_generation import preview_supervoxel_families as preview

        ct, segs = phantom()

        def save(volume, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            data = np.transpose(np.flip(volume, 0), (1, 2, 0)).astype(np.float32)
            nib.save(nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4)), str(path))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campus = root / "campus"
            save(ct, campus / "scans" / "caseA.nii.gz")
            names = {name: f"{{case}}_{name}.nii.gz" for name in corpus.SEG_TEMPLATES if name != "lungs"}
            for name in names:
                save(segs[name].astype(np.float32), campus / "segs" / names[name].format(case="caseA"))
            pair = campus / "pairs" / "caseA" / "pair0"
            save(ct[:, :, :1], pair / "prior.nii.gz")
            (pair / "params.json").write_text(json.dumps(dict(effective_anomaly_type="consolidation",
                                                              direction="appearance")))
            for name in ("current", "diff_map"):
                save(ct[:, :, :1] * 0, pair / f"{name}.nii.gz")
            manifest = campus / "manifest_split.jsonl"
            manifest.write_text(json.dumps(dict(
                case="caseA", ct_source_path=str(campus / "scans" / "caseA.nii.gz"), split="train",
                patient_key="caseA", pair_id="caseA/pair0", realized_change=True,
                effective_anomaly_type="consolidation", prior_path=str(pair / "prior.nii.gz"),
                current_path=str(pair / "current.nii.gz"), diff_map_path=str(pair / "diff_map.nii.gz"),
                params_path=str(pair / "params.json"))) + "\n")
            templates = {name: str(campus / "segs" / pattern) for name, pattern in names.items()}
            with unittest.mock.patch.dict(corpus.SEG_TEMPLATES, templates, clear=True):
                export.main(["--source_manifest", str(manifest), "--output", str(root / "bundle"), "--count", "1"])
            cases = json.loads((root / "bundle" / "cases.json").read_text())
            self.assertEqual(cases[0]["case"], "caseA")
            self.assertTrue((root / "bundle" / "caseA" / "heart.nii.gz").is_file())
            self.assertEqual(len(cases[0]["reference_pairs"]), 1)
            original = tasks.CTContext
            with unittest.mock.patch.object(tasks, "CTContext", side_effect=lambda c, s, **k: original(
                    c, s, downsample=2, supervoxel_downsample=2, n_supervoxels=300, supervoxels=k.get("supervoxels"))):
                preview.main(["--cases_root", str(root / "bundle"), "--output", str(root / "gallery"),
                              "--families", "heart_resize", "t1_density@lungs", "lobar_collapse",
                              "--scenes", "1", "--threads", "2"])
            stats = json.loads((root / "gallery" / "stats.json").read_text())
            produced = {row["family"] for row in stats["families"] if row["produced"]}
            self.assertIn("heart_resize", produced)
            for row in stats["families"]:
                if row["produced"]:
                    self.assertTrue((root / "gallery" / row["png"]).is_file())
            self.assertTrue((root / "gallery" / "scenes" / "caseA_scene0.png").is_file())
            self.assertTrue((root / "gallery" / "nuisance" / "caseA_inspiration.png").is_file())
            self.assertEqual(len(stats["references"]), 1)
            self.assertIn("heart_resize", (root / "gallery" / "index.md").read_text())
            with unittest.mock.patch.object(tasks, "CTContext", side_effect=lambda c, s, **k: original(
                    c, s, downsample=2, supervoxel_downsample=2, n_supervoxels=300, supervoxels=k.get("supervoxels"))):
                corpus.main(["--cases_root", str(root / "bundle"), "--output", str(root / "corpus"),
                             "--scenes_per_ct", "2", "--devices_prob", "0", "--benchmark"])
            speed = json.loads((root / "corpus" / "logs" / "speed_slice0.json").read_text())
            self.assertEqual((speed["cases"], speed["scenes"]), (1, 2))
            self.assertEqual(set(speed["scene_s"]), {"changes", "devices", "render", "save"})
            record = json.loads((root / "corpus" / "caseA" / "scene0001" / "params.json").read_text())
            self.assertIn("render", record["timing_s"])
            progress = corpus.status(root / "corpus")
            self.assertEqual((progress["done"], progress["expected"], progress["cases_with_errors"]), (2, 2, 0))
            self.assertIsNone(corpus.status(root / "gallery")["expected"])  # no worker started there
            estimate = corpus.projection(speed, 2000, 8)
            self.assertGreater(estimate["single_worker_hours"], 0)
            self.assertGreaterEqual(estimate["suggested_workers"], 1)
        available = [(name, "", "p", {}) for name in ("train_1_a_1", "train_2_a_1", "1.3.6.1.x", "volume-3")]
        kinds = {export.source_kind(item[0]) for item in export.diverse_choice(available, 3, 0)}
        self.assertEqual(kinds, {"ct_rate", "luna", "lits"})

    def test_ctrate_fetch_helpers_offline(self):
        from data_generation import fetch_ctrate_preview_cases as fetch

        self.assertEqual(fetch.volume_paths("train_12_a_1.nii.gz"), (
            "dataset/train_fixed/train_12/train_12_a/train_12_a_1.nii.gz",
            "dataset/ts_seg/ts_total/train_fixed/train_12/train_12_a/train_12_a_1.nii.gz"))
        rows = [dict(VolumeName=f"train_{index}_a_{recon}.nii.gz", ZSpacing=str(spacing), NumberofSlices="300",
                     XYSpacing="[0.7, 0.7]", RescaleSlope="1", RescaleIntercept="-1024")
                for index, spacing in ((1, 1.0), (2, 5.0), (3, 1.25), (4, 1.0), (5, 0.8)) for recon in (1, 2)]
        chosen = fetch.choose(rows, {"train_4_a_1.nii.gz"}, 2, 0, None)
        self.assertEqual(sorted(name for name, _ in chosen), ["train_1_a_1.nii.gz", "train_5_a_1.nii.gz"])
        raw = np.full((8, 8, 4), 1024, np.float32)
        raw[2:5, 2:5] = 0
        image, notes = fetch.to_hu(nib.Nifti1Image(raw, np.eye(4)), rows[0])
        hu = np.asarray(image.dataobj)
        self.assertEqual(int(hu.min()), -1024)
        self.assertEqual(notes["rescaled"]["intercept"], -1024.0)
        self.assertAlmostEqual(float(image.header.get_zooms()[0]), 0.7, places=3)
        ct, segs = phantom()
        labels = np.zeros(ct.shape, np.int16)
        for name, value in (("upper_right_lobe", 15), ("middle_right_lobe", 16), ("lower_right_lobe", 17),
                            ("upper_left_lobe", 13), ("lower_left_lobe", 14)):
            labels[segs[name]] = value
        labels[segs["heart"]] = 46
        version, masks, _ = fetch.decode_total(nib.Nifti1Image(labels, np.eye(4)), nib.Nifti1Image(ct, np.eye(4)))
        self.assertEqual(version, "total_v1")
        np.testing.assert_array_equal(masks["upper_left_lobe"], segs["upper_left_lobe"])
        with self.assertRaises(SystemExit):
            fetch.main(["--output", str(Path(fetch.__file__).resolve().parents[1] / "tmp_ctrate")])


if __name__ == "__main__":
    unittest.main()
