"""Supervoxel change generator on a small synthetic chest phantom (CPU, seconds)."""

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import nibabel as nib
import numpy as np

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
        for timepoint, sign in (("current", 1), ("prior", -1)):
            scene = render.render_scene(self.ctx, [(change, timepoint)], np.random.default_rng(4))
            values = scene["maps"][0]
            self.assertGreater(int((np.sign(values) == sign).sum()), 50)
            self.assertEqual(int((np.sign(values) == -sign).sum()), 0)
            self.assertEqual(scene["prior"].shape, (512, 512))
        moved = scene["registered_prior"] - scene["current"]
        self.assertGreater(float(np.abs(moved).max()), 0)

    def test_scene_changes_are_disjoint_distinct_families(self):
        changes = tasks.compose_scene(self.ctx, np.random.default_rng(7), n_changes=(3, 3))
        self.assertGreaterEqual(len(changes), 2)
        self.assertEqual(len({change.family for change, _ in changes}), len(changes))
        occupied = np.zeros(self.ctx.shape, int)
        for change, timepoint in changes:
            self.assertIn(timepoint, ("prior", "current"))
            occupied[change.delta.box] += change.delta.support()
        self.assertLessEqual(int(occupied.max()), 1)
        tiers = {change.tier for change, _ in changes}
        self.assertTrue(tiers <= {"T1", "T2"})

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
            with unittest.mock.patch.object(tasks, "CTContext", side_effect=lambda c, s: original(
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
        available = [(name, "", "p", {}) for name in ("train_1_a_1", "train_2_a_1", "1.3.6.1.x", "volume-3")]
        kinds = {export.source_kind(item[0]) for item in export.diverse_choice(available, 3, 0)}
        self.assertEqual(kinds, {"ct_rate", "luna", "lits"})


if __name__ == "__main__":
    unittest.main()
