"""CPU checks: python -m unittest training.test_dpt_head."""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.evaluate_rq1 import _tokens, run_eval, select_review_cases
from losses import change_map_loss
from models import DPTCrossAttentionHead, DPTDifferenceHead, DifferenceHead, build_change_head
from training.train_foundation_diff import head_spec, parse_args
from training import train_foundation_diff as trainer
from training.cache_features import slice_last_layer_cache


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_photometric_images_independent_bounded_and_reproducible(self):
        from datasets.augmentations import augment_cxr

        image = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).repeat(12, 1, 1, 1)
        before = image.clone()
        torch.manual_seed(42)
        augmented = augment_cxr(image)
        torch.manual_seed(42)
        self.assertTrue(torch.equal(augmented, augment_cxr(image)))
        self.assertTrue(torch.equal(image, before))
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertGreaterEqual(augmented.min().item(), 0)
        self.assertLessEqual(augmented.max().item(), 1)
        self.assertEqual(augmented.shape, image.shape)
        self.assertGreater(torch.unique(augmented.flatten(1), dim=0).shape[0], 1)
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            augment_cxr(image + 1)

    def test_online_token_augmentation_train_only_and_gt_unchanged(self):
        from datasets.augmentations import augment_cxr

        image = torch.full((8, 1, 5, 5), .5)
        gt = torch.randn_like(image)
        batch = dict(img_prior=image.clone(), img_curr=image.clone(), gt_diff=gt.clone())
        observed = []

        def backbone(images):
            observed.append(images.clone())
            self.assertFalse(torch.is_grad_enabled())
            return dict(patch_tokens=images.flatten(2).transpose(1, 2), cls_token=images.mean((2, 3)))

        torch.manual_seed(42)
        with patch.object(trainer, "augment_cxr", wraps=augment_cxr) as augmentation:
            trainer._tokens_from_batch(batch, backbone, "cpu", augment=True)
            self.assertEqual(augmentation.call_count, 2)
            self.assertFalse(torch.equal(observed[0], observed[1]))
            trainer._tokens_from_batch(batch, backbone, "cpu")
            self.assertEqual(augmentation.call_count, 2)
        self.assertTrue(torch.equal(observed[-1], image))
        self.assertTrue(torch.equal(batch["img_prior"], image))
        self.assertTrue(torch.equal(batch["gt_diff"], gt))
        with self.assertRaisesRegex(ValueError, "cached"):
            trainer._tokens_from_batch(batch, None, "cpu", augment=True)

    def test_augmentation_cli_guards(self):
        for flags in (["--photometric_augmentation"],
                      ["--photometric_augmentation", "--no_cache", "--overfit", "2"],
                      ["--benchmark_batches", "-1"]):
            with self.subTest(flags=flags), patch.object(sys, "argv", ["train", *flags]), \
                 patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_online_augmented_benchmark_training_and_resume(self):
        import nibabel as nib
        import numpy as np

        class TinyFrozenBackbone(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
                self.frozen = kwargs["freeze"]

            def forward(self, images):
                self.assertions(images)
                tokens = torch.nn.functional.adaptive_avg_pool2d(images, (5, 5)).flatten(2).transpose(1, 2)
                return dict(patch_tokens=tokens.expand(-1, -1, 16) * self.scale,
                            cls_token=images.mean((2, 3)).expand(-1, 16) * self.scale)

            def assertions(backbone, images):
                self.assertTrue(backbone.frozen)
                self.assertFalse(backbone.training)
                self.assertFalse(torch.is_grad_enabled())

        config = str(Path(__file__).resolve().parents[1] / "configs/last1_upright_ce_dice_thr0p001_aug.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split in ("train", "val"):
                for index in range(2):
                    pair_id = f"{split}_{index}"
                    folder = root / pair_id
                    folder.mkdir()
                    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
                    gt = np.zeros_like(image)
                    gt[:16] = .2
                    gt[16:] = -.2
                    for name, values in (("prior", image), ("current", image), ("diff_map", gt)):
                        nib.save(nib.Nifti1Image(values, np.eye(4)), folder / f"{name}.nii.gz")
                    records.append(dict(pair_id=pair_id, split=split, realized_change=True,
                                        effective_anomaly_type="consolidation"))
            (root / "manifest_split.jsonl").write_text("\n".join(map(json.dumps, records)))
            arguments = ["train", "--config", config, "--dataset_root", str(root),
                         "--cache_dir", str(root / "NONEXISTENT_CACHE"),
                         "--save_folder", str(root / "checkpoints"), "--plots_folder", str(root / "plots"),
                         "--batch_size", "2", "--accum", "1", "--num_workers", "0",
                         "--device", "cpu", "--dpt_channels", "8", "--epochs", "1"]
            with patch.multiple(trainer.C, BACKBONE_DIM=16, FEATURE_GRID=5, IMG_SIZE=32, EMBED_DIM=12), \
                 patch("models.FrozenCXRBackbone", TinyFrozenBackbone), \
                 patch.object(trainer, "plot_curves"), patch.object(trainer, "save_sample_plots"), \
                 patch("sys.stdout", io.StringIO()):
                with patch.object(sys, "argv", arguments + ["--benchmark_batches", "1"]):
                    trainer.main()
                report = json.loads((root / "checkpoints/benchmark.json").read_text())
                self.assertEqual(report["batches"], 1)
                self.assertGreater(report["seconds_per_batch"], 0)
                self.assertEqual(report["training_health"]["optimizer_steps"], 1)
                self.assertTrue((root / "plots/augmentation_preview.png").is_file())
                self.assertFalse((root / "checkpoints/run_config.json").exists())
                self.assertFalse(list((root / "checkpoints").glob("*.pt")))
                from datasets.augmentations import augment_cxr
                with patch.object(trainer, "augment_cxr", wraps=augment_cxr) as augmentation, \
                     patch.object(sys, "argv", arguments):
                    trainer.main()
                    self.assertEqual(augmentation.call_count, 2)
                state = torch.load(root / "checkpoints/last.pt", weights_only=True)
                self.assertEqual(state["best_score"], state["val"]["target_signed_dice"])
                self.assertTrue(state["run_config"]["no_cache"])
                with patch.object(sys, "argv", arguments + ["--resume", "--no-photometric_augmentation"]), \
                     self.assertRaisesRegex(ValueError, "photometric_augmentation"):
                    trainer.main()
                with patch.object(sys, "argv", arguments + ["--resume", "--epochs", "2"]):
                    trainer.main()
                state = torch.load(root / "checkpoints/last.pt", weights_only=True)
                self.assertEqual(state["epoch"], 2)

    def test_augmented_pipeline_config_and_artifacts(self):
        import yaml
        from training import run_ce_dice_clinical as pipeline

        baseline = yaml.safe_load(pipeline.DEFAULT_CONFIG.read_text())
        variant = yaml.safe_load((pipeline.ROOT / "configs/last1_upright_ce_dice_thr0p001_aug.yaml").read_text())
        self.assertEqual({key for key in baseline if baseline[key] != variant[key]},
                         {"save_folder", "plots_folder"})
        self.assertEqual(set(variant) - set(baseline), {"no_cache", "photometric_augmentation"})
        self.assertTrue(variant["no_cache"] and variant["photometric_augmentation"])
        self._check_ce_pipeline_artifacts("last1_upright_ce_dice_thr0p001_aug.yaml",
                                         "clinical100_upright_ce_dice_thr0p001_aug_")

    def test_ce_pipeline_failure_gate_and_scoped_complete_artifacts(self):
        self._check_ce_pipeline_artifacts('last1_upright_ce_dice_thr0p001.yaml',
                                          'clinical100_upright_ce_dice_thr0p001_')

    def test_ce_weight_decay_pipeline_and_config_isolation(self):
        import yaml
        from training import run_ce_dice_clinical as pipeline

        baseline = yaml.safe_load(pipeline.DEFAULT_CONFIG.read_text())
        variant_path = pipeline.ROOT / 'configs/last1_upright_ce_dice_thr0p001_wd0p05.yaml'
        variant = yaml.safe_load(variant_path.read_text())
        self.assertEqual({key for key in baseline if baseline[key] != variant[key]},
                         {'weight_decay', 'save_folder', 'plots_folder'})
        self.assertEqual(variant['weight_decay'], .05)
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / 'config.yaml'
            for changes in ({'save_folder': baseline['save_folder']},
                            {'weight_decay': .01}, {'plots_folder': baseline['plots_folder']}):
                with self.subTest(changes=changes):
                    config_path.write_text(yaml.safe_dump(dict(variant, **changes)))
                    with patch.object(pipeline.subprocess, 'run') as train:
                        with self.assertRaisesRegex(ValueError, 'dedicated CE run|inside their dedicated'):
                            pipeline.run(config_path, Path(temporary))
                        train.assert_not_called()
        self._check_ce_pipeline_artifacts('last1_upright_ce_dice_thr0p001_wd0p05.yaml',
                                          'clinical100_upright_ce_dice_thr0p001_wd0p05_')

    def _check_ce_pipeline_artifacts(self, config_name, result_prefix):
        import nibabel as nib
        import numpy as np
        import yaml
        from training import run_ce_dice_clinical as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'results').mkdir()
            source = root / 'current.nii.gz'
            affine = np.diag([2., 3., 1., 1.])
            nib.save(nib.Nifti1Image(np.zeros((3, 5)), affine), source)
            annotation = root / 'annotation.json'
            annotation.write_text('[]')
            cases = [dict(pair_id=f'pair{number}', prior=str(source), current=str(source),
                          annotations={doctor: str(annotation) for doctor in ('Avi', 'Benny', 'Sigal', 'Smadar', 'Nitzan')})
                     for number in range(1, 101)]
            (root / 'manifest_split.jsonl').write_text('{}\n')
            config = yaml.safe_load((pipeline.ROOT / 'configs' / config_name).read_text())
            run = root / Path(config['save_folder']).parent.name
            config.update(dataset_root=str(root), cache_dir=str(root / 'cache'),
                          save_folder=str(run / 'checkpoints'), plots_folder=str(run / 'plots'), epochs=1)
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))

            def finish_training(command, **kwargs):
                self.assertEqual(kwargs['env']['CACHE_DIR'], config['cache_dir'])
                checkpoints = run / 'checkpoints'
                if config.get("photometric_augmentation") and "--benchmark_batches" in command:
                    self.assertEqual(command, ['bash', 'training/run_last4_dpt.sh', '--benchmark_batches', '32'])
                    checkpoints.mkdir(parents=True)
                    (checkpoints / 'benchmark.json').write_text('{}')
                    plots = run / 'plots'
                    plots.mkdir()
                    (plots / 'augmentation_preview.png').write_bytes(b'preview')
                    return
                self.assertEqual(command, ['bash', 'training/run_last4_dpt.sh'])
                checkpoints.mkdir(parents=True, exist_ok=True)
                state = dict(epoch=1, head_spec=dict(kwargs=dict(output_mode='three_class')), run_config=config)
                for name in ('best.pt', 'last.pt'):
                    torch.save(state, checkpoints / name)
                (checkpoints / 'metrics.csv').write_text('epoch\n1\n')
                (checkpoints / 'run_config.json').write_text(json.dumps(config))
                (checkpoints / 'diagnostics').mkdir()
                for split in ('train', 'val'):
                    (checkpoints / 'diagnostics' / f'epoch_0001_{split}.json').write_text('{}')

            def finish_clinical(args):
                args.out_dir.mkdir(parents=True)
                metadata = dict(num_pairs=100, prediction_threshold=.5, rotation_k=3, flip_horizontal=True)
                (args.out_dir / 'metadata.json').write_text(json.dumps(metadata))
                (args.out_dir / 'report.json').write_text(json.dumps(dict(pairs_completed=100)))
                (args.out_dir / 'per_pair.csv').write_text('pair_id\n')
                for case in cases:
                    folder = args.out_dir / case['pair_id']
                    folder.mkdir()
                    for name, value in [('output.nii.gz', .6), ('mask.nii.gz', 1.),
                                        ('probability_unchanged.nii.gz', .2),
                                        ('probability_positive.nii.gz', .7),
                                        ('probability_negative.nii.gz', .1)]:
                        nib.save(nib.Nifti1Image(np.full((3, 5), value, dtype=np.float32), affine), folder / name)
                    (folder / 'source.json').write_text(json.dumps(case))

            def finish_f1(args):
                args.out_dir.mkdir()
                for name in ('metadata.json', 'report.json', 'f1_per_clinician.csv'):
                    (args.out_dir / name).write_text('{}')

            with patch.object(pipeline, 'CODE_PATHS', ()), \
                 patch.object(pipeline.subprocess, 'check_output', return_value='commit\n'), \
                 patch.object(pipeline.evaluate_clinical, 'discover_cases', return_value=cases), \
                 patch.object(pipeline.evaluate_clinical, 'run', side_effect=finish_clinical) as infer, \
                 patch.object(pipeline.compare_clinical_models, 'run', side_effect=finish_f1) as f1, \
                 patch.object(pipeline, 'publish') as publish, patch('sys.stdout', io.StringIO()):
                with patch.object(pipeline.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'train')):
                    with self.assertRaises(subprocess.CalledProcessError):
                        pipeline.run(config_path, root, True)
                infer.assert_not_called()
                f1.assert_not_called()
                publish.assert_not_called()
                with patch.object(pipeline, 'ROOT', root), \
                     patch.object(pipeline.subprocess, 'run', side_effect=finish_training):
                    destination = pipeline.run(config_path, root, True)
                publish.assert_called_once_with(destination)
                self.assertTrue(destination.name.startswith(result_prefix))
                verification = json.loads((destination / 'verification.json').read_text())
                self.assertEqual(verification['pairs_verified'], 100)
                self.assertEqual(len(verification['probability_sha256']), 100)
                self.assertEqual(len(verification['clinical_source_sha256']), 200)
                self.assertEqual(len(verification['annotation_sha256']), 500)
                self.assertEqual(len(list(destination.glob('pair*/probability_*.nii.gz'))), 300)
                self.assertFalse(list(destination.rglob('*.pt')))
                self.assertFalse(list(destination.rglob('current.nii.gz')))
                self.assertTrue((destination / 'detection_f1/f1_per_clinician.csv').is_file())

    def test_three_class_score_eval_and_probability_export(self):
        import nibabel as nib
        import numpy as np
        from models import signed_change_map
        from evaluation import evaluate_clinical as clinical

        logits = torch.tensor([[[[-8., -8., 8.]], [[8., -8., -8.]], [[-8., 8., -8.]]]])
        scores = signed_change_map(logits)
        torch.testing.assert_close(scores, torch.tensor([[[[1., -1., 0.]]]]), atol=1e-6, rtol=0)
        ambiguous = torch.tensor([[[[-8.]], [[8.]], [[8.]]]])
        self.assertEqual(float(signed_change_map(ambiguous)), 0.)
        self.assertGreater(float(ambiguous.softmax(1)[0, 1]), .49)

        class FixedHead(torch.nn.Module):
            def forward(self, *inputs):
                return logits, None

        batch = dict(p_prior=torch.zeros(1, 1, 1), p_curr=torch.zeros(1, 1, 1),
                     cls_prior=torch.zeros(1, 1), cls_curr=torch.zeros(1, 1),
                     gt_diff=torch.tensor([[[[.2, -.2, 0.]]]]), is_pathology=[1], anomaly_type=[1])
        report, _ = run_eval(FixedHead(), None, [batch], 'cpu', .5, gt_thr=.05)
        self.assertEqual(report['overall']['dice'], 1.)
        with tempfile.TemporaryDirectory() as temporary:
            shape = (7, 5)
            probabilities = logits.softmax(1).expand(1, 3, 7, 3)
            restored = np.stack([clinical.restore_prediction(channel, shape, 3, True)
                                 for channel in probabilities[0]])
            signed = restored[1] - restored[2]
            affine = np.diag([2., 3., 1., 1.])
            nifti = nib.Nifti1Image(np.zeros(shape), affine)
            masks = (np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))
            with patch.object(clinical, 'plot_case'):
                clinical.export_case(dict(pair_id='pair1'), signed, np.zeros(shape), np.zeros(shape),
                                     np.zeros(shape), nifti, {'Avi': masks}, Path(temporary), .5,
                                     probabilities=restored)
            for index, name in enumerate(('unchanged', 'positive', 'negative')):
                saved = nib.load(Path(temporary) / 'pair1' / f'probability_{name}.nii.gz')
                np.testing.assert_allclose(np.asarray(saved.dataobj), restored[index])
                np.testing.assert_array_equal(saved.affine, affine)
            saved = nib.load(Path(temporary) / 'pair1/output.nii.gz')
            np.testing.assert_allclose(np.asarray(saved.dataobj), signed)
            from training.run_ce_dice_clinical import verify_probabilities
            source = Path(temporary) / 'current.nii.gz'
            nib.save(nifti, source)
            cases = [dict(pair_id='pair1', current=str(source))]
            hashes = verify_probabilities(Path(temporary), cases)
            self.assertEqual(len(hashes['pair1']), 3)
            invalid = np.zeros(shape, dtype=np.float32)
            nib.save(nib.Nifti1Image(invalid, affine),
                     Path(temporary) / 'pair1/probability_positive.nii.gz')
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                verify_probabilities(Path(temporary), cases)

    def test_three_class_cached_run_resume_guards_and_logged_objective(self):
        config = str(Path(__file__).resolve().parents[1] / 'configs/last1_upright_ce_dice_thr0p001.yaml')
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split in ('train', 'val'):
                for index in range(2):
                    pair_id = f'{split}_{index}'
                    records.append(dict(pair_id=pair_id, split=split,
                                        effective_anomaly_type='consolidation' if index == 0 else 'none'))
                    folder = root / 'cache' / pair_id
                    folder.mkdir(parents=True)
                    gt = torch.zeros(1, 32, 32)
                    if index == 0:
                        gt[:, :16] = .2
                        gt[:, 16:] = -.2
                    torch.save(dict(p_prior=torch.randn(25, 16), p_curr=torch.randn(25, 16),
                                    cls_prior=torch.randn(16), cls_curr=torch.randn(16), gt_diff=gt,
                                    anomaly_type=1 if index == 0 else 0, direction=0,
                                    is_pathology=int(index == 0), change_group_id=pair_id,
                                    pair_id=pair_id, rotation_k=3, flip_horizontal=True), folder / 'feat.pt')
            (root / 'manifest_split.jsonl').write_text('\n'.join(map(json.dumps, records)))
            arguments = ['train', '--config', config, '--dataset_root', str(root),
                         '--cache_dir', str(root / 'cache'), '--save_folder', str(root / 'checkpoints'),
                         '--plots_folder', str(root / 'plots'), '--batch_size', '2', '--accum', '1',
                         '--num_workers', '0', '--device', 'cpu', '--dpt_channels', '8', '--epochs', '1']
            output = io.StringIO()
            with patch.multiple(trainer.C, BACKBONE_DIM=16, FEATURE_GRID=5, IMG_SIZE=32, EMBED_DIM=12), \
                 patch.object(trainer, 'plot_curves'), patch.object(trainer, 'save_sample_plots'), \
                  patch('models.FrozenCXRBackbone', side_effect=AssertionError('Cached training must not load the encoder')), \
                 patch('sys.stdout', output):
                with patch.object(sys, 'argv', arguments):
                    trainer.main()
                checkpoint = torch.load(root / 'checkpoints/last.pt', weights_only=True)
                self.assertEqual(checkpoint['head_spec']['kwargs']['output_mode'], 'three_class')
                self.assertEqual(checkpoint['head']['output.2.weight'].shape[0], 3)
                self.assertEqual(checkpoint['best_dice'], checkpoint['val']['target_signed_dice'])
                self.assertEqual(checkpoint['best_metric'], 'target_signed_dice')
                self.assertAlmostEqual(checkpoint['val']['loss_total'],
                                       checkpoint['val']['loss_ce'] + checkpoint['val']['loss_dice'],
                                       delta=1e-6)
                config_path = root / 'checkpoints/run_config.json'
                before = config_path.read_bytes()
                with patch.object(sys, 'argv', arguments), self.assertRaisesRegex(ValueError, 'already contains'):
                    trainer.main()
                with patch.object(sys, 'argv', arguments + ['--resume', '--w_ce', '2']), \
                     self.assertRaisesRegex(ValueError, 'w_ce'):
                    trainer.main()
                legacy = dict(checkpoint, run_config=dict(checkpoint['run_config'], loss_mode='regression'))
                torch.save(legacy, root / 'legacy.pt')
                with patch.object(sys, 'argv', arguments + ['--resume_from', str(root / 'legacy.pt')]), \
                     self.assertRaisesRegex(ValueError, 'loss_mode'):
                    trainer.main()
                self.assertEqual(config_path.read_bytes(), before)
                with patch.object(sys, 'argv', arguments + ['--epochs', '2', '--resume']):
                    trainer.main()
                checkpoint = torch.load(root / 'checkpoints/last.pt', weights_only=True)
                self.assertEqual(checkpoint['epoch'], 2)
                diagnostic = json.loads((root / 'checkpoints/diagnostics/epoch_0002_val.json').read_text())
                losses = diagnostic['metadata']['loss_batch_means']['val']
                self.assertEqual(set(losses), {'total', 'ce', 'dice', 'dice_positive', 'dice_negative'})
                self.assertEqual(diagnostic['groups']['overall']['pairs'], 2)
                with (root / 'checkpoints/metrics.csv').open() as stream:
                    rows = list(__import__('csv').DictReader(stream))
                self.assertEqual(len(rows), 2)
                self.assertIn('train_ce', rows[0])
                self.assertIn('val_target_signed_dice', rows[0])
            text = output.getvalue()
            self.assertIn('positive_dice_loss=', text)
            self.assertIn('negative_dice_loss=', text)
            self.assertIn('[checkpoint selection] target_signed_dice=', text)
            self.assertIn('raw_gt_abs>0.05', text)
            self.assertIn('raw_gt_abs>0.001', text)

    def test_three_class_ce_dice_boundaries_sign_and_empty_background(self):
        from losses.change_map_losses import three_class_change_map_loss

        gt = torch.tensor([[[[.001, -.001, .2, -.2, 0.]]]])
        labels = torch.tensor([[[0, 0, 1, 2, 0]]])
        logits = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float() * 16
        loss, terms = three_class_change_map_loss(logits, gt)
        self.assertLess(float(loss), 1e-5)
        self.assertLess(terms['dice_positive'], 1e-5)
        self.assertLess(terms['dice_negative'], 1e-5)
        swapped, _ = three_class_change_map_loss(logits[:, [0, 2, 1]], gt)
        self.assertGreater(float(swapped), 1)
        empty_logits = torch.zeros(2, 3, 2, 2, requires_grad=True)
        empty_gt = torch.zeros(2, 1, 2, 2)
        empty_loss, _ = three_class_change_map_loss(empty_logits, empty_gt)
        empty_loss.backward()
        self.assertTrue(torch.isfinite(empty_logits.grad).all())
        self.assertTrue((empty_logits.grad[:, 0] < 0).all())
        self.assertTrue((empty_logits.grad[:, 1:] > 0).all())
        with self.assertRaises(ValueError):
            three_class_change_map_loss(empty_logits, empty_gt, class_weights=(0, 1, 1))
        with self.assertRaises(ValueError):
            three_class_change_map_loss(empty_logits[:, :1], empty_gt)

    def test_validation_total_matches_active_training_objective(self):
        class FixedHead(torch.nn.Module):
            def forward(self, *inputs):
                return torch.tensor([[[[.8, .7, -.2, .1]]]]), None

        gt = torch.tensor([[[[.04, -.2, .00001, 0.]]]])
        batch = dict(p_prior=torch.zeros(1, 1, 1), p_curr=torch.zeros(1, 1, 1),
                     cls_prior=torch.zeros(1, 1), cls_curr=torch.zeros(1, 1),
                     gt_diff=gt, is_pathology=[1])
        options = dict(target_mode="signed_mask", target_threshold=.0001,
                       background_weight=.75, tau=.02, pos_weight=10.,
                       w_l1=.7, w_l2=1.3, w_dice=.2, w_sign=.4)
        head = FixedHead()
        _, expected = change_map_loss(head()[0], gt, **options)
        actual = trainer.evaluate(head, None, [batch], "cpu", threshold=.5,
                                  gt_threshold=.05, **options)
        for key in ("total", "dice", "sign"):
            self.assertAlmostEqual(actual[f"loss_{key}"], expected[key])
        self.assertAlmostEqual(actual["l1"], expected["l1"])
        self.assertAlmostEqual(actual["l2"], expected["l2"])

    def test_printed_metrics_distinguish_sign_cutoffs_and_empty_references(self):
        from evaluation.training_diagnostics import PredictionDiagnostics

        diagnostics = PredictionDiagnostics([.5], [.0001, .05], target_mode="signed_mask",
                                            target_threshold=.0001, components=True)
        gt = torch.tensor([[[[.04, -.2, 0., 0.]]], [[[0., 0., 0., 0.]]]])
        pred = torch.tensor([[[[-1., 1., 0., 0.]]], [[[.8, 0., 0., 0.]]]])
        diagnostics.update(pred, gt, dict(is_pathology=[1, 0], pair_id=["change", "empty"]))
        stream = io.StringIO()
        with patch("sys.stdout", stream):
            trainer.print_prediction_metrics(diagnostics, "train", .0001)
            trainer.print_prediction_metrics(None, "val")
        text = stream.getvalue()
        self.assertIn("[train online sampled]", text)
        self.assertIn("raw_gt_abs>0.05", text)
        self.assertIn("raw_gt_abs>0.0001", text)
        self.assertIn("support_dice=1.0000 support_iou=1.0000 signed_dice=0.0000", text)
        self.assertIn("nuisance_pairs=1 fp_area_mean=0.2500", text)
        self.assertIn("pooled_signed_F1=0.0000", text)
        self.assertIn("[val fixed head] overlap diagnostics not collected", text)
        empty = PredictionDiagnostics([.5], [.05])
        with patch("sys.stdout", stream):
            trainer.print_prediction_metrics(empty, "val")
        self.assertIn("support_dice=n/a", stream.getvalue())
        with patch.object(sys, "argv", ["train", "--target_mode", "signed_mask",
                                        "--target_threshold", ".0001"]):
            args = parse_args()
        self.assertIn(.0001, trainer.make_diagnostics(args).gt_thresholds)

    def test_signed_overlap_empty_references_and_exports(self):
        import csv
        import gzip
        from evaluation.training_diagnostics import PredictionDiagnostics, gt_statistics

        gt = torch.tensor([[[[1., -1.], [0., 0.]]], [[[0., 0.], [0., 0.]]]])
        pred = torch.tensor([[[[-1., 1.], [0., 0.]]], [[[.8, 0.], [0., 0.]]]], requires_grad=True)
        diagnostics = PredictionDiagnostics([.5], [0., .05], target_mode="signed_mask", components=True)
        diagnostics.update(pred, gt, dict(anomaly_type=[1, 0], is_pathology=[1, 0], pair_id=["change", "empty"]))
        changed, empty = diagnostics.rows
        self.assertEqual(changed["support_dice"], 1)
        self.assertEqual(changed["signed_dice"], 0)
        self.assertEqual(changed["wrong_sign_pixels"], 2)
        self.assertIsNone(empty["signed_dice"])
        self.assertEqual(empty["predicted_area"], .25)
        self.assertEqual(empty["positive_precision"], 0)
        self.assertIsNone(pred.grad)
        self.assertAlmostEqual(changed["weighted_l1"], changed["foreground_l1_contribution"] +
                               changed["background_l1_contribution"])
        stats = gt_statistics([0., 1e-8, .04, -.2], [0., .05])
        self.assertEqual(stats["cutoff_0_pixels"], 3)
        self.assertEqual(stats["cutoff_0.05_pixels"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics.save(temporary, 1, "train")
            report = json.loads((Path(temporary) / "epoch_0001_train.json").read_text())
            self.assertEqual(report["groups"]["pathology"]["signed_dice"]["n"], 1)
            self.assertIsNone(report["groups"]["nuisance"]["signed_dice"]["mean"])
            with gzip.open(Path(temporary) / "epoch_0001_train.csv.gz", "rt") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_component_matching_agrees_with_legacy_reference_first(self):
        import numpy as np
        from evaluation.evaluate_clinical import clinical_tools
        from evaluation.training_diagnostics import component_counts

        tools = clinical_tools()
        tools.struct = tools.STRUCT_3x3
        generator = np.random.default_rng(42)
        for _ in range(12):
            pred, reference = generator.random((2, 16, 16)) > .7
            expected, _ = tools.get_pairwise_detections(reference, pred)
            self.assertEqual(component_counts(pred, reference)["matched"], expected)

    def test_candidate_cutoff_components_signs_strict_boundary_and_fragmentation(self):
        import numpy as np
        from evaluation.audit_gt import component_statistics

        values = np.zeros((8, 12), dtype=np.float32)
        values[1:3, 1:3] = .03
        values[1:3, 4:6] = .03
        values[1, 3] = .005
        values[5, 8] = -.03
        values[6, 9] = -.03
        values[7, 0] = .01
        stats = component_statistics(values, [.001, .01, .02])
        self.assertEqual(stats["components_0.001_positive_count"], 2)
        self.assertEqual(stats["components_0.01_positive_count"], 2)
        self.assertEqual(stats["components_0.001_positive_pixels"], 10)
        self.assertEqual(stats["components_0.01_positive_pixels"], 8)
        self.assertEqual(stats["components_0.001_positive_under_5px"], 1)
        self.assertEqual(stats["components_0.001_positive_under_5px_pixels"], 1)
        self.assertEqual(stats["components_0.01_positive_under_5px_pixels"], 8)
        self.assertEqual(stats["components_0.02_negative_count"], 1)
        self.assertEqual(stats["components_0.02_negative_pixels"], 2)
        self.assertEqual(stats["components_0.02_negative_under_50px_pixel_fraction"], 1)
        self.assertAlmostEqual(stats["components_0.02_positive_area"], 8 / 96)
        self.assertEqual(stats["components_0.02_positive_largest_component_pixel_fraction"], .5)
        empty = component_statistics(np.zeros_like(values), [.001])
        self.assertEqual(empty["components_0.001_negative_count"], 0)
        self.assertIsNone(empty["components_0.001_negative_under_50px_pixel_fraction"])

    def test_gt_audit_end_to_end_and_split_exclusion(self):
        import nibabel as nib
        import numpy as np
        from types import SimpleNamespace
        from evaluation.audit_gt import DEFAULT_THRESHOLDS, run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split, amplitude in (("train", .04), ("val", -.12), ("test", .2)):
                folder = root / "data" / split
                folder.mkdir(parents=True)
                image = np.arange(96, dtype=np.float32).reshape(8, 12)
                gt = np.zeros_like(image)
                gt[2:5, 4:8] = amplitude
                for name, values in (("prior", image), ("current", image + gt), ("diff_map", gt)):
                    nib.save(nib.Nifti1Image(values, np.eye(4)), folder / f"{name}.nii.gz")
                records.append(dict(pair_id=split, split=split, effective_anomaly_type="consolidation"))
            (root / "data" / "manifest_split.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            args = SimpleNamespace(dataset_root=root / "data", out_dir=root / "audit", splits=["train", "val"],
                                   limit=0, seed=42, img_size=16, rotation_k=3, thresholds=DEFAULT_THRESHOLDS,
                                   component_thresholds=[0., .05], overlays_per_group=1)
            report = run(args)
            self.assertEqual(report["audited_pairs"], 2)
            self.assertFalse(report["issues"])
            self.assertEqual(len(report["previews"]), 2)
            self.assertTrue((args.out_dir / "cutoff_diagnostics.png").is_file())
            for preview in report["previews"]:
                with np.load(args.out_dir / preview["arrays"], allow_pickle=False) as arrays:
                    self.assertEqual(arrays["gt_native"].shape, (8, 12))
                    self.assertEqual(arrays["gt_training_float32"].shape, (16, 16))
                    self.assertEqual(str(arrays["pair_id"]), preview["pair_id"])
            self.assertNotIn("test", {row["split"] for row in report["summaries"]})
            summary = next(row for row in report["summaries"] if row["split"] == "train" and
                           row["stage"] == "training_float32" and row["anomaly_type"] == "overall")
            self.assertEqual(summary["cutoff_0.05_entirely_removed_pairs"], 1)
            from evaluation.compare_gt_thresholds import run as compare_targets
            comparison = compare_targets(SimpleNamespace(audit_dir=args.out_dir, evidence_dir=None,
                                                         out_dir=root / "comparison",
                                                         thresholds=[.001, .01, .02]))
            self.assertEqual(len(comparison["summaries"]), 12)
            self.assertFalse(any(row["component_metrics_available"] for row in comparison["summaries"]))
            with self.assertRaisesRegex(ValueError, "new output directory"):
                compare_targets(SimpleNamespace(audit_dir=args.out_dir, evidence_dir=None,
                                                out_dir=root / "comparison", thresholds=[.001, .01, .02]))
            with self.assertRaisesRegex(ValueError, "new output directory"):
                run(args)
            corrected_args = SimpleNamespace(**{**vars(args), "out_dir": root / "publication" / "audit",
                                                 "flip_horizontal": True, "verify_alignment": True,
                                                 "target_threshold": 0.0001,
                                                 "component_thresholds": [.001, .01, .02]})
            corrected = run(corrected_args)
            self.assertEqual(corrected["alignment_verified_pairs"], 2)
            self.assertFalse(corrected["issues"])
            with patch("training.run_upright_clinical.publish") as publisher:
                comparison = compare_targets(SimpleNamespace(audit_dir=corrected_args.out_dir, evidence_dir=None,
                                                             out_dir=root / "publication" / "comparison",
                                                             thresholds=[.001, .01, .02], publish=True))
                publisher.assert_called_once_with(root / "publication")
            with patch("training.run_upright_clinical.publish") as publisher:
                with self.assertRaisesRegex(ValueError, "dedicated parent"):
                    compare_targets(SimpleNamespace(audit_dir=corrected_args.out_dir, evidence_dir=None,
                                                    out_dir=root / "publication" / "another_comparison",
                                                    thresholds=[.001, .01, .02], publish=True))
                publisher.assert_not_called()
            self.assertTrue(all(row["component_metrics_available"] for row in comparison["summaries"]))
            train_target = next(row for row in comparison["summaries"] if row["split"] == "train" and
                                row["anomaly_type"] == "overall" and row["threshold"] == .01)
            self.assertEqual(train_target["positive_components"], 1)
            self.assertEqual(train_target["negative_components"], 0)
            self.assertEqual(train_target["mixed_sign_pairs"], 0)
            self.assertAlmostEqual(train_target["background_area"] + train_target["positive_area"], 1)
            preview = corrected["previews"][0]
            with np.load(corrected_args.out_dir / preview["arrays"], allow_pickle=False) as arrays:
                expected = torch.nn.functional.interpolate(
                    torch.from_numpy(arrays["gt_native"])[None, None], (16, 16),
                    mode="bilinear", align_corners=False)[0, 0].numpy().T
                np.testing.assert_array_equal(arrays["gt_training_float32"], expected)
            wrong_affine = np.eye(4)
            wrong_affine[0, 3] = 2
            native_path = root / "data/train/diff_map.nii.gz"
            native = np.asarray(nib.load(native_path).dataobj)
            nib.save(nib.Nifti1Image(native, wrong_affine), native_path)
            corrected_args.out_dir = root / "misaligned"
            broken = run(corrected_args)
            self.assertEqual(broken["alignment_verified_pairs"], 1)
            self.assertTrue(any("affine mismatch" in issue["error"] for issue in broken["issues"]))

    def test_diagnostics_do_not_change_optimizer_update(self):
        class TinyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output = torch.nn.Parameter(torch.tensor([[[[.2, -.1], [.05, 0.]]]]))

            def forward(self, *inputs):
                return self.output, None

        arguments = ["train", "--accum", "1", "--w_dice", "0", "--w_sign", "0", "--w_l2", "1"]
        with patch.object(sys, "argv", arguments):
            args = parse_args()
        batch = dict(p_prior=torch.zeros(1, 1, 1), p_curr=torch.zeros(1, 1, 1),
                     cls_prior=torch.zeros(1, 1), cls_curr=torch.zeros(1, 1),
                     gt_diff=torch.tensor([[[[.1, -.2], [0., 0.]]]]), is_pathology=[1], pair_id=["toy"])
        results = []
        for diagnostics in (None, trainer.make_diagnostics(args)):
            head = TinyHead()
            optimizer = torch.optim.SGD(head.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
            trainer.train_one_epoch(head, None, [batch], optimizer, scheduler, "cpu", args,
                                    diagnostics=diagnostics)
            results.append(head.output.detach().clone())
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


class ClinicalInferenceTests(unittest.TestCase):
    def test_cache_input_evidence_exports_six_real_loader_examples_and_scopes_publish(self):
        import contextlib
        import hashlib
        import nibabel as nib
        import numpy as np
        import yaml
        from types import SimpleNamespace
        from evaluation import export_cache_input_evidence as evidence
        from datasets.pair_dataset import LongitudinalPairDataset

        with tempfile.TemporaryDirectory(dir=evidence.ROOT / "results") as temporary:
            root = Path(temporary)
            data, cache, clinical = root / "data", root / "cache", root / "clinical"
            data.mkdir()
            cache.mkdir()
            records = []
            shape = (23, 17)
            for number, anomaly in enumerate(("consolidation", "fluid_overload", "pleural_effusion")):
                pair_id = f"case{number}/pair0/variant0"
                pair = data / pair_id
                pair.mkdir(parents=True)
                image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                gt = np.zeros(shape, dtype=np.float32)
                gt[2:8, 3:9], gt[12:17, 10:14] = .00010002, -.2
                for name, array in (("prior.nii.gz", image), ("current.nii.gz", image + 2),
                                    ("diff_map.nii.gz", gt)):
                    nib.save(nib.Nifti1Image(array, np.eye(4)), pair / name)
                records.append(dict(pair_id=pair_id, split="train", effective_anomaly_type=anomaly,
                                    realized_change=True, direction="mixed", change_group_id=f"group{number}"))
            manifest = data / "manifest_split.jsonl"
            manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            config = dict(dataset_root=str(data), cache_dir=str(cache), save_folder=str(root / "checkpoints"),
                          feat_last_k=1, rotation_k=3, flip_horizontal=True, target_mode="signed_mask",
                          target_threshold=.0001, eval_gt_threshold=.05, eval_threshold=.5,
                          tau=.02, pos_weight=10., background_weight=.75)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            Path(config["save_folder"]).mkdir()
            (Path(config["save_folder"]) / "run_config.json").write_text(json.dumps(config))
            processor = SimpleNamespace(image_mean=[.5] * 3, image_std=[.5] * 3)
            with patch.object(evidence.C, "IMG_SIZE", 16), patch.object(evidence.C, "BACKBONE_IMG_SIZE", 18):
                dataset = LongitudinalPairDataset(str(data), split="train", img_size=16,
                                                  rotation_k=3, flip_horizontal=True)
                source_hashes = {}
                for index, record in enumerate(records):
                    sample = dataset[index]
                    payload = {key: value for key, value in sample.items() if not key.startswith("img_")}
                    payload["gt_diff"] = sample["gt_diff"].half()
                    payload.update(p_prior=torch.zeros(4, 8), p_curr=torch.zeros(4, 8),
                                   cls_prior=torch.zeros(8), cls_curr=torch.zeros(8))
                    folder = cache / record["pair_id"]
                    folder.mkdir(parents=True)
                    torch.save(payload, folder / "feat.pt")
                    source_hashes[str(folder / "feat.pt")] = evidence.sha256(folder / "feat.pt")
                cache_config = dict(backbone=evidence.C.RAD_DINO_MODEL, last_n_layers=1,
                                    rotation_k=3, image_size=16, flip_horizontal=True,
                                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
                (cache / "cache_config.json").write_text(json.dumps(cache_config))
                for number in range(1, 4):
                    pair = clinical / "Pairs1" / f"pair{number}"
                    pair.mkdir(parents=True)
                    image = np.arange(31 * 19, dtype=np.float32).reshape(31, 19)
                    for name in ("0_prior.nii.gz", "1_current.nii.gz"):
                        nib.save(nib.Nifti1Image(image, np.eye(4)), pair / name)
                    for doctor in evidence.evaluate_clinical.PHYSICIANS:
                        path = clinical / "Annotations" / doctor / f"pair{number}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps([{}, dict(cx=5, cy=7, rx=2, ry=2,
                                                             angle=0, label="Appearance")]))
                args = SimpleNamespace(config=config_path, dataset_root=None, cache_dir=None,
                                       clinical_root=clinical, clinical_pairs=[1, 2, 3],
                                       out_dir=root / "export", publish=True)
                with patch("transformers.AutoImageProcessor.from_pretrained", return_value=processor), \
                     patch("training.run_upright_clinical.publish") as publish, \
                     contextlib.redirect_stdout(io.StringIO()):
                    destination = evidence.run(args)
                publish.assert_called_once_with(destination)
                report = json.loads((destination / "report.json").read_text())
                self.assertEqual(len(report["examples"]), 6)
                self.assertFalse(report["model_weights_loaded"])
                self.assertFalse(report["inference_performed"])
                self.assertFalse(report["existing_cache_modified"])
                self.assertIsNotNone(report["active_run_config_sha256"])
                self.assertEqual(len(list(destination.glob("*/review.png"))), 6)
                for example in report["examples"]:
                    folder = destination / example["folder"]
                    self.assertTrue((folder / "current_cache_input.png").is_file())
                    self.assertTrue((folder / "current_encoder_rgb_preview.png").is_file())
                    with np.load(folder / "exact_tensors.npz") as arrays:
                        self.assertEqual(arrays["current_pixel_values"].shape, (3, 18, 18))
                        self.assertEqual(arrays["current_cache_input"].shape, (1, 16, 16))
                        if example["kind"] == "drr_train":
                            gt = torch.from_numpy(arrays["cached_continuous_gt"])
                            expected = evidence.signed_target(gt, .0001).numpy()
                            np.testing.assert_array_equal(arrays["actual_loss_target"], expected)
                            self.assertTrue(example["cached_gt_matches_fresh_half"])
                        else:
                            self.assertEqual(arrays["native_Avi_positive"].shape, (31, 19))
                            self.assertTrue(example["model_space_reference_is_display_only"])
                for path, digest in source_hashes.items():
                    self.assertEqual(evidence.sha256(path), digest)
                args.out_dir = root / "wrong_cache"
                cache_config["flip_horizontal"] = False
                (cache / "cache_config.json").write_text(json.dumps(cache_config))
                with self.assertRaisesRegex(ValueError, "cache metadata"):
                    evidence.run(args)
                self.assertFalse(args.out_dir.exists())

    def test_model_free_cache_capture_runs_actual_backbone_preprocessing(self):
        from types import SimpleNamespace
        from evaluation.export_cache_input_evidence import CacheInputCapture

        processor = SimpleNamespace(image_mean=[.2, .3, .4], image_std=[.5, .6, .7])
        with patch("transformers.AutoImageProcessor.from_pretrained", return_value=processor):
            capture = CacheInputCapture(1)
        images = torch.arange(35, dtype=torch.float32).reshape(1, 1, 7, 5) / 34
        actual = capture(images)
        expected = torch.nn.functional.interpolate(images.repeat(1, 3, 1, 1),
                    size=(capture.backbone.backbone_img_size,) * 2,
                    mode="bilinear", align_corners=False)
        expected = (expected - capture.backbone._mean) / capture.backbone._std
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(list(capture.backbone.parameters()), [])

    def test_prediction_mirror_is_horizontal_after_display_transpose(self):
        import numpy as np
        from evaluation.mirror_clinical_predictions import mirror_prediction

        scores = np.arange(35, dtype=np.float32).reshape(7, 5)
        original = scores.copy()
        mirrored = mirror_prediction(scores)
        np.testing.assert_array_equal(mirrored.T, np.fliplr(scores.T))
        self.assertFalse(np.array_equal(mirrored, np.fliplr(scores)))
        np.testing.assert_array_equal(mirror_prediction(mirrored), original)
        np.testing.assert_array_equal(scores, original)
        self.assertFalse(np.shares_memory(mirrored, scores))
        for invalid in (np.zeros((2, 3, 1)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                mirror_prediction(invalid)

    def test_prediction_mirror_export_and_real_ov_preserve_originals(self):
        import contextlib
        import csv
        import hashlib
        import nibabel as nib
        import numpy as np
        from types import SimpleNamespace
        from evaluation import compare_clinical_models as comparison
        from evaluation import mirror_clinical_predictions as mirror

        tools = comparison.clinical_tools()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clinical_root = root / "clinical"
            pair = clinical_root / "Pairs1" / "pair1"
            pair.mkdir(parents=True)
            shape = (24, 18)
            affine = np.diag([2., 3., 1., 1.])
            image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            for filename in ("0_prior.nii.gz", "1_current.nii.gz"):
                nib.save(nib.Nifti1Image(image, affine), pair / filename)
            annotations = [{}, dict(cx=5, cy=4, rx=2, ry=2, angle=0, label="Appearance"),
                           dict(cx=6, cy=13, rx=2, ry=2, angle=0, label="Disappearance")]
            hashes = {}
            for doctor in comparison.PHYSICIANS:
                path = clinical_root / "Annotations" / doctor / "pair1.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(annotations))
                hashes[f"pair1/{doctor}"] = hashlib.sha256(path.read_bytes()).hexdigest()
            positive, negative = tools.load_labels_map(path, shape)
            target = .8 * (positive > 0).astype(np.float32) - .8 * (negative < 0).astype(np.float32)
            source_run = root / "source"
            (source_run / "pair1").mkdir(parents=True)
            source = source_run / "pair1" / "output.nii.gz"
            scores = np.flip(target, axis=0).copy()
            nib.save(nib.Nifti1Image(scores, affine), source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            metadata = dict(num_pairs=1, prediction_threshold=.5, checkpoint_epoch=49,
                            checkpoint_sha256="1" * 64, annotation_sha256=hashes,
                            image_size=24, run_config=dict(rotation_k=3))
            (source_run / "metadata.json").write_text(json.dumps(metadata))
            args = SimpleNamespace(clinical_run=source_run, clinical_root=clinical_root,
                                   out_dir=root / "mirrored", num_pairs=1)
            with patch.object(comparison, "clinical_tools", return_value=tools), \
                 patch.object(tools, "plot_matrix"), contextlib.redirect_stdout(io.StringIO()):
                mirror.run(args)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            output = nib.load(args.out_dir / "predictions" / "pair1" / "output.nii.gz")
            np.testing.assert_array_equal(np.asarray(output.dataobj), target)
            np.testing.assert_array_equal(output.affine, affine)
            with (args.out_dir / "f1_comparison.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            for row in rows:
                self.assertEqual(float(row["original_F1_all"]), 0.)
                self.assertEqual(float(row["mirrored_F1_all"]), 1.)
            verification = json.loads((args.out_dir / "verification.json").read_text())
            self.assertTrue(verification["human_human_matrices_unchanged"])
            self.assertTrue(verification["original_predictions_unchanged"])
            self.assertEqual(verification["transform"]["nifti_array_axis"], 0)
            self.assertTrue((args.out_dir / "predictions" / "pair1" / "review.png").is_file())
            with self.assertRaisesRegex(ValueError, "new output directory"):
                mirror.run(args)
            args.out_dir = root / "changed_annotations"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Changed annotation"):
                mirror.run(args)
            self.assertFalse(args.out_dir.exists())

    def test_detection_only_cli_does_not_require_itamar(self):
        import contextlib
        from evaluation import compare_clinical_models as comparison

        arguments = ["compare", "--clinical-run", "saved", "--clinical-root", "clinical",
                     "--out-dir", "scores"]
        with patch.object(sys, "argv", arguments + ["--foundation-only"]):
            args = comparison.parse_args()
        self.assertTrue(args.foundation_only)
        self.assertIsNone(args.itamar_preds)
        with patch.object(sys, "argv", arguments), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                comparison.parse_args()
        self.assertEqual(caught.exception.code, 2)
        with patch.object(sys, "argv", arguments + ["--itamar-preds", "baseline"]):
            self.assertFalse(comparison.parse_args().foundation_only)

    def test_saved_detection_f1_uses_real_legacy_matching_and_checks_inputs(self):
        import contextlib
        import csv
        import hashlib
        import nibabel as nib
        import numpy as np
        from types import SimpleNamespace
        from evaluation import compare_clinical_models as comparison

        tools = comparison.clinical_tools()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clinical_root = root / "clinical"
            pair = clinical_root / "Pairs1" / "pair1"
            pair.mkdir(parents=True)
            shape = (24, 24)
            for filename in ("0_prior.nii.gz", "1_current.nii.gz"):
                nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), pair / filename)
            annotations = [{}, dict(cx=5, cy=5, rx=2, ry=2, angle=0, label="Appearance"),
                           dict(cx=16, cy=5, rx=2, ry=2, angle=0, label="Appearance"),
                           dict(cx=16, cy=16, rx=2, ry=2, angle=0, label="Disappearance")]
            hashes = {}
            for doctor in comparison.PHYSICIANS:
                path = clinical_root / "Annotations" / doctor / "pair1.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(annotations))
                hashes[f"pair1/{doctor}"] = hashlib.sha256(path.read_bytes()).hexdigest()
            positive, negative = tools.load_labels_map(path, shape)
            scores = np.zeros(shape, dtype=np.float32)
            scores[tuple(np.argwhere(positive > 0)[0])] = .8
            scores[tuple(np.argwhere(negative < 0)[0])] = -.8
            scores[22, 1] = .8
            clinical_run = root / "predictions"
            (clinical_run / "pair1").mkdir(parents=True)
            prediction = clinical_run / "pair1" / "output.nii.gz"
            nib.save(nib.Nifti1Image(scores, np.eye(4)), prediction)
            metadata = dict(num_pairs=1, prediction_threshold=.5, checkpoint_epoch=49,
                            checkpoint_sha256="1" * 64, annotation_sha256=hashes)
            (clinical_run / "metadata.json").write_text(json.dumps(metadata))
            args = SimpleNamespace(clinical_run=clinical_run, clinical_root=clinical_root,
                                   out_dir=root / "scores", num_pairs=1,
                                   foundation_only=True, itamar_preds=None)
            with patch.object(comparison, "clinical_tools", return_value=tools), \
                 patch.object(tools, "plot_matrix"), contextlib.redirect_stdout(io.StringIO()):
                comparison.run(args)
            with (args.out_dir / "f1_per_clinician.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["clinician"] for row in rows}, set(comparison.PHYSICIANS))
            for row in rows:
                self.assertAlmostEqual(float(row["F1 pos"]), .5)
                self.assertAlmostEqual(float(row["F1 neg"]), 1.)
                self.assertAlmostEqual(float(row["F1 all"]), 2 / 3)
            exported = json.loads((args.out_dir / "metadata.json").read_text())
            self.assertEqual(exported["annotation_sha256"], hashes)
            self.assertEqual(exported["prediction_sha256"]["pair1"],
                             hashlib.sha256(prediction.read_bytes()).hexdigest())
            self.assertEqual(exported["protocol"]["pairs_processed"], 1)
            self.assertEqual(exported["protocol"]["model_space"], "original")
            self.assertEqual(exported["protocol"]["prediction_threshold"], .5)
            self.assertEqual(exported["protocol"]["min_cc_size"], 0)
            self.assertEqual(exported["protocol"]["min_cc_intensity"], 0)
            args.out_dir = root / "changed_annotations"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Changed annotation"):
                comparison.run(args)
            self.assertFalse(args.out_dir.exists())
            path.write_text(json.dumps(annotations))
            prediction.unlink()
            with self.assertRaises(FileNotFoundError):
                comparison.run(args)
            self.assertFalse(args.out_dir.exists())

    def test_documented_clinical_guard_normalizes_checkpoint_spec(self):
        import contextlib
        import hashlib
        from evaluation import evaluate_clinical as clinical

        study_root = Path(__file__).resolve().parents[1]
        instructions = (study_root / "configs" / "README.md").read_text(encoding="utf-8")
        section = instructions.split("## Clinical Cutoff0.0001 Best-Checkpoint Evaluation", 1)[1]
        guard = section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotation.json"
            annotation.write_text("[]")
            cases = [dict(pair_id=f"pair{index}", annotations={
                doctor: str(annotation) for doctor in clinical.PHYSICIANS})
                for index in range(1, 101)]
            spec = head_spec(768, "dpt_cross_attention", 96)
            reference = dict(head_spec=json.loads(json.dumps(spec)), rotation_k=3,
                             prediction_threshold=0.5, num_pairs=100, annotation_sha256={
                f"{case['pair_id']}/{doctor}": hashlib.sha256(annotation.read_bytes()).hexdigest()
                for case in cases for doctor in clinical.PHYSICIANS})
            metadata = root / "review_outputs" / "clinical100_20261017_112038" / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps(reference))
            checkpoint = root / "run" / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(dict(epoch=49, head_spec=spec, run_config=dict(
                target_mode="signed_mask", target_threshold=0.0001,
                rotation_k=3, eval_threshold=0.5)), checkpoint)
            with contextlib.chdir(root), \
                 patch.dict(os.environ, RUN=str(root / "run"), CLINICAL_ROOT=str(root)), \
                 patch.object(clinical, "discover_cases", return_value=cases):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(guard, "documented_clinical_guard", "exec"), {})
                self.assertEqual(output.getvalue().strip(),
                                 hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                reference["head_spec"]["kwargs"]["channels"] = 97
                metadata.write_text(json.dumps(reference))
                with self.assertRaisesRegex(AssertionError, "Head architecture changed"):
                    exec(compile(guard, "documented_clinical_guard", "exec"), {})

    def test_clinical_launcher_checkpoint_hash_override(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = (str(git_bash) if git_bash.is_file() else None) if os.name == "nt" else shutil.which("bash")
        if bash is None:
            self.skipTest("Native Bash is required to exercise the clinical launcher")
        launcher = Path(__file__).resolve().with_name("run_clinical100.sh")
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary) / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "activate").write_text("")
            environment = dict(os.environ, FCD_VENV=venv.as_posix(),
                               RUN="/tmp/clinical-test-run", OUT="/tmp/clinical-test-out")
            environment.pop("EXPECTED_SHA256", None)
            for override in (None, "1" * 64):
                with self.subTest(override=override):
                    if override is not None:
                        environment["EXPECTED_SHA256"] = override
                    result = subprocess.run(
                        [bash, "-c", 'python() { printf "%s\\n" "$@"; }; source "$1"',
                         "clinical-launcher-test", launcher.as_posix()],
                        env=environment, capture_output=True, text=True, check=True)
                    arguments = result.stdout.splitlines()
                    expected = override or "0b7b49cc916806453ae5a2348e8b0546dd59dfbe5ff0449c034ba43d214a8053"
                    self.assertEqual(arguments[arguments.index("--expected_sha256") + 1], expected)
                    self.assertEqual(arguments[arguments.index("--threshold") + 1], "0.5")
                    self.assertEqual(arguments[arguments.index("--num_pairs") + 1], "100")
                    self.assertEqual(arguments[arguments.index("--ckpt") + 1],
                                     "/tmp/clinical-test-run/checkpoints/best.pt")

    def test_legacy_original_and_square_crop_coordinates(self):
        import nibabel as nib
        import numpy as np
        from evaluation.evaluate_clinical import clinical_tools

        tools = clinical_tools()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.nii.gz"
            scores = np.zeros((4, 4), dtype=np.float32)
            scores[0, 1], scores[2, 3], scores[1, 2] = 0.8, -0.8, 0.5
            nib.save(nib.Nifti1Image(scores, np.eye(4)), output)
            positive, negative, restored = tools.load_model_labels_map(
                output, None, scores.shape, 0, 0, prediction_threshold=0.5)
            np.testing.assert_array_equal(restored, scores)
            np.testing.assert_array_equal(positive, scores > 0.5)
            np.testing.assert_array_equal(negative, scores < -0.5)
            with self.assertRaises(ValueError):
                tools.load_model_labels_map(output, None, (4, 5), 0, 0)
            crop = Path(temporary) / "crop_info.json"
            crop.write_text(json.dumps(dict(current=dict(sq_x_min=1, sq_y_min=2,
                                                         square_size=4, orig_h=7, orig_w=9))))
            expected = np.zeros((7, 9))
            expected[1:5, 2:6] = scores.T
            positive, negative, restored = tools.load_model_labels_map(
                output, crop, (9, 7), 0, 0)
            np.testing.assert_array_equal(restored, expected.T)
            np.testing.assert_array_equal(positive, expected.T > 0)
            np.testing.assert_array_equal(negative, expected.T < 0)

    def test_clinical_annotation_reader_does_not_require_seaborn(self):
        from evaluation.evaluate_clinical import clinical_tools

        clinical_tools()
        with patch.dict(sys.modules, {"seaborn": None}):
            tools = clinical_tools()
            with tempfile.TemporaryDirectory() as temporary:
                doctor = Path(temporary) / "Avi"
                doctor.mkdir()
                annotation = doctor / "pair1.json"
                annotation.write_text(json.dumps([{}, dict(cx=4, cy=5, rx=2, ry=2,
                                                           angle=0, label="Appearance")]))
                resolved = tools.resolve_annotation_path(temporary, "Avi", 1)
                positive, negative = tools.load_labels_map(resolved, (11, 13))
                self.assertTrue(positive.any())
                self.assertFalse(negative.any())

    def test_training_transform_and_asymmetric_inverse(self):
        import numpy as np
        from datasets.pair_dataset import _norm01, _to_chw
        from evaluation.evaluate_clinical import prepare_image, restore_prediction

        array = np.arange(17 * 29, dtype=np.float32).reshape(17, 29)
        expected = torch.rot90(_norm01(_to_chw(array, 32)), 3, (-2, -1))
        torch.testing.assert_close(prepare_image(array, 32, 3), expected)
        original_display = _norm01(_to_chw(array, 32))[0].numpy().T
        np.testing.assert_array_equal(prepare_image(array, 32, 3)[0].numpy(),
                          np.fliplr(original_display))
        np.testing.assert_array_equal(prepare_image(array, 32, 3, True)[0].numpy(),
                                      original_display)
        landmarks = np.zeros((17, 29), dtype=np.float32)
        landmarks[2, 5], landmarks[14, 24] = 1, -1
        for rotation in range(4):
            rotated = torch.rot90(torch.from_numpy(landmarks), rotation, (-2, -1))
            np.testing.assert_array_equal(restore_prediction(rotated, landmarks.shape, rotation),
                                          landmarks)
            flipped = torch.flip(rotated, (-1,))
            np.testing.assert_array_equal(restore_prediction(flipped, landmarks.shape, rotation, True),
                                          landmarks)

    def test_signed_metrics_and_empty_references(self):
        import numpy as np
        from evaluation.evaluate_clinical import score_masks

        positive = np.zeros((9, 13), dtype=bool)
        positive[2:5, 3:7] = True
        empty = np.zeros_like(positive)
        self.assertEqual(score_masks((positive, empty), (positive, empty))["signed_dice"], 1)
        self.assertEqual(score_masks((empty, positive), (positive, empty))["signed_dice"], 0)
        self.assertIsNone(score_masks((empty, empty), (empty, empty))["signed_dice"])
        self.assertGreater(score_masks((positive, empty), (empty, empty))["fp_area_if_no_change"], 0)

    def test_mocked_clinical_export_preserves_geometry_and_provenance(self):
        import contextlib
        import nibabel as nib
        import numpy as np
        from types import SimpleNamespace
        from evaluation import evaluate_clinical as clinical

        class TinyBackbone(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, image):
                return dict(patch_tokens=image, cls_token=torch.zeros(len(image), 1))

        class TinyHead(torch.nn.Module):
            def forward(self, prior, current, *args):
                return (current - prior).clamp(-1, 1), None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (17, 29)
            affine = np.diag([0.7, 0.9, 1, 1])
            prior, current = root / "prior.nii.gz", root / "current.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), prior)
            nib.save(nib.Nifti1Image(np.arange(17 * 29, dtype=np.float32).reshape(shape), affine), current)
            annotation = root / "annotation.json"
            annotation.write_text("[]")
            case = dict(pair_id="pair1", prior=str(prior), current=str(current),
                        annotations={doctor: str(annotation) for doctor in clinical.PHYSICIANS})
            tools = SimpleNamespace(load_labels_map=lambda path, shape:
                                    (np.ones(shape), np.zeros(shape)))
            checkpoint = root / "fake.pt"
            torch.save(dict(epoch=45, head={}, head_spec=dict(type="test_stub", kwargs=dict(
                backbone_dim=1, patch_dim=1, out_size=32)),
                run_config=dict(rotation_k=3, eval_threshold=0.5, target_mode="signed_mask")), checkpoint)
            args = SimpleNamespace(clinical_root=root, num_pairs=1, preflight=False,
                                   ckpt=checkpoint, out_dir=root / "out", device="cpu",
                                   threshold=None, expected_sha256=None)
            with patch.object(clinical, "discover_cases", return_value=[case]), \
                 patch.object(clinical, "clinical_tools", return_value=tools), \
                 patch("models.FrozenCXRBackbone", TinyBackbone), \
                 patch("models.build_change_head", return_value=TinyHead()), \
                 contextlib.redirect_stdout(io.StringIO()):
                clinical.run(args)
                with self.assertRaises(ValueError):
                    clinical.run(args)
            output = nib.load(args.out_dir / "pair1" / "output.nii.gz")
            self.assertEqual(output.shape, shape)
            np.testing.assert_allclose(output.affine, affine)
            self.assertEqual(json.loads((args.out_dir / "metadata.json").read_text())["checkpoint_epoch"], 45)
            self.assertEqual(json.loads((args.out_dir / "report.json").read_text())["pairs_completed"], 1)
            self.assertTrue((args.out_dir / "pair1" / "review.png").is_file())
            self.assertEqual(len((args.out_dir / "per_pair.csv").read_text().splitlines()), 7)
            state = torch.load(checkpoint, weights_only=False)
            state["run_config"]["flip_horizontal"] = True
            torch.save(state, checkpoint)
            args.out_dir = root / "corrected"
            with patch.object(clinical, "discover_cases", return_value=[case]), \
                 patch.object(clinical, "clinical_tools", return_value=tools), \
                 patch("models.FrozenCXRBackbone", TinyBackbone), \
                 patch("models.build_change_head", return_value=TinyHead()), \
                 contextlib.redirect_stdout(io.StringIO()):
                clinical.run(args)
            corrected = nib.load(args.out_dir / "pair1/output.nii.gz")
            np.testing.assert_allclose(np.asarray(corrected.dataobj), np.asarray(output.dataobj), atol=1e-6)
            np.testing.assert_allclose(corrected.affine, affine)
            self.assertTrue(json.loads((args.out_dir / "metadata.json").read_text())["flip_horizontal"])


class UprightPipelineTests(unittest.TestCase):
    def test_reviewed_code_update_keeps_original_evidence_guards(self):
        from training import run_upright_clinical as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "training").mkdir()
            config_path = root / "config.yaml"
            config_path.write_text("flip_horizontal: true\n")
            report = root / "report.json"
            report.write_text("{}\n")
            manifest = root / "manifest_split.jsonl"
            manifest.write_text("{}\n")
            panel = root / "panel.png"
            panel.write_bytes(b"reviewed panel")
            names = ("training/train_foundation_diff.py", "training/run_upright_clinical.py")
            for name in names:
                (root / name).write_text("version = 1\n")
            provenance = dict(config_sha256=pipeline.sha256(config_path),
                              audit_sha256=pipeline.sha256(report),
                              manifest_sha256=pipeline.sha256(manifest),
                              code_sha256={name: pipeline.sha256(root / name) for name in names},
                              panels_sha256={"panel.png": pipeline.sha256(panel)})
            pipeline.write_json(root / "orientation_provenance.json", provenance)
            digest = pipeline.approval_hash(provenance)
            for name in names:
                (root / name).write_text("version = 2\n")
            review = dict(approved_preflight_sha256=digest, code_updates={
                name: dict(previous_sha256=provenance["code_sha256"][name],
                           reviewed_sha256=pipeline.sha256(root / name)) for name in names})
            review_path = root / "review.json"
            pipeline.write_json(review_path, review)
            review_hash = pipeline.sha256(review_path)
            with patch.object(pipeline, "ROOT", root), \
                    patch.object(pipeline, "load_config", return_value=dict(dataset_root=str(root))):
                with self.assertRaisesRegex(ValueError, "code changed"):
                    pipeline.validate_approval(config_path, root, digest)
                result = pipeline.validate_approval(config_path, root, digest, review_path, review_hash)
                self.assertEqual(result["reviewed_code_update"], review)
                self.assertEqual(json.loads((root / "orientation_provenance.json").read_text()), provenance)
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    pipeline.validate_approval(config_path, root, digest, review_path, "wrong")
                for path, payload, error in ((config_path, b"changed config", "configuration changed"),
                                              (report, b"changed report", "report changed"),
                                              (panel, b"changed panel", "overlay changed"),
                                              (manifest, b"changed data", "manifest changed"),
                                              (root / names[0], b"version = 3\n", "code changed")):
                    original = path.read_bytes()
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(ValueError, error):
                        pipeline.validate_approval(config_path, root, digest, review_path, review_hash)
                    path.write_bytes(original)
                for changes, error in ((dict(approved_preflight_sha256="wrong"), "different preflight"),
                                        (dict(code_updates={"datasets/pair_dataset.py": {}}), "only the trainer"),
                                        (dict(code_updates={names[0]: dict(previous_sha256="wrong",
                                                                          reviewed_sha256="new")}),
                                         "does not match audited code")):
                    pipeline.write_json(review_path, {**review, **changes})
                    with self.assertRaisesRegex(ValueError, error):
                        pipeline.validate_approval(config_path, root, digest, review_path,
                                                   pipeline.sha256(review_path))

    def test_failed_audit_publishes_evidence_without_training_or_approval(self):
        from training import run_upright_clinical as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            pipeline.write_json(destination / "report.json", dict(issues=["GT affine mismatch"]))
            arguments = ["pipeline", "--preflight-only", "--preflight-dir", str(destination)]
            with patch.object(sys, "argv", arguments), \
                 patch.object(pipeline, "preflight", side_effect=ValueError("GT affine mismatch")), \
                 patch.object(pipeline, "publish") as publish, \
                 patch.object(pipeline, "train_and_predict") as train:
                with self.assertRaisesRegex(ValueError, "GT affine mismatch"):
                    pipeline.main()
                publish.assert_called_once_with(destination)
                train.assert_not_called()
            self.assertFalse(json.loads((destination / "preflight_failure.json").read_text())["training_allowed"])
            self.assertFalse((destination / "orientation_provenance.json").exists())

    def test_orientation_only_recipe_and_approval_guards(self):
        from training import run_upright_clinical as pipeline

        config = pipeline.load_config(pipeline.DEFAULT_CONFIG)
        self.assertTrue(config["flip_horizontal"])
        self.assertEqual(config["rotation_k"], 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text("flip_horizontal: true\n")
            manifest = root / "manifest_split.jsonl"
            manifest.write_text("{}\n")
            report = root / "report.json"
            report.write_text("{}\n")
            code = root / "code.py"
            code.write_text("value = 1\n")
            panel = root / "panel.png"
            panel.write_bytes(b"reviewed panel")
            provenance = dict(config_sha256=pipeline.sha256(config_path),
                              audit_sha256=pipeline.sha256(report),
                              manifest_sha256=pipeline.sha256(manifest),
                              code_sha256={str(code): pipeline.sha256(code)},
                              panels_sha256={"panel.png": pipeline.sha256(panel)})
            pipeline.write_json(root / "orientation_provenance.json", provenance)
            digest = pipeline.approval_hash(provenance)
            with patch.object(pipeline, "load_config", return_value=dict(dataset_root=str(root))):
                self.assertEqual(pipeline.validate_approval(config_path, root, digest), provenance)
                with self.assertRaisesRegex(ValueError, "approval hash"):
                    pipeline.validate_approval(config_path, root, "wrong")
                panel.write_bytes(b"changed panel")
                with self.assertRaisesRegex(ValueError, "overlay changed"):
                    pipeline.validate_approval(config_path, root, digest)
                panel.write_bytes(b"reviewed panel")
                code.write_text("value = 2\n")
                with self.assertRaisesRegex(ValueError, "code changed"):
                    pipeline.validate_approval(config_path, root, digest)
                code.write_text("value = 1\n")
                manifest.write_text("changed dataset\n")
                with self.assertRaisesRegex(ValueError, "manifest changed"):
                    pipeline.validate_approval(config_path, root, digest)

    def test_all100_predictions_require_original_geometry_and_signed_masks(self):
        import nibabel as nib
        import numpy as np
        from training import run_upright_clinical as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "current.nii.gz"
            affine = np.diag([.7, .9, 1, 1])
            values = np.zeros((7, 11), dtype=np.float32)
            values[1, 3], values[5, 8] = .8, -.9
            mask = (values > .5).astype(np.int8) - (values < -.5).astype(np.int8)
            nib.save(nib.Nifti1Image(np.zeros_like(values), affine), current)
            pipeline.write_json(root / "metadata.json", dict(num_pairs=100, prediction_threshold=.5,
                                                            rotation_k=3, flip_horizontal=True))
            pipeline.write_json(root / "report.json", dict(pairs_completed=100))
            cases = []
            for number in range(1, 101):
                folder = root / f"pair{number}"
                folder.mkdir()
                nib.save(nib.Nifti1Image(values, affine), folder / "output.nii.gz")
                nib.save(nib.Nifti1Image(mask, affine), folder / "mask.nii.gz")
                cases.append(dict(pair_id=folder.name, current=str(current)))
            hashes = pipeline.verify_predictions(root, cases)
            self.assertEqual(len(hashes), 100)
            with self.assertRaisesRegex(ValueError, "exact clinical100"):
                pipeline.verify_predictions(root, cases[:-1])
            wrong = affine.copy()
            wrong[0, 3] = 10
            nib.save(nib.Nifti1Image(values, wrong), root / "pair100/output.nii.gz")
            with self.assertRaisesRegex(ValueError, "original image coordinates"):
                pipeline.verify_predictions(root, cases)
            nib.save(nib.Nifti1Image(values, affine), root / "pair100/output.nii.gz")
            nib.save(nib.Nifti1Image(-mask, affine), root / "pair100/mask.nii.gz")
            with self.assertRaises(AssertionError):
                pipeline.verify_predictions(root, cases)

    def test_publication_is_scoped_and_training_failure_prevents_prediction(self):
        from training import run_upright_clinical as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "study"
            destination = root / "results/approved_export"
            destination.mkdir(parents=True)
            (destination / "report.json").write_text("{}\n")
            (root / "unrelated.txt").write_text("do not stage")
            with patch.object(pipeline, "ROOT", root), \
                 patch.object(pipeline.subprocess, "check_output", return_value="main\n"), \
                 patch.object(pipeline.subprocess, "run") as execute:
                pipeline.publish(destination)
            commands = [call.args[0] for call in execute.call_args_list]
            self.assertEqual(commands[1], ["git", "add", "-f", "--", "study/results/approved_export/report.json"])
            self.assertEqual(commands[-1], ["git", "push", "origin", "HEAD:main"])
            self.assertNotIn("unrelated.txt", str(commands))
            config = dict(save_folder=str(root / "run/checkpoints"), dataset_root="data", cache_dir="cache")
            with patch.object(pipeline, "load_config", return_value=config), \
                 patch.object(pipeline, "validate_approval", return_value={}), \
                 patch.object(pipeline.subprocess, "check_output", return_value="commit\n"), \
                 patch.object(pipeline.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "train")), \
                 patch.object(pipeline.evaluate_clinical, "run") as predict, \
                 patch.object(pipeline, "publish") as publish:
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.train_and_predict(root / "config.yaml", destination, "approved", root)
                predict.assert_not_called()
                publish.assert_not_called()


class RotatedMaskRunTests(unittest.TestCase):
    def test_small_cutoff_config_and_signed_target_boundary(self):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        with patch.object(sys, "argv", ["train", "--config", str(config_root / "last1_rot90cw_mask_fp075.yaml")]):
            baseline = parse_args()
        with patch.object(sys, "argv", ["train", "--config", str(config_root / "last1_rot90cw_mask_thr0p0001_fp075.yaml")]):
            args = parse_args()
        allowed_changes = {"config", "save_folder", "plots_folder", "target_threshold",
                           "diagnostics_gt_thresholds", "diagnostics_components"}
        self.assertEqual({key: value for key, value in vars(args).items() if key not in allowed_changes},
                         {key: value for key, value in vars(baseline).items() if key not in allowed_changes})
        self.assertEqual(args.target_threshold, .0001)
        self.assertNotEqual(args.save_folder, baseline.save_folder)
        self.assertNotEqual(args.plots_folder, baseline.plots_folder)
        self.assertEqual((args.eval_threshold, args.eval_gt_threshold), (.5, .05))
        self.assertTrue(args.diagnostics_components)
        self.assertIn(.0001, args.diagnostics_gt_thresholds)
        gt = torch.tensor([[[[0., .000099, -.000099, .0001, -.0001, .000101, -.000101, .04]]]])
        target = torch.tensor([[[[0., 0., 0., 0., 0., 1., -1., 1.]]]])
        pred = torch.full_like(gt, .2, requires_grad=True)
        options = {key: getattr(args, key) for key in ("tau", "w_l1", "w_l2", "w_dice", "w_sign",
                   "pos_weight", "target_mode", "target_threshold", "background_weight")}
        loss, _ = change_map_loss(pred, gt, **options)
        weight = torch.where(target != 0, 11., .75)
        expected = (weight * ((pred - target).abs() + (pred - target).square())).mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(float(change_map_loss(target, gt, **options)[0]), 0.)
        loss.backward()
        self.assertGreater(float(pred.grad[..., 0]), 0.)
        self.assertLess(float(pred.grad[..., 5]), 0.)
        self.assertGreater(float(pred.grad[..., 6]), 0.)
        diagnostics = trainer.make_diagnostics(args)
        diagnostics.update(pred.detach(), gt, dict(pair_id=["boundary"], is_pathology=[1]))
        self.assertAlmostEqual(diagnostics.rows[0]["target_area"], 3 / 8)

    def test_training_and_validation_use_mask_targets_and_raw_gt_support(self):
        class PerfectMaskHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output = torch.nn.Parameter(torch.tensor([[[[1.0, -1.0, 0.0]]]]))

            def forward(self, *inputs):
                return self.output, None

        batch = dict(p_prior=torch.zeros(1, 1, 1), p_curr=torch.zeros(1, 1, 1),
                     cls_prior=torch.zeros(1, 1), cls_curr=torch.zeros(1, 1),
                     gt_diff=torch.tensor([[[[0.08, -0.12, 0.0]]]]), is_pathology=torch.ones(1))
        arguments = ["train", "--target_mode", "signed_mask", "--background_weight", "0.75",
                     "--w_dice", "0", "--w_sign", "0", "--w_l2", "1", "--accum", "1"]
        with patch.object(sys, "argv", arguments):
            args = parse_args()
        head = PerfectMaskHead()
        optimizer = torch.optim.SGD(head.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        metrics = trainer.train_one_epoch(head, None, [batch], optimizer, scheduler, "cpu", args)
        self.assertEqual(metrics["total"], 0.0)
        validation = trainer.evaluate(head, None, [batch], "cpu", threshold=0.5,
                                      gt_threshold=0.05, target_mode="signed_mask", background_weight=0.75)
        self.assertAlmostEqual(validation["dice"], 1.0)
        self.assertAlmostEqual(validation["sens_pos"], 1.0, places=5)
        self.assertAlmostEqual(validation["sens_neg"], 1.0, places=5)
        self.assertEqual(validation["l1_ch"], 0.0)

    def test_native_rotated_cache_metadata_preview_and_reuse_guard(self):
        from training import cache_features
        from datasets.pair_dataset import CachedPairDataset

        sample = dict(img_prior=torch.arange(16).float().reshape(1, 4, 4) / 15,
                      img_curr=torch.ones(1, 4, 4), gt_diff=torch.full((1, 4, 4), -0.08),
                      anomaly_type=1, direction=0, is_pathology=1,
                      change_group_id="case/pair0", pair_id="case/pair0/variant0", rotation_k=3)

        class TinyBackbone(torch.nn.Module):
            def forward(self, images):
                values = images.mean((-2, -1))
                return dict(patch_tokens=values[:, None].expand(-1, 4, -1), cls_token=values)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest_split.jsonl").write_text(json.dumps(dict(
                pair_id=sample["pair_id"], split="train", effective_anomaly_type="consolidation")) + "\n")
            cache = root / "cache"
            preview = root / "input_preview.png"
            arguments = ["cache", "--dataset_root", str(root), "--cache_dir", str(cache),
                         "--last_n_layers", "1", "--rotation_k", "3", "--device", "cpu",
                         "--num_workers", "0", "--preview_path", str(preview)]
            with patch.object(sys, "argv", arguments), \
                 patch.object(cache_features, "LongitudinalPairDataset", return_value=[sample]) as dataset, \
                 patch.object(cache_features, "_build_backbone", return_value=TinyBackbone()):
                self.assertEqual(cache_features.main(), 0)
                dataset.assert_called_with(str(root), split=None, img_size=512, rotation_k=3,
                                           flip_horizontal=False)
                self.assertEqual(cache_features.main(), 0)
            self.assertTrue(preview.is_file())
            metadata = json.loads((cache / "cache_config.json").read_text())
            self.assertEqual(metadata["rotation_k"], 3)
            cached = CachedPairDataset(str(root), str(cache), rotation_k=3)
            self.assertEqual(cached[0]["rotation_k"], 3)
            torch.testing.assert_close(cached[0]["gt_diff"], sample["gt_diff"], atol=1e-4, rtol=0)
            with self.assertRaisesRegex(ValueError, "rotation"):
                CachedPairDataset(str(root), str(cache), rotation_k=0)
            with self.assertRaisesRegex(ValueError, "horizontal flip"):
                CachedPairDataset(str(root), str(cache), rotation_k=3, flip_horizontal=True)
            with patch.object(sys, "argv", arguments + ["--flip_horizontal"]), \
                 self.assertRaisesRegex(ValueError, "configuration differs"):
                cache_features.main()
            corrected_arguments = list(arguments) + ["--flip_horizontal"]
            corrected_cache = root / "corrected_cache"
            corrected_arguments[corrected_arguments.index("--cache_dir") + 1] = str(corrected_cache)
            corrected_sample = dict(sample, flip_horizontal=True)
            with patch.object(sys, "argv", corrected_arguments), \
                 patch.object(cache_features, "LongitudinalPairDataset", return_value=[corrected_sample]), \
                 patch.object(cache_features, "_build_backbone", return_value=TinyBackbone()):
                self.assertEqual(cache_features.main(), 0)
            corrected = CachedPairDataset(str(root), str(corrected_cache), rotation_k=3, flip_horizontal=True)
            self.assertTrue(corrected[0]["flip_horizontal"])
            self.assertTrue(json.loads((corrected_cache / "cache_config.json").read_text())["flip_horizontal"])
            cache_path = Path(corrected.paths[0])
            payload = torch.load(cache_path, weights_only=False)
            payload["flip_horizontal"] = False
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ValueError, "horizontal flip mismatch"):
                corrected[0]
            arguments[arguments.index("--rotation_k") + 1] = "0"
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(ValueError, "configuration differs"):
                cache_features.main()

    def test_signed_mask_loss_and_modest_background_discount(self):
        gt = torch.tensor([[[[0.08, -0.12, 0.0, 0.04]]]])
        perfect = torch.tensor([[[[1.0, -1.0, 0.0, 0.0]]]])
        options = dict(w_l1=1, w_l2=1, w_dice=0, w_sign=0,
                       target_mode="signed_mask", target_threshold=0.05)
        self.assertEqual(float(change_map_loss(perfect, gt, **options)[0]), 0.0)
        wrong = perfect.clone()
        wrong[..., 0] = -1
        self.assertGreater(float(change_map_loss(wrong, gt, **options)[0]), 0.0)
        false_positive = perfect.clone()
        false_positive[..., 2] = 0.5
        original = change_map_loss(false_positive, gt, **options)[0]
        discounted = change_map_loss(false_positive, gt, background_weight=0.75, **options)[0]
        torch.testing.assert_close(discounted, 0.75 * original)
        missed = torch.zeros_like(gt)
        torch.testing.assert_close(change_map_loss(missed, gt, **options)[0],
                                   change_map_loss(missed, gt, background_weight=0.75, **options)[0])
        nuisance = change_map_loss(torch.full_like(gt, 0.5), torch.zeros_like(gt),
                                   background_weight=0.75, **options)[0]
        self.assertGreater(float(nuisance), 0.0)
        self.assertTrue(torch.isfinite(nuisance))

    def test_distinct_prediction_and_gt_thresholds(self):
        from evaluation.change_detection_metrics import directional_sensitivity

        pred = torch.tensor([[[[0.9, -0.8, 0.0]]]])
        gt = torch.tensor([[[[0.08, -0.12, 0.0]]]])
        metrics = directional_sensitivity(pred, gt, 0.5, 0.05)
        self.assertAlmostEqual(metrics["sensitivity_positive"], 1.0, places=5)
        self.assertAlmostEqual(metrics["sensitivity_negative"], 1.0, places=5)

    def test_dataset_rotates_images_and_signed_gt_together(self):
        import numpy as np
        import nibabel as nib
        from datasets.pair_dataset import LongitudinalPairDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "case/pair0/variant0"
            folder.mkdir(parents=True)
            prior = np.arange(16, dtype=np.float32).reshape(4, 4)
            gt = np.zeros((4, 4), dtype=np.float32)
            gt[0, 1], gt[2, 3] = 0.2, -0.1
            for name, array in (("prior", prior), ("current", prior + gt), ("diff_map", gt)):
                nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f"{name}.nii.gz")
            record = dict(pair_id="case/pair0/variant0", split="train", realized_change=True,
                          effective_anomaly_type="consolidation")
            (root / "manifest_split.jsonl").write_text(json.dumps(record) + "\n")
            original = LongitudinalPairDataset(str(root), img_size=4)[0]
            for rotation in (1, 2, 3):
                rotated = LongitudinalPairDataset(str(root), img_size=4, rotation_k=rotation)[0]
                for key in ("img_prior", "img_curr", "gt_diff"):
                    torch.testing.assert_close(rotated[key], torch.rot90(original[key], rotation, (-2, -1)))
                self.assertEqual(rotated["rotation_k"], rotation)
            corrected = LongitudinalPairDataset(str(root), img_size=4, rotation_k=3,
                                                flip_horizontal=True)[0]
            from evaluation.evaluate_clinical import prepare_image
            for key in ("img_prior", "img_curr", "gt_diff"):
                torch.testing.assert_close(corrected[key], original[key].transpose(-2, -1))
            torch.testing.assert_close(corrected["img_curr"], prepare_image(prior + gt, 4, 3, True))
            self.assertTrue(corrected["flip_horizontal"])
            with self.assertRaises(ValueError):
                LongitudinalPairDataset(str(root), rotation_k=4)


class ReviewSelectionTests(unittest.TestCase):
    def test_review_export_end_to_end(self):
        import numpy as np
        import nibabel as nib
        from PIL import Image
        from evaluation import evaluate_rq1 as evaluator

        spec = {"type": "dpt_cross_attention", "kwargs": dict(
            backbone_dim=16, patch_dim=16, cls_dim=16, grid=5,
            channels=8, out_size=32, embed_dim=12, num_layers=1)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for index, value in enumerate((0.2, -0.2, 0.0)):
                pair_id = f"case/pair{index}/variant0"
                folder = root / pair_id
                folder.mkdir(parents=True)
                prior = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
                gt = np.zeros((32, 32), dtype=np.float32)
                gt[8:20, 8:20] = value
                for name, array in (("prior", prior), ("current", prior + gt), ("diff_map", gt)):
                    nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f"{name}.nii.gz")
                records.append(dict(pair_id=pair_id, split="val", realized_change=bool(value),
                                    effective_anomaly_type="consolidation" if value else "none",
                                    change_group_id=str(index)))
                cache = root / "cache" / pair_id
                cache.mkdir(parents=True)
                torch.save(dict(p_prior=torch.randn(25, 16), p_curr=torch.randn(25, 16),
                                cls_prior=torch.randn(16), cls_curr=torch.randn(16),
                                gt_diff=torch.from_numpy(gt)[None], anomaly_type=1 if value else 0,
                                direction=0, is_pathology=int(bool(value)),
                                change_group_id=str(index), pair_id=pair_id), cache / "feat.pt")
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8")
            checkpoint = root / "best.pt"
            torch.save(dict(head=build_change_head(spec).state_dict(), head_spec=spec,
                            epoch=55), checkpoint)
            output = root / "review"
            arguments = ["eval", "--dataset_root", str(root), "--cache_dir", str(root / "cache"),
                         "--ckpt", str(checkpoint), "--split", "val", "--threshold", "0.05",
                         "--device", "cpu", "--batch_size", "2", "--num_workers", "0",
                         "--out_dir", str(output), "--review"]
            with patch.object(sys, "argv", arguments):
                evaluator.main()
            cases = json.loads((output / "review_cases.json").read_text())
            metadata = json.loads((output / "review_metadata.json").read_text())
            self.assertEqual(len(cases), 3)
            self.assertEqual(metadata["evaluated_pairs"], 3)
            self.assertEqual(metadata["checkpoint_epoch"], 55)
            self.assertEqual(metadata["checkpoint_sha256"], evaluator.file_sha256(checkpoint))
            self.assertTrue((output / "index.md").is_file())
            self.assertEqual(len((output / "per_pair_val.csv").read_text().splitlines()), 4)
            self.assertEqual(len(list(output.glob("contact_*.png"))), 3)
            for row in cases:
                with Image.open(output / row["panel"]) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertTrue(any(low != high for low, high in image.convert("RGB").getextrema()))
            for record in records:
                path = root / "cache" / record["pair_id"] / "feat.pt"
                cached = torch.load(path, weights_only=False)
                cached["rotation_k"] = 3
                cached["gt_diff"] = torch.rot90(cached["gt_diff"], 3, (-2, -1))
                torch.save(cached, path)
            run_config = dict(rotation_k=3, target_mode="signed_mask", eval_threshold=0.5,
                              eval_gt_threshold=0.05)
            torch.save(dict(head=build_change_head(spec).state_dict(), head_spec=spec,
                            epoch=1, run_config=run_config), checkpoint)
            rotated_arguments = list(arguments)
            threshold_index = rotated_arguments.index("--threshold")
            del rotated_arguments[threshold_index:threshold_index + 2]
            rotated_output = root / "rotated_review"
            rotated_arguments[rotated_arguments.index("--out_dir") + 1] = str(rotated_output)
            with patch.object(sys, "argv", rotated_arguments):
                evaluator.main()
            rotated_metadata = json.loads((rotated_output / "review_metadata.json").read_text())
            self.assertEqual(rotated_metadata["rotation_k"], 3)
            self.assertEqual(rotated_metadata["threshold"], 0.5)
            self.assertEqual(rotated_metadata["gt_threshold"], 0.05)
            self.assertEqual(rotated_metadata["target_mode"], "signed_mask")
            rotated_cases = json.loads((rotated_output / "review_cases.json").read_text())
            self.assertEqual({row["direction"] for row in rotated_cases if row["is_pathology"]},
                             {"positive", "negative"})
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(ValueError, "empty"):
                evaluator.main()
            rows = [dict(row) for row in cases]
            rows.pop()
            dataset = evaluator.CachedPairDataset(str(root), str(root / "cache"), split="val")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                evaluator.export_review(build_change_head(spec), None, dataset, rows, [],
                                        str(root), "val", output, "cpu", 0.05, img_size=32)
            (root / "cache" / records[0]["pair_id"] / "feat.pt").unlink()
            arguments[arguments.index("--out_dir") + 1] = str(root / "new_review")
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(FileNotFoundError, "complete cache"):
                evaluator.main()

    def test_stratified_selection_is_reproducible_and_diverse(self):
        rows = []
        for anomaly in ("consolidation", "pleural_effusion"):
            for direction in ("positive", "negative", "mixed"):
                for index in range(40):
                    rows.append(dict(pair_id=f"{anomaly}_{direction}_{index:02}",
                                     change_group_id=f"{anomaly}_{direction}_{index // 2}",
                                     anomaly_type=anomaly, direction=direction,
                                     is_pathology=1, dice=index / 40, nuisance_fp=None))
        for index in range(60):
            rows.append(dict(pair_id=f"nuisance_{index:02}", change_group_id="",
                             anomaly_type="none", direction="below_threshold",
                             is_pathology=0, dice=None, nuisance_fp=index / 60))
        chosen = select_review_cases(rows, per_group=12, nuisance_count=20)
        self.assertEqual(chosen, select_review_cases(list(reversed(rows)), 12, 20))
        self.assertEqual(len(chosen), 6 * 12 + 20)
        self.assertEqual(len({row["pair_id"] for row in chosen}), len(chosen))
        self.assertEqual({row["selection"] for row in chosen},
                         {"lowest_dice", "middle_dice", "highest_dice", "seeded_random", "highest_fp"})
        self.assertIn("nuisance_59", {row["pair_id"] for row in chosen})
        for group in {row["review_group"] for row in chosen} - {"nuisance"}:
            identities = [row["change_group_id"] for row in chosen if row["review_group"] == group]
            self.assertEqual(len(identities), len(set(identities)))

    def test_small_buckets_and_invalid_counts(self):
        row = dict(pair_id="one", anomaly_type="consolidation", direction="positive",
                   is_pathology=1, dice=0.5, nuisance_fp=None)
        self.assertEqual(len(select_review_cases([row])), 1)
        for count in (1, 2, 3):
            rows = [dict(row, pair_id=str(index)) for index in range(8)]
            self.assertEqual(len(select_review_cases(rows, per_group=count)), count)
        with self.assertRaises(ValueError):
            select_review_cases([row], per_group=0)

    def test_review_metrics_include_nuisance_and_direction(self):
        class FixedHead(torch.nn.Module):
            def forward(self, *inputs):
                return torch.tensor([[[[0.2, -0.2]]], [[[0.2, 0.0]]]]), None

        batch = dict(p_prior=torch.zeros(2, 1, 1), p_curr=torch.zeros(2, 1, 1),
                     cls_prior=torch.zeros(2, 1), cls_curr=torch.zeros(2, 1),
                     gt_diff=torch.tensor([[[[0.2, -0.2]]], [[[0.0, 0.0]]]]),
                     anomaly_type=torch.tensor([1, 0]), is_pathology=torch.tensor([1, 0]),
                     pair_id=["pathology", "control"], change_group_id=["a", "b"])
        rows = []
        report, _ = run_eval(FixedHead(), None, [batch], "cpu", 0.05, review_rows=rows)
        self.assertEqual(rows[0]["direction"], "mixed")
        self.assertEqual(rows[0]["dice"], 1.0)
        self.assertEqual(rows[1]["nuisance_fp"], 0.5)
        self.assertIsNone(rows[1]["dice"])
        self.assertEqual(report["overall"]["n"], 1)


class DPTHeadTests(unittest.TestCase):
    head_type = "dpt"
    config_path = "configs/last4_dpt.yaml"
    num_layers = 4

    def setUp(self):
        torch.manual_seed(0)
        self.spec = {"type": self.head_type, "kwargs": dict(
            backbone_dim=16, patch_dim=16 * self.num_layers, cls_dim=16, grid=5,
            channels=8, out_size=32, embed_dim=12)}
        if self.num_layers == 1:
            self.spec["kwargs"]["num_layers"] = 1
        self.prior = torch.randn(2, 25, 16 * self.num_layers)
        self.current = torch.randn_like(self.prior)
        self.cls_prior = torch.randn(2, 16)
        self.cls_current = torch.randn(2, 16)

    def test_forward_loss_backward_and_reload(self):
        head = build_change_head(self.spec)
        inputs = (self.prior, self.current, self.cls_prior, self.cls_current)
        prediction, embedding = head(*inputs)
        self.assertEqual(prediction.shape, (2, 1, 32, 32))
        self.assertEqual(embedding.shape, (2, 12))
        self.assertLessEqual(float(prediction.detach().abs().max()), 1.0)
        loss, _ = change_map_loss(prediction, torch.zeros_like(prediction),
                                  w_l1=1, w_l2=1, w_dice=0, w_sign=0)
        loss.backward()
        for project in head.projections:
            self.assertGreater(float(project.weight.grad.abs().sum()), 0)
        for parameter in head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer = torch.optim.AdamW(head.parameters())
        optimizer.step()
        buffer = io.BytesIO()
        torch.save({"head": head.state_dict(), "head_spec": self.spec,
                    "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_change_head(checkpoint["head_spec"])
        restored.load_state_dict(checkpoint["head"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.testing.assert_close(restored(*inputs)[0], head(*inputs)[0])

    def test_reject_wrong_cache(self):
        with self.assertRaises(ValueError):
            DPTDifferenceHead(patch_dim=768)
        head = build_change_head(self.spec)
        with self.assertRaises(ValueError):
            head(self.prior[..., :-1], self.current[..., :-1])
        with self.assertRaises(ValueError):
            head(self.prior[:, :-1], self.current[:, :-1])

    def test_default_head_and_config_override(self):
        self.assertIsInstance(build_change_head(head_spec(768)), DifferenceHead)
        with patch.object(sys, "argv", ["train", "--config", self.config_path,
                                        "--batch_size", "2"]):
            args = parse_args()
        self.assertEqual((args.head, args.feat_last_k, args.batch_size),
                 (self.head_type, self.num_layers, 2))
        spec = head_spec(768 * self.num_layers, args.head)
        self.assertEqual(spec["type"], self.head_type)
        self.assertEqual(spec["kwargs"].get("num_layers", 4), self.num_layers)
        self.assertEqual((args.w_l1, args.w_l2, args.w_dice, args.w_sign), (1, 1, 0, 0))

    def test_evaluator_and_feature_slicing(self):
        batch = dict(p_prior=self.prior, p_curr=self.current,
                     cls_prior=self.cls_prior, cls_curr=self.cls_current,
                     gt_diff=torch.zeros(2, 1, 32, 32),
                     anomaly_type=torch.tensor([1, 0]), is_pathology=torch.tensor([1, 0]))
        batch["gt_diff"][0, 0, 8:20, 8:20] = 0.2
        sliced = _tokens(batch, None, "cpu", 16)
        torch.testing.assert_close(sliced[0], self.prior[..., -16:])
        report, _ = run_eval(build_change_head(self.spec), None, [batch], "cpu", 0.05,
                     self.spec["kwargs"]["patch_dim"])
        self.assertEqual(report["overall"]["n"], 1)
        self.assertEqual(report["nuisance_false_positive_area"]["n"], 1)

    def test_cached_training_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split in ("train", "val"):
                for index in range(2):
                    pair_id = f"{split}_{index}"
                    records.append(dict(pair_id=pair_id, split=split,
                                        effective_anomaly_type="none"))
                    folder = root / "cache" / pair_id
                    folder.mkdir(parents=True)
                    torch.save(dict(p_prior=self.prior[index], p_curr=self.current[index],
                                    cls_prior=self.cls_prior[index], cls_curr=self.cls_current[index],
                                    gt_diff=torch.ones(1, 32, 32) * 0.2,
                                    anomaly_type=1, direction=1, is_pathology=1,
                                    change_group_id=pair_id, pair_id=pair_id), folder / "feat.pt")
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8")
            arguments = ["train", "--config", self.config_path,
                         "--dataset_root", str(root), "--cache_dir", str(root / "cache"),
                         "--save_folder", str(root / "checkpoints"),
                         "--plots_folder", str(root / "plots"), "--batch_size", "2",
                         "--accum", "1", "--num_workers", "0", "--device", "cpu",
                         "--dpt_channels", "8"]
            training_output = io.StringIO()
            with (
                patch.multiple(trainer.C, BACKBONE_DIM=16, FEATURE_GRID=5,
                               IMG_SIZE=32, EMBED_DIM=12),
                patch.object(trainer, "plot_curves"),
                patch.object(trainer, "save_sample_plots"),
                patch("sys.stdout", training_output),
            ):
                with patch.object(sys, "argv", arguments + ["--epochs", "1"]):
                    trainer.main()
                first = torch.load(root / "checkpoints" / "last.pt", weights_only=True)
                self.assertEqual(first["best_dice"], first["val"]["dice"])
                with patch.object(sys, "argv", arguments + ["--epochs", "2", "--resume"]):
                    trainer.main()
            output = training_output.getvalue()
            self.assertIn("[train online sampled] loss_batch_mean total=", output)
            self.assertIn("[val fixed head] loss_batch_mean total=", output)
            self.assertIn("[val checkpoint metric] change_only_support_dice=", output)
            self.assertIn("[train online sampled] pred_abs>", output)
            self.assertIn("[val fixed head] pred_abs>", output)
            checkpoint = torch.load(root / "checkpoints" / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["best_dice"], max(first["best_dice"], checkpoint["val"]["dice"]))
            self.assertEqual(checkpoint["head_spec"]["type"], self.head_type)
            self.assertEqual(len((root / "checkpoints" / "metrics.csv").read_text().splitlines()), 3)
            for epoch in (1, 2):
                for split in ("train", "val"):
                    report = root / "checkpoints" / "diagnostics" / f"epoch_{epoch:04d}_{split}.json"
                    diagnostics = json.loads(report.read_text())
                    self.assertEqual(diagnostics["groups"]["overall"]["pairs"], 2)
                    self.assertEqual(diagnostics["metadata"]["training_health"]["optimizer_steps"], 1)
                    self.assertIn("total", diagnostics["metadata"]["loss_batch_means"][split])


class DPTCrossAttentionTests(DPTHeadTests):
    head_type = "dpt_cross_attention"
    config_path = "configs/last4_dpt_cross_attention.yaml"

    def test_temporal_attention_gradients_and_order(self):
        head = build_change_head(self.spec)
        prediction, _ = head(self.prior, self.current)
        prediction.square().mean().backward()
        for blocks in head.temporal_blocks:
            for block in blocks:
                gradient = block.attention.in_proj_weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
        with torch.no_grad():
            forward = head._project_streams(self.prior, self.current)
            reverse = head._project_streams(self.current, self.prior)
            identical = head._project_streams(self.prior, self.prior)
            offset = torch.randn_like(self.prior)
            shifted = head._project_streams(self.prior + offset, self.current + offset)
        for actual, swapped, unchanged in zip(forward, reverse, identical):
            torch.testing.assert_close(actual, -swapped)
            torch.testing.assert_close(unchanged, torch.zeros_like(unchanged))
        self.assertTrue(any(not torch.allclose(actual, shifted_actual)
                            for actual, shifted_actual in zip(forward, shifted)))

    def test_reject_invalid_attention_settings(self):
        for settings in ({"num_heads": 3}, {"num_heads": 0}, {"attention_depth": 0}):
            with self.assertRaises(ValueError):
                DPTCrossAttentionHead(**self.spec["kwargs"], **settings)


class SingleLayerCrossAttentionTests(DPTCrossAttentionTests):
    config_path = "configs/last1_dpt_cross_attention.yaml"
    num_layers = 1


class CacheSliceTests(unittest.TestCase):
    def test_online_launcher_preflight_without_cache_or_test_images(self):
        import nibabel as nib
        import numpy as np

        script = Path(__file__).with_name("run_last4_dpt.sh").read_text()
        preflight = script.split("python -u - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(pair_id=split, split=split, patient_key=split)
                       for split in ("train", "val", "test")]
            (root / "manifest_split.jsonl").write_text("\n".join(map(json.dumps, records)))
            for split in ("train", "val"):
                folder = root / split
                folder.mkdir()
                for name in ("prior", "current", "diff_map"):
                    nib.save(nib.Nifti1Image(np.zeros((8, 8), dtype=np.float32), np.eye(4)),
                             folder / f"{name}.nii.gz")
            with patch.dict(os.environ, DATA=str(root), CACHE_DIR=str(root / "absent"),
                            CACHE_LAYERS="1", NO_CACHE="1", ROTATION_K="3", FLIP_HORIZONTAL="1"), \
                 patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch.cuda, "mem_get_info", return_value=(12_000_000_000, 24_000_000_000)):
                exec(compile(preflight, "online_preflight", "exec"), {})
                (root / "val/current.nii.gz").unlink()
                with self.assertRaisesRegex(SystemExit, "Missing online input"):
                    exec(compile(preflight, "online_preflight", "exec"), {})

    def test_layer_aware_launcher_preflight(self):
        script = Path(__file__).with_name("run_last4_dpt.sh").read_text()
        preflight = script.split("python -u - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(pair_id=split, split=split, patient_key=split)
                       for split in ("train", "val", "test")]
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8")
            for layers in (1, 4):
                cache = root / str(layers)
                for record in records:
                    folder = cache / record["pair_id"]
                    folder.mkdir(parents=True)
                    torch.save(dict(p_prior=torch.zeros(1369, 768 * layers).half(),
                                    p_curr=torch.zeros(1369, 768 * layers).half(),
                                    cls_prior=torch.zeros(768).half(), cls_curr=torch.zeros(768).half(),
                                    gt_diff=torch.zeros(1, 512, 512).half()), folder / "feat.pt")
                with patch.dict(os.environ, DATA=str(root), CACHE_DIR=str(cache), CACHE_LAYERS=str(layers)), \
                        patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "mem_get_info", return_value=(12_000_000_000, 24_000_000_000)):
                    exec(compile(preflight, "launcher_preflight", "exec"), {})
                    if layers == 4:
                        with patch.dict(os.environ, CACHE_LAYERS="1"), self.assertRaises(SystemExit):
                            exec(compile(preflight, "launcher_preflight", "exec"), {})

    def test_compact_exact_resumable_slice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            (source / "pair").mkdir(parents=True)
            (root / "manifest_split.jsonl").write_text(
                json.dumps(dict(pair_id="pair", split="train")), encoding="utf-8")
            cached = dict(p_prior=torch.randn(25, 64).half(), p_curr=torch.randn(25, 64).half(),
                          cls_prior=torch.randn(16).half(), cls_curr=torch.randn(16).half(),
                          gt_diff=torch.randn(1, 32, 32).half(), pair_id="pair",
                          anomaly_type=1, direction=1, is_pathology=1, change_group_id="group")
            source_path = source / "pair" / "feat.pt"
            torch.save(cached, source_path)
            before = source_path.read_bytes()
            with patch.multiple(trainer.C, NUM_PATCH_TOKENS=25, BACKBONE_DIM=16):
                self.assertEqual(slice_last_layer_cache(root, source, target), {"written": 1, "skipped": 0})
                self.assertEqual(slice_last_layer_cache(root, source, target), {"written": 0, "skipped": 1})
                result = torch.load(target / "pair" / "feat.pt", weights_only=True)
                for key in ("p_prior", "p_curr"):
                    torch.testing.assert_close(result[key], cached[key][..., -16:])
                    self.assertEqual(result[key].untyped_storage().nbytes(), result[key].numel() * 2)
                torch.testing.assert_close(result["gt_diff"], cached["gt_diff"])
                self.assertEqual(source_path.read_bytes(), before)
                with self.assertRaises(ValueError):
                    slice_last_layer_cache(root, source, source)
                result["p_prior"].zero_()
                torch.save(result, target / "pair" / "feat.pt")
                with self.assertRaises(ValueError):
                    slice_last_layer_cache(root, source, target)
                source_path.unlink()
                with self.assertRaises(FileNotFoundError):
                    slice_last_layer_cache(root, source, target)


if __name__ == "__main__":
    unittest.main()