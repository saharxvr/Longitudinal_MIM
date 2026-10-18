"""
RQ1 evaluation — load a trained D-GLoRI head and report change-detection metrics.

Answers the RQ1 claims quantitatively on a held-out split (default: test):
    * Dice / IoU on the signed change map, broken down PER ANOMALY TYPE.
    * Directional sensitivity (sens+ = appeared, sens- = resolved).
    * Nuisance false-positive area: on nuisance-only pairs (is_pathology == 0, GT ~ 0),
      the fraction of pixels the model wrongly flags as change -> tests the
      "ignores devices / projection angle" claim (lower is better).

Writes a JSON report + a per-type bar chart + a qualitative GT-vs-pred panel.

PAI vs. radiologists is deferred to the real annotated ICU/PNIMIT pairs (no radiologist
GT exists for the synthetic set); see pairwise_agreement_index in change_detection_metrics.

Run:
    python -m evaluation.evaluate_rq1 --dataset_root /path/to/fcd_train \
        --cache_dir ./feature_cache --ckpt ./checkpoints/best.pt --split test
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
import textwrap

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from models import DifferenceHead, build_change_head
from datasets import CachedPairDataset, LongitudinalPairDataset
from evaluation.change_detection_metrics import dice_score, iou_score, directional_sensitivity


def build_head(device):
    return DifferenceHead(
        backbone_dim=C.BACKBONE_DIM, patch_dim=C.BACKBONE_DIM * C.LAST_N_LAYERS, cls_dim=C.BACKBONE_DIM,
        d_glori=C.D_GLORI,
        num_change_queries=C.NUM_CHANGE_QUERIES, num_heads=C.GLORI_NUM_HEADS,
        grid=C.FEATURE_GRID, embed_dim=C.EMBED_DIM, fusion_mode=C.FUSION_MODE,
        use_adaptive_temperature=C.USE_ADAPTIVE_TEMPERATURE,
        use_pyramid_patch_merging=C.USE_PYRAMID_PATCH_MERGING,
        integrate_global_cls=C.INTEGRATE_GLOBAL_CLS,
        out_size=C.IMG_SIZE, out_range=C.DECODER_OUT_RANGE,
    ).to(device)


def _tokens(batch, backbone, device, feat_dim=None):
    if backbone is None:
        prior, current = batch["p_prior"].to(device), batch["p_curr"].to(device)
        if feat_dim is not None:
            prior, current = prior[..., -feat_dim:], current[..., -feat_dim:]
        return (prior, current,
                batch["cls_prior"].to(device), batch["cls_curr"].to(device))
    fp = backbone(batch["img_prior"].to(device)); fc = backbone(batch["img_curr"].to(device))
    return fp["patch_tokens"], fc["patch_tokens"], fp["cls_token"], fc["cls_token"]


def _mean(x):
    return float(sum(x) / max(1, len(x)))


@torch.no_grad()
def run_eval(head, backbone, loader, device, thr, feat_dim=None, review_rows=None, gt_thr=None):
    head.eval()
    gt_thr = thr if gt_thr is None else gt_thr
    per_type = defaultdict(lambda: {"dice": [], "iou": [], "sens_pos": [], "sens_neg": []})
    nuis_fp = []                        # predicted-change area on nuisance-only pairs
    examples = {}                       # one (gt, pred) per anomaly type for the panel

    for batch in loader:
        p_prior, p_curr, cls_prior, cls_curr = _tokens(batch, backbone, device, feat_dim)
        gt = batch["gt_diff"].to(device)
        pred, _ = head(p_prior, p_curr, cls_prior, cls_curr)

        atype = batch["anomaly_type"]
        is_path = batch["is_pathology"]
        pm = (pred.abs() > thr).float()
        gm = (gt.abs() > gt_thr).float()

        for b in range(pred.shape[0]):
            a = int(atype[b]); path = int(is_path[b])
            name = C.ANOMALY_TYPES[a] if a < len(C.ANOMALY_TYPES) else str(a)
            if review_rows is not None:
                positive = bool((gt[b] > gt_thr).any())
                negative = bool((gt[b] < -gt_thr).any())
                direction = ("mixed" if positive and negative else "positive" if positive
                             else "negative" if negative else "below_threshold")
                directional = directional_sensitivity(pred[b:b + 1], gt[b:b + 1], thr, gt_thr)
                mask = gm[b].bool()
                review_rows.append(dict(
                    pair_id=batch["pair_id"][b],
                    change_group_id=batch.get("change_group_id", [""] * len(pred))[b],
                    anomaly_type=name, is_pathology=path, direction=direction,
                    dice=dice_score(pm[b], gm[b]) if path else None,
                    iou=iou_score(pm[b], gm[b]) if path else None,
                    sens_pos=directional["sensitivity_positive"] if positive else None,
                    sens_neg=directional["sensitivity_negative"] if negative else None,
                    gt_area=float(gm[b].mean()), pred_area=float(pm[b].mean()),
                    nuisance_fp=float(pm[b].mean()) if not path else None,
                    l1=float((pred[b] - gt[b]).abs().mean()),
                    wrong_sign_fraction=(float(((pred[b] * gt[b]) < 0)[mask].float().mean())
                                         if mask.any() else None),
                    gt_peak=float(gt[b].abs().max()), pred_peak=float(pred[b].abs().max()),
                ))
            if path == 0:
                nuis_fp.append(float(pm[b].mean()))       # any flagged pixel is a false positive
                continue
            per_type[name]["dice"].append(dice_score(pm[b], gm[b]))
            per_type[name]["iou"].append(iou_score(pm[b], gm[b]))
            d = directional_sensitivity(pred[b:b + 1], gt[b:b + 1], thr, gt_thr)
            # Only count a direction's recall on pairs that contain that direction.
            if (gt[b] > gt_thr).any():
                per_type[name]["sens_pos"].append(d["sensitivity_positive"])
            if (gt[b] < -gt_thr).any():
                per_type[name]["sens_neg"].append(d["sensitivity_negative"])
            if name not in examples:
                examples[name] = (gt[b, 0].cpu(), pred[b, 0].cpu())

    summary = {}
    all_d, all_i = [], []
    for name, m in per_type.items():
        summary[name] = {"n": len(m["dice"]), "dice": _mean(m["dice"]), "iou": _mean(m["iou"]),
                         "sens_pos": _mean(m["sens_pos"]), "sens_neg": _mean(m["sens_neg"])}
        all_d += m["dice"]; all_i += m["iou"]
    report = {
        "overall": {"n": len(all_d), "dice": _mean(all_d), "iou": _mean(all_i)},
        "per_anomaly_type": summary,
        "nuisance_false_positive_area": {"n": len(nuis_fp), "mean_fp_area": _mean(nuis_fp),
                        "p95_fp_area": float(torch.quantile(torch.tensor(nuis_fp), 0.95)) if nuis_fp else 0.0,
                        "max_fp_area": max(nuis_fp, default=0.0)},
        "threshold": thr,
        "gt_threshold": gt_thr,
    }
    return report, examples


def select_review_cases(rows, per_group=24, nuisance_count=48, seed=42):
    if per_group < 1 or nuisance_count < 0:
        raise ValueError("Review counts require per_group >= 1 and nuisance_count >= 0")
    generator = random.Random(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["pair_id"]):
        group = (f"{row['anomaly_type']}/{row['direction']}" if row["is_pathology"]
                 else "nuisance")
        groups[group].append(row)
    selected = []
    for group, candidates in sorted(groups.items()):
        nuisance = group == "nuisance"
        budget = nuisance_count if nuisance else per_group
        ordered = sorted(candidates, key=lambda item: (
            -item["nuisance_fp"] if nuisance else item["dice"], item["pair_id"]))
        used_pairs, used_changes = set(), set()
        chosen = []

        def take(pool, count, reason):
            if count <= 0:
                return
            added = 0
            for row in pool:
                identity = row.get("change_group_id") or row["pair_id"]
                if row["pair_id"] in used_pairs or identity in used_changes:
                    continue
                chosen.append(dict(row, review_group=group, selection=reason))
                used_pairs.add(row["pair_id"])
                used_changes.add(identity)
                added += 1
                if added >= count:
                    break

        if not budget:
            continue
        quota = max(1, budget // (2 if nuisance else 4))
        take(ordered, min(quota, budget), "highest_fp" if nuisance else "lowest_dice")
        if not nuisance:
            middle = sorted(ordered, key=lambda row: (
                abs(row["dice"] - ordered[len(ordered) // 2]["dice"]), row["pair_id"]))
            take(middle, min(quota, budget - len(chosen)), "middle_dice")
            take(reversed(ordered), min(quota, budget - len(chosen)), "highest_dice")
        shuffled = list(candidates)
        generator.shuffle(shuffled)
        if len(chosen) < budget:
            take(shuffled, budget - len(chosen), "seeded_random")
        selected.extend(chosen)
    return selected


@torch.no_grad()
def export_review(head, backbone, dataset, rows, selected, dataset_root, split,
                  out_dir, device, threshold, feat_dim=None, batch_size=8, img_size=512,
                  gt_threshold=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    gt_threshold = threshold if gt_threshold is None else gt_threshold
    output = Path(out_dir)
    panel_dir = output / "cases"
    panel_dir.mkdir(parents=True, exist_ok=True)
    raw = LongitudinalPairDataset(dataset_root, split=split, img_size=img_size,
                                  rotation_k=getattr(dataset, "rotation_k", 0),
                                  flip_horizontal=getattr(dataset, "flip_horizontal", False))
    raw_indices = {record["pair_id"]: index for index, record in enumerate(raw.records)}
    cached_indices = {row["pair_id"]: index for index, row in enumerate(rows)}
    if len(cached_indices) != len(dataset) or len(raw_indices) != len(raw):
        raise ValueError("Duplicate pair IDs or evaluation dataset/row mismatch")
    missing = set(raw_indices) - set(cached_indices)
    if missing:
        raise ValueError(f"Review requires complete split coverage; missing {len(missing)} pairs")
    selected_by_id = {row["pair_id"]: row for row in selected}
    loader = DataLoader(Subset(dataset, [cached_indices[row["pair_id"]] for row in selected]),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    head.eval()
    for batch in loader:
        tokens = _tokens(batch, backbone, device, feat_dim)
        predictions, _ = head(*tokens)
        for index, pair_id in enumerate(batch["pair_id"]):
            row = selected_by_id[pair_id]
            source = raw[raw_indices[pair_id]]
            current = source["img_curr"][0].numpy()
            gt = batch["gt_diff"][index, 0].cpu().numpy()
            pred = predictions[index, 0].cpu().numpy()
            if current.shape != gt.shape or pred.shape != gt.shape:
                raise ValueError(f"Image/map shape mismatch for {pair_id}")
            gt_mask = np.where(np.abs(gt) > gt_threshold, np.sign(gt), 0)
            pred_mask = np.where(np.abs(pred) > threshold, np.sign(pred), 0)
            zoom = max(float(np.abs(gt).max()), float(np.abs(pred).max()), threshold)
            figure, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.ravel()
            try:
                axes[0].imshow(source["img_prior"][0], cmap="gray", vmin=0, vmax=1)
                axes[1].imshow(current, cmap="gray", vmin=0, vmax=1)
                for axis, values in ((axes[2], gt), (axes[3], pred)):
                    axis.imshow(values, cmap="bwr", vmin=-1, vmax=1)
                axes[4].imshow(gt, cmap="bwr", vmin=-zoom, vmax=zoom)
                for axis, mask in ((axes[5], gt_mask), (axes[6], pred_mask)):
                    axis.imshow(current, cmap="gray", vmin=0, vmax=1)
                    axis.imshow(np.ma.masked_where(mask == 0, mask), cmap="bwr",
                                vmin=-1, vmax=1, alpha=0.55)
                axes[7].imshow(gt_mask, cmap="bwr", vmin=-1, vmax=1)
                axes[8].imshow(pred_mask, cmap="bwr", vmin=-1, vmax=1)
                axes[9].imshow(pred, cmap="bwr", vmin=-zoom, vmax=zoom)
                titles = ("Prior", "Current", "Signed GT [-1,1]", "Signed prediction [-1,1]",
                          f"GT zoom [-{zoom:.3f},+{zoom:.3f}]",
                          "GT mask / current", "Prediction mask / current",
                          "Signed GT mask", "Signed prediction mask",
                          f"Prediction zoom [-{zoom:.3f},+{zoom:.3f}]")
                for axis, title in zip(axes, titles):
                    axis.set_title(title)
                    axis.axis("off")
                metric = (f"Dice={row['dice']:.3f}" if row["is_pathology"]
                          else f"Nuisance FP area={row['nuisance_fp']:.4f}")
                figure.suptitle("\n".join(textwrap.wrap(pair_id, 120)) + "\n" +
                               f"{row['review_group']} | {row['selection']} | {metric} | "
                               f"peaks GT/pred={row['gt_peak']:.3f}/{row['pred_peak']:.3f}\n"
                               f"Red: positive; blue: negative; thresholds: GT={gt_threshold}, prediction={threshold}", fontsize=10)
                figure.tight_layout(rect=(0, 0, 1, 0.87))
                filename = hashlib.sha256(pair_id.encode()).hexdigest()[:16] + ".png"
                figure.savefig(panel_dir / filename, dpi=110)
                row["panel"] = f"cases/{filename}"
            finally:
                plt.close(figure)
        print(f"[review] rendered {sum('panel' in row for row in selected)}/{len(selected)}", flush=True)

    with (output / f"per_pair_{split}.csv").open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "review_cases.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    lines = [f"# {split.title()} Prediction Review", "",
             f"Full split: {len(rows)} pairs. Selected: {len(selected)} panels. Prediction threshold: {threshold}; GT threshold: {gt_threshold}.", "",
             "Red = positive change; blue = negative change. Heatmaps use fixed [-1,1] limits.",
             "Additional zoom panels share the same per-case GT/prediction limits, printed in their titles.",
             "Overlays show signed thresholded masks on the current image, not attention weights.",
             "Source images use the same resizing and per-image normalization as the dataset loader.",
             "Selection is enriched for failures and successes, not a representative performance estimate.",
             "See report JSON for full-split metrics, CSV for every pair, and review_metadata.json for provenance.", ""]
    groups = defaultdict(list)
    for row in selected:
        groups[row["review_group"]].append(row)
    for group, cases in sorted(groups.items()):
        sheet = Image.new("RGB", (3 * 480, ((len(cases) + 2) // 3) * 240), "white")
        for index, row in enumerate(cases):
            with Image.open(output / row["panel"]) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((480, 240))
                sheet.paste(thumbnail, ((index % 3) * 480, (index // 3) * 240))
        filename = "contact_" + hashlib.sha256(group.encode()).hexdigest()[:12] + ".png"
        sheet.save(output / filename)
        lines.extend([f"## {group}", "", f"![{group}]({filename})", ""])
        for row in cases:
            metric = (f"Dice {row['dice']:.3f}" if row["is_pathology"]
                      else f"FP {row['nuisance_fp']:.4f}")
            lines.append(f"- [{Path(row['panel']).stem}]({row['panel']}): {row['selection']}, {metric}")
        lines.append("")
    (output / "index.md").write_text("\n".join(lines), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_per_type(report, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    types = list(report["per_anomaly_type"].keys())
    if not types:
        return
    dice = [report["per_anomaly_type"][t]["dice"] for t in types]
    iou = [report["per_anomaly_type"][t]["iou"] for t in types]
    x = range(len(types))
    fig, ax = plt.subplots(figsize=(1.6 * len(types) + 2, 4))
    ax.bar([i - 0.2 for i in x], dice, width=0.4, label="Dice")
    ax.bar([i + 0.2 for i in x], iou, width=0.4, label="IoU")
    ax.set_xticks(list(x)); ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.legend()
    ax.set_title("RQ1: change-map Dice / IoU per anomaly type")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_examples(examples, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = list(examples.keys())
    if not names:
        return
    n = len(names)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    ax = ax.reshape(2, n)
    for i, name in enumerate(names):
        gt, pred = examples[name]
        ax[0, i].imshow(gt, cmap="bwr", vmin=-1, vmax=1)
        ax[0, i].set_title(f"{name}\nGT"); ax[0, i].axis("off")
        ax[1, i].imshow(pred, cmap="bwr", vmin=-1, vmax=1)
        ax[1, i].set_title("pred"); ax[1, i].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="RQ1 evaluation of the D-GLoRI change-map head")
    p.add_argument("-o", "--dataset_root", required=True)
    p.add_argument("--cache_dir", default=C.FEATURE_CACHE_DIR)
    p.add_argument("--ckpt", default=os.path.join(C.SAVE_FOLDER, "best.pt"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--threshold", type=float, default=None, help="Prediction threshold; defaults to checkpoint config or 0.1")
    p.add_argument("--gt_threshold", type=float, default=None, help="GT threshold; defaults to checkpoint config or prediction threshold")
    p.add_argument("--rotation_k", type=int, choices=[0, 1, 2, 3], default=None)
    p.add_argument("--flip_horizontal", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--out_dir", default=os.path.join(C.PLOTS_FOLDER, "eval"))
    p.add_argument("--review", action="store_true", help="Export source-image review panels and all-pair metrics")
    p.add_argument("--review_per_group", type=int, default=24, help="Panels per pathology/GT-sign group")
    p.add_argument("--review_nuisance", type=int, default=48, help="Highest-FP and random nuisance panels")
    p.add_argument("--review_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if args.review:
        if args.review_per_group < 1 or args.review_nuisance < 0:
            raise ValueError("Invalid review sample counts")
        if Path(args.out_dir).exists() and any(Path(args.out_dir).iterdir()):
            raise ValueError("Use a new, empty review output directory to avoid mixing runs")
        import matplotlib
        from PIL import Image
        source = LongitudinalPairDataset(args.dataset_root, split=args.split)
        pair_ids = [record["pair_id"] for record in source.records]
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("Duplicate pair IDs in the review split manifest")
        if not args.no_cache:
            missing = [pair_id for pair_id in pair_ids
                       if not (Path(args.cache_dir) / pair_id / "feat.pt").is_file()]
            if missing:
                raise FileNotFoundError(f"Review requires complete cache: {len(missing)} missing; first={missing[0]}")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.review:
        checkpoint_bytes = Path(args.ckpt).read_bytes()
        checkpoint_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location=device)
        del checkpoint_bytes
    else:
        state = torch.load(args.ckpt, map_location=device)
    spec = state.get("head_spec")
    run_config = state.get("run_config", {})
    if args.threshold is None:
        args.threshold = run_config.get("eval_threshold", 0.1)
    if args.gt_threshold is None:
        args.gt_threshold = run_config.get("eval_gt_threshold")
        if args.gt_threshold is None:
            args.gt_threshold = args.threshold
    if args.rotation_k is None:
        args.rotation_k = run_config.get("rotation_k", 0)
    if args.flip_horizontal is None:
        args.flip_horizontal = run_config.get("flip_horizontal", False)
    if args.flip_horizontal != run_config.get("flip_horizontal", False):
        raise ValueError("Evaluation horizontal flip differs from the checkpoint")
    feat_dim = spec["kwargs"]["patch_dim"] if spec else None

    if args.no_cache:
        from models import FrozenCXRBackbone
        backbone = FrozenCXRBackbone(
            name=C.BACKBONE, model_id=C.RAD_DINO_MODEL,
            backbone_img_size=C.BACKBONE_IMG_SIZE, num_patch_tokens=C.NUM_PATCH_TOKENS,
            last_n_layers=(feat_dim // spec["kwargs"]["backbone_dim"] if spec else C.LAST_N_LAYERS),
            freeze=C.FREEZE_BACKBONE,
        ).to(device).eval()
        ds = LongitudinalPairDataset(args.dataset_root, split=args.split, img_size=C.IMG_SIZE,
                         rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
    else:
        backbone = None
        ds = CachedPairDataset(args.dataset_root, args.cache_dir, split=args.split,
                       rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    head = build_change_head(spec, device) if spec else build_head(device)
    head.load_state_dict(state["head"])
    print(f"[eval] loaded {args.ckpt} (epoch {state.get('epoch', '?')}) | split={args.split} | n={len(ds)}")

    review_rows = [] if args.review else None
    report, examples = run_eval(head, backbone, loader, device, args.threshold, feat_dim,
                                review_rows, args.gt_threshold)
    report["head_spec"] = spec
    report["checkpoint_epoch"] = state.get("epoch")
    report["rotation_k"] = args.rotation_k
    report["flip_horizontal"] = args.flip_horizontal
    report["target_mode"] = run_config.get("target_mode", "continuous")

    if args.review:
        selected = select_review_cases(review_rows, args.review_per_group,
                                       args.review_nuisance, args.review_seed)
        export_review(head, backbone, ds, review_rows, selected, args.dataset_root,
                      args.split, args.out_dir, device, args.threshold, feat_dim,
                      args.batch_size, spec["kwargs"].get("out_size", C.IMG_SIZE) if spec else C.IMG_SIZE,
                      gt_threshold=args.gt_threshold)
        manifest = Path(args.dataset_root) / "manifest_split.jsonl"
        if not manifest.is_file():
            manifest = Path(args.dataset_root) / "manifest.jsonl"
        metadata = dict(checkpoint=Path(args.ckpt).name, checkpoint_sha256=checkpoint_digest,
                        checkpoint_epoch=state.get("epoch"), head_spec=spec,
                cache_dir=args.cache_dir if not args.no_cache else None,
                        manifest_sha256=file_sha256(manifest), split=args.split,
                        threshold=args.threshold, gt_threshold=args.gt_threshold,
                        rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal,
                        target_mode=report["target_mode"], seed=args.review_seed,
                        per_group=args.review_per_group, nuisance_count=args.review_nuisance,
                        evaluated_pairs=len(review_rows), selected_pairs=len(selected),
                        selection="Low/middle/high Dice and seeded random per pathology/GT sign; "
                                  "highest-FP and seeded random nuisance; unique change groups per bucket")
        (Path(args.out_dir) / "review_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        report["review_selected"] = len(selected)
        total_bytes = sum(path.stat().st_size for path in Path(args.out_dir).rglob("*") if path.is_file())
        print(f"[review] {len(selected)} panels; {total_bytes / 1e6:.1f} MB before summary plots", flush=True)

    json_path = os.path.join(args.out_dir, f"report_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    plot_per_type(report, os.path.join(args.out_dir, f"per_type_{args.split}.png"))
    plot_examples(examples, os.path.join(args.out_dir, f"examples_{args.split}.png"))

    o = report["overall"]; nf = report["nuisance_false_positive_area"]
    print(f"[eval] overall (n={o['n']}): Dice {o['dice']:.4f} | IoU {o['iou']:.4f}")
    for name, m in report["per_anomaly_type"].items():
        print(f"    {name:16s} n={m['n']:5d}  Dice {m['dice']:.3f}  IoU {m['iou']:.3f}  "
              f"sens+ {m['sens_pos']:.3f}  sens- {m['sens_neg']:.3f}")
    print(f"[eval] nuisance-only FP area (n={nf['n']}): {nf['mean_fp_area']:.4f}  (lower is better)")
    print(f"[eval] wrote {json_path} + plots to {args.out_dir}")


if __name__ == "__main__":
    main()
