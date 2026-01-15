"""Quick preview for the DRR triplet pipeline.

Run (example):
    python test_scripts/drr_triplet_preview.py --ct_dir <CT_DIR> --lungs_dir <LUNGS_SEG_DIR> --bronchi_dir <BRONCHI_SEG_DIR>

This saves a few triplets as PNGs in --out_dir.
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import matplotlib.pyplot as plt

from CT_entities.drr_triplet_pipeline import (
    DRRTripletFromCTDataset,
    RotationParams,
    default_segs_path_builder,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ct_dir', type=str, required=True)
    p.add_argument('--lungs_dir', type=str, required=True)
    p.add_argument('--bronchi_dir', type=str, default=None)
    p.add_argument('--out_dir', type=str, default='triplet_preview_out')
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--p_entity', type=float, default=0.5)
    p.add_argument('--rotation_params', nargs='+', type=float, default=[17.5, 37.5, 0.0, 1.75])
    return p.parse_args()


def main():
    args = parse_args()

    ct_paths = sorted(glob(os.path.join(args.ct_dir, '*.nii.gz')))
    if len(ct_paths) == 0:
        raise RuntimeError(f'No .nii.gz CTs found in {args.ct_dir}')

    os.makedirs(args.out_dir, exist_ok=True)

    def builder(ct_path: str):
        return default_segs_path_builder(
            ct_path,
            lungs_dir=args.lungs_dir,
            bronchi_dir=args.bronchi_dir,
        )

    assert len(args.rotation_params) == 4
    rot_range, max_sum, min_sum, exp = args.rotation_params

    ds = DRRTripletFromCTDataset(
        ct_paths,
        builder,
        rotation=RotationParams(rot_ranges=(rot_range, rot_range, rot_range), max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=exp),
        p_entity=args.p_entity,
        return_meta=True,
    )

    for i in range(min(args.n, len(ds))):
        d0, d1, d2, meta = ds[i]
        # d? are (1,512,512)
        d0 = d0.squeeze(0).numpy()
        d1 = d1.squeeze(0).numpy()
        d2 = d2.squeeze(0).numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(d0, cmap='gray')
        axs[0].set_title('view0 (clean)')
        axs[1].imshow(d1, cmap='gray')
        axs[1].set_title('view1 (rotated)')
        axs[2].imshow(d2, cmap='gray')
        axs[2].set_title(f"view2 (rot+entity={meta['entity_applied']})")
        for ax in axs:
            ax.axis('off')

        out_path = os.path.join(args.out_dir, f'triplet_{i:03d}.png')
        fig.suptitle(f"angles={meta['angles_deg']}, entities={meta.get('entity_names', [])}")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f'Saved {min(args.n, len(ds))} triplets to {args.out_dir}')


if __name__ == '__main__':
    main()


# python [drr_triplet_preview.py](http://_vscodecontentref_/0) --ct_dir "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans" --lungs_dir "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs" --bronchi_dir "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs" --n 5 --out_dir "triplet_preview_out" --rotation_params 17.5 37.5 0 1.75 --p_entity 0.35