"""Side-by-side comparison of Thaer vs Nathalia annotations on batch-1 pairs.

For each pair_AN_x_y under
``python_files/annotation tool/Pairs_PNIMIT_1_pairs``, load the *current*
NIfTI (the one the annotation tool draws on, i.e. the second sorted scan)
and draw both annotators' ellipses with label-based colors (the same
scheme used by the annotation tool itself):

  - Appearance    -> red
  - Disappearance -> green
  - Persistence   -> yellow

Layout per pair (one PNG): [Thaer]   [Nathalia]

Coordinates in the JSON files live in the annotation tool's 792x792 canvas
space (see ``main.py`` ``load_image`` -> ``img.resize((792, 792))``), so the
current image is rendered at 792x792.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRS_ROOT = SCRIPT_DIR.parent / "annotation tool" / "Pairs_PNIMIT_1_pairs"
THAER_DIR = (
    SCRIPT_DIR.parent
    / "annotation tool"
    / "Annotations_Pnimit"
    / "JSON_only_named"
    / "Thaer"
)
NATHALIA_DIR = (
    SCRIPT_DIR.parent
    / "annotation tool"
    / "Annotations_Pnimit"
    / "JSON_only_named"
    / "Nathalia"
)
OUT_DIR = SCRIPT_DIR / "files" / "thaer_vs_nathalia_compare"

CANVAS = 792

PAIR_RE = re.compile(r"^pair_(A\d+)_(\d+)_(\d+)$")

# Same scheme as the annotation tool (main.py: self.label_colors).
LABEL_COLORS = {
    "Appearance": "red",
    "Disappearance": "lime",  # bright green for visibility on grayscale CXR
    "Persistence": "yellow",
}
DEFAULT_COLOR = "white"


@dataclass(frozen=True)
class EllipseAnn:
    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float
    label: str


def load_ellipses(json_path: Path) -> list[EllipseAnn]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        return []
    out: list[EllipseAnn] = []
    for item in data[1:]:
        if not isinstance(item, dict):
            continue
        out.append(
            EllipseAnn(
                cx=float(item.get("cx", 0.0)),
                cy=float(item.get("cy", 0.0)),
                rx=float(item.get("rx", 0.0)),
                ry=float(item.get("ry", 0.0)),
                angle_deg=float(item.get("angle", 0.0)),
                label=str(item.get("label", "")),
            )
        )
    return out


def load_current_image(pair_dir: Path) -> np.ndarray:
    """Load the current scan in the same way the annotation tool does.

    See ``main.py`` ``load_image``: ``nii.get_fdata().T`` then min/max
    normalised, resized to 792x792. We must match that so ellipse coords
    line up.
    """
    # Only the original scans match the form "SynapseExport (N).nii.gz".
    # Skip every variant produced later by the seg pipeline (_lung_seg,
    # _notranspose, _resized, etc.).
    scan_re = re.compile(r"^SynapseExport \((\d+)\)\.nii\.gz$")
    nii_files = sorted(
        (p for p in pair_dir.iterdir() if scan_re.match(p.name)),
        key=lambda p: int(scan_re.match(p.name).group(1)),
    )
    if len(nii_files) != 2:
        raise FileNotFoundError(f"Expected 2 scans in {pair_dir}, found {len(nii_files)}")

    arr = nib.load(str(nii_files[1])).get_fdata().T.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]

    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr, dtype=np.float32)

    if arr.shape != (CANVAS, CANVAS):
        img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
        arr = (
            np.asarray(
                img.resize((CANVAS, CANVAS), Image.Resampling.BILINEAR), dtype=np.float32
            )
            / 255.0
        )
    return arr


def draw_ellipses(
    ax: plt.Axes,
    ellipses: list[EllipseAnn],
    linewidth: float = 2.0,
) -> None:
    for e in ellipses:
        color = LABEL_COLORS.get(e.label, DEFAULT_COLOR)
        if e.rx <= 0 and e.ry <= 0:
            # point annotation -> draw a small marker
            ax.plot(e.cx, e.cy, "o", color=color, markersize=6, markerfacecolor="none", mew=1.5)
            continue
        ax.add_patch(
            Ellipse(
                (e.cx, e.cy),
                width=2.0 * e.rx,
                height=2.0 * e.ry,
                angle=e.angle_deg,
                fill=False,
                linewidth=linewidth,
                edgecolor=color,
            )
        )


def setup_axes(ax: plt.Axes, current: np.ndarray, title: str) -> None:
    ax.imshow(current, cmap="gray", origin="upper")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, CANVAS)
    ax.set_ylim(CANVAS, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_pair_figure(pair_name: str, out_path: Path) -> bool:
    pair_dir = PAIRS_ROOT / pair_name
    if not pair_dir.is_dir():
        print(f"  skip (no pair dir): {pair_name}")
        return False

    thaer_json = THAER_DIR / f"Thaer_{pair_name}.json"
    nat_json = NATHALIA_DIR / f"Nathalia_{pair_name}.json"

    if not thaer_json.exists() and not nat_json.exists():
        print(f"  skip (no annotations): {pair_name}")
        return False

    current = load_current_image(pair_dir)
    t_ell = load_ellipses(thaer_json)
    n_ell = load_ellipses(nat_json)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7.5), dpi=150)

    setup_axes(axs[0], current, f"{pair_name} - Thaer ({len(t_ell)})")
    draw_ellipses(axs[0], t_ell)

    setup_axes(axs[1], current, f"{pair_name} - Nathalia ({len(n_ell)})")
    draw_ellipses(axs[1], n_ell)

    # Legend: colors mean labels, not annotators.
    legend_handles = [
        plt.Line2D([0], [0], color=LABEL_COLORS["Appearance"], lw=2, label="Appearance"),
        plt.Line2D([0], [0], color=LABEL_COLORS["Disappearance"], lw=2, label="Disappearance"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        f"{pair_name}   |   Thaer: {len(t_ell)} ellipses   |   Nathalia: {len(n_ell)} ellipses",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def list_pairs() -> list[str]:
    pairs = [p.name for p in PAIRS_ROOT.iterdir() if p.is_dir() and PAIR_RE.match(p.name)]

    def sort_key(name: str) -> tuple[int, int, int]:
        m = PAIR_RE.match(name)
        return (int(m.group(1)[1:]), int(m.group(2)), int(m.group(3)))

    return sorted(pairs, key=sort_key)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs()
    print(f"Found {len(pairs)} pair folders under {PAIRS_ROOT}")
    written = 0
    for name in pairs:
        out_path = OUT_DIR / f"{name}_thaer_vs_nathalia.png"
        ok = make_pair_figure(name, out_path)
        if ok:
            written += 1
            print(f"  wrote {out_path.name}")
    print(f"\nDone. {written}/{len(pairs)} figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
