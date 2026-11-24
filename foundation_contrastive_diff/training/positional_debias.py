"""Noise-derived patch-feature diagnostics; never modifies the frozen encoder."""

import torch
from torch.nn import functional as F


NOISE_METHOD = "independent Gaussian pixels clipped to [-3,3] and mapped to [0,1]"


@torch.no_grad()
def noise_features(backbone, seed, device):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(
        1, 3, backbone.backbone_img_size, backbone.backbone_img_size,
        generator=generator).clamp(-3, 3).add(3).div(6)
    return backbone(image.to(device))["patch_tokens"][0].double().cpu()


def remove_subspace(features, basis):
    return features - (features @ basis) @ basis.T


def positional_similarity(left, right):
    similarities = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T
    same_position = similarities.diag().mean()
    count = similarities.shape[0]
    other_position = (similarities.sum() - similarities.diag().sum()) / (count * (count - 1))
    return dict(same_position=float(same_position), other_position=float(other_position),
                positional_gap=float(same_position - other_position))


@torch.no_grad()
def diagnose_features(backbone, train_dataset, ranks, seed, device, train_pairs):
    width = noise_features(backbone, seed, device)
    if (not ranks or ranks != sorted(set(ranks)) or ranks[0] != 0
            or any(type(rank) is not int or not 0 <= rank < width.shape[1] for rank in ranks)
            or train_pairs < 1 or not len(train_dataset)):
        raise ValueError("Diagnostics require sorted unique ranks including zero and training examples")
    _, singular_values, right_vectors = torch.linalg.svd(width, full_matrices=False)
    if max(ranks) >= len(singular_values):
        raise ValueError("Noise feature matrix is too small for the diagnostic ranks")
    basis = right_vectors.T[:, :max(ranks)]
    left = noise_features(backbone, seed + 1, device)
    right = noise_features(backbone, seed + 2, device)
    generator = torch.Generator().manual_seed(seed + 3)
    indices = torch.randperm(len(train_dataset), generator=generator)[:train_pairs].tolist()
    rows = []
    energies = {rank: [] for rank in ranks}
    for index in indices:
        sample = train_dataset[index]
        for key in ("img_prior", "img_curr"):
            features = backbone(sample[key][None].to(device))["patch_tokens"][0].double().cpu()
            energy = features.square().sum()
            if not torch.isfinite(features).all() or energy <= 0:
                raise ValueError("Invalid features in training-only retention diagnostics")
            for rank in ranks:
                retained = remove_subspace(features, basis[:, :rank]).square().sum() / energy
                energies[rank].append(float(retained))
    for rank in ranks:
        values = energies[rank]
        rows.append(dict(
            rank=rank, **positional_similarity(
                remove_subspace(left, basis[:, :rank]), remove_subspace(right, basis[:, :rank])),
            train_energy_retained_mean=sum(values) / len(values),
            train_energy_retained_min=min(values),
            noise_energy_removed=float(singular_values[:rank].square().sum()
                                       / singular_values.square().sum())))
    if not all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in rows):
        raise ValueError("Nonfinite positional diagnostics")
    return dict(basis=basis.float(), rows=rows,
                training_pair_ids=[train_dataset.records[index]["pair_id"] for index in indices],
                noise_seed=seed, noise_method=NOISE_METHOD,
                note="Whole-patch projection; not INSID3's selective cross-image matching. "
                     "Retention alone does not establish semantic preservation.")
