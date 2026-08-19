"""
Latent-space analysis for the difference embeddings (RQ2 / RQ3).

Validates that the contrastive objective produces a semantically structured latent space:
    - t-SNE / PCA projection colored by change category
    - clustering quality (silhouette score, centroid separation)
    - linear-probe / kNN classification accuracy on embeddings

Run after training to confirm embeddings are linearly separable by change type.
"""

import numpy as np


def collect_embeddings(backbone, head, proj, loader, device):
    """Run the model over a loader and gather (embeddings, labels).

    Returns:
        embeddings: [N, D] numpy array (z_path)
        labels:     [N] numpy array (anomaly type)
    """
    import torch

    head.eval()
    proj.eval()
    embs, labs = [], []
    with torch.no_grad():
        for batch in loader:
            f_prior = backbone(batch["img_prior"].to(device))
            f_curr = backbone(batch["img_curr"].to(device))
            _, z = head(
                f_prior["patch_tokens"], f_curr["patch_tokens"],
                f_prior["cls_token"], f_curr["cls_token"],
            )
            z_path, _ = proj(z)
            embs.append(z_path.cpu().numpy())
            labs.append(batch["anomaly_type"].numpy())
    return np.concatenate(embs), np.concatenate(labs)


def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score of embeddings clustered by label (higher = better separation)."""
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(embeddings, labels))


def linear_probe_accuracy(embeddings: np.ndarray, labels: np.ndarray, test_size: float = 0.3) -> float:
    """Train a logistic-regression probe to predict change type from embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    x_tr, x_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=test_size, random_state=42, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
    return float(clf.score(x_te, y_te))


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, out_path: str = "tsne.png"):
    """Save a t-SNE scatter of embeddings colored by change type."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    proj = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(embeddings)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=8)
    plt.legend(*scatter.legend_elements(), title="change type", loc="best", fontsize=8)
    plt.title("Difference embeddings (t-SNE)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
