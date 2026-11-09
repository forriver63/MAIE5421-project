import numpy as np


def paired_image_bootstrap(differences, replicates=2000, seed=20261106):
    """One input per image, already averaging payloads/layouts within that image."""
    d = np.asarray(differences, dtype=float)
    if not len(d) or not np.all(np.isfinite(d)):
        raise ValueError('Expected finite per-image differences')
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(replicates, len(d)))].mean(axis=1)
    return dict(difference=float(d.mean()), lower=float(np.quantile(means, .025)),
                upper=float(np.quantile(means, .975)), images=len(d))
