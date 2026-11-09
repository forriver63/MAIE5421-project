"""Exact DBSCAN for binary 32-bit vectors, epsilon=1, with duplicate weights.

Euclidean distance <=1 on binary vectors is exactly Hamming distance <=1.
Deduplicating in first-occurrence order preserves DBSCAN's core counts and traversal
order; sample_weight restores the multiplicity of every original pixel. Constructing
the exact radius graph avoids the quadratic all-pixel distance calculation.
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN


def binary_dbscan(bit_logits, mask_probability, epsilon=1, min_samples=500):
    if epsilon != 1 or bit_logits.shape[0] != 1 or bit_logits.shape[1] != 32:
        raise ValueError('Optimized exact graph supports one 32-bit image and epsilon=1')
    valid_mask = (mask_probability[0].numpy() > .5).reshape(-1)
    bits = (bit_logits[0].permute(1, 2, 0).numpy().reshape(-1, 32)[valid_mask] > 0).astype(np.uint32)
    if not len(bits):
        return {}, torch.full(mask_probability.shape[-2:], -1.)
    powers = np.left_shift(np.uint32(1), np.arange(31, -1, -1, dtype=np.uint32))
    words = (bits * powers).sum(axis=1, dtype=np.uint32)
    unique, first, inverse, counts = np.unique(words, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    unique, weights, mapped = unique[order], counts[order], rank[inverse]
    lookup = {int(w): i for i, w in enumerate(unique)}
    rows, cols = [], []
    for i, w in enumerate(unique):
        neighbors = [i] + [lookup[int(w) ^ (1 << b)] for b in range(32) if int(w) ^ (1 << b) in lookup]
        for j in sorted(neighbors):
            rows.append(i)
            cols.append(j)
    # Distances are 0 on the diagonal and 1 for one-bit neighbors. Store diagonal
    # zeros explicitly; sparse precomputed DBSCAN includes self in the neighborhood.
    distances = np.array([0. if i == j else 1. for i, j in zip(rows, cols)])
    graph = csr_matrix((distances, (rows, cols)), shape=(len(unique), len(unique)))
    # Row values must be sorted by distance for sklearn's precomputed-neighbor API.
    from sklearn.neighbors import sort_graph_by_row_values
    graph = sort_graph_by_row_values(graph, warn_when_not_sorted=False)
    db = DBSCAN(eps=1, min_samples=min_samples, metric='precomputed').fit(graph, sample_weight=weights)
    pixel_labels = db.labels_[mapped]
    full = np.full(valid_mask.shape, -1, dtype=np.int32)
    full[valid_mask] = pixel_labels
    centroids = {int(label): torch.from_numpy((bits[pixel_labels == label].mean(axis=0) > .5).astype(np.int64))
                 for label in sorted(set(pixel_labels)) if label != -1}
    return centroids, torch.from_numpy(full.reshape(mask_probability.shape[-2:])).float()
