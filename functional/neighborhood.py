"""A forward-local XYZ neighborhood, shared without retaining an N x N cache."""
from dataclasses import dataclass

import torch


@dataclass
class PointNeighborhood:
    indices: torch.Tensor
    distances: torch.Tensor

    def excluding_first(self, k):
        # Preserve the existing GCN/PCA convention: discard nearest rank zero.
        return self.indices[..., 1:k + 1], self.distances[..., 1:k + 1]


def build_neighborhood(xyz, k):
    with torch.no_grad():
        distances = torch.cdist(xyz.float(), xyz.float())
        nearest = distances.topk(min(k, xyz.shape[1]), dim=-1, largest=False, sorted=True)
    return PointNeighborhood(nearest.indices, nearest.values)
