"""Batch-local primitive grouping shared by Stage 1 attribute losses."""
import torch


class InstanceGroups:
    def __init__(self, labels, primitive=None):
        self.shape = labels.shape
        batch = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(labels)
        keys = torch.stack((batch.reshape(-1), labels.reshape(-1).long()), dim=-1)
        unique, self.inverse, self.counts = torch.unique(
            keys, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        self.size = unique.shape[0]
        self.primitive = None
        if primitive is not None:
            votes = torch.zeros(self.size * 5, device=labels.device, dtype=torch.long)
            index = self.inverse * 5 + primitive.reshape(-1).long()
            votes.index_add_(0, index, torch.ones_like(index))
            self.primitive = votes.view(self.size, 5).argmax(dim=-1)

    def sum(self, values):
        """Sum [B*N, ...] values without constructing per-instance masks."""
        result = values.new_zeros((self.size, *values.shape[1:]))
        return result.index_add(0, self.inverse, values)

    def mean(self, values):
        counts = self.counts.to(values.dtype).clamp_min(1)
        return self.sum(values) / counts.reshape((-1,) + (1,) * (values.ndim - 1))

    def first_indices(self):
        positions = torch.arange(self.inverse.numel(), device=self.inverse.device)
        first = torch.full((self.size,), self.inverse.numel(), device=self.inverse.device, dtype=torch.long)
        return first.scatter_reduce_(0, self.inverse, positions, reduce="amin", include_self=True)


def instance_groups(labels, primitive=None):
    return labels if isinstance(labels, InstanceGroups) else InstanceGroups(labels, primitive)
