"""One host transfer for named tensor finite checks; detailed errors on failure."""
import torch


def assert_finite_tensors(values, label):
    tensors = [(name, value) for name, value in values.items() if torch.is_tensor(value)]
    if not tensors:
        return
    # Reductions stay on device. Transfer all flags together instead of testing
    # every CUDA scalar in Python, which synchronizes once per tensor.
    flags = torch.stack([torch.isfinite(value).all() for _, value in tensors]).cpu().tolist()
    bad = [name for (name, _), finite in zip(tensors, flags) if not finite]
    if bad:
        raise FloatingPointError(f"non-finite {label}: {bad}")
