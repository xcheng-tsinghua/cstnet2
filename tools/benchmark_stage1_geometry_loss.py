"""Compare old/new attribute-loss forward + backward on synthetic primitives.

This measures losses only, not dataset loading or a complete training epoch.
Example: python tools/benchmark_stage1_geometry_loss.py --bs 80 --n_points 2048
"""
import argparse
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch
import torch.nn.functional as F
from functional import loss as optimized
from functional.instance_groups import InstanceGroups
import reference_stage1_geometry as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--n_points", type=int, default=512)
    parser.add_argument("--instances", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.bs, args.n_points, args.instances, args.repeats) <= 0:
        parser.error("batch, points, instances, and repeats must be positive")
    device = torch.device(args.device)
    torch.manual_seed(123)
    torch.set_num_threads(2)
    labels = (torch.arange(args.n_points, device=device) % args.instances).expand(args.bs, -1)
    primitive = labels % 5
    xyz = torch.randn(args.bs, args.n_points, 3, device=device)
    target = (F.normalize(torch.randn_like(xyz), dim=-1),
              torch.rand(args.bs, args.n_points, device=device), torch.randn_like(xyz))
    predictions = [torch.randn(args.bs, args.n_points, 5, device=device).log_softmax(-1).requires_grad_(),
                   torch.randn_like(xyz).requires_grad_(),
                   torch.rand_like(target[1]).requires_grad_(), torch.randn_like(xyz).requires_grad_()]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step(module):
        for value in predictions:
            value.grad = None
        groups = InstanceGroups(labels, primitive) if module is optimized else labels
        pmt, mad, dim, loc = predictions
        weights = module._parameter_observability_weights(xyz, primitive, *target, groups)
        total = (
            module._robust_dimension_loss(dim, target[1], primitive, groups, weights[0])
            + module._robust_location_loss(loc, target[2], primitive, groups, weights[1])
            + module._masked_vector_mse(mad, target[0], primitive < 3, sign_invariant=True)
            + module._stage1_geometry_losses(xyz, mad, dim, loc, primitive, groups, *weights)["geom_loss"]
            + module.instance_consistency_loss(pmt, mad, dim, loc, groups, primitive)
        )
        total.backward()
        return total.detach()

    timings = {}
    values = []
    for name, module in (("original", reference), ("optimized", optimized)):
        values.append(step(module))
        sync()
        elapsed = []
        for _ in range(args.repeats):
            start = perf_counter()
            step(module)
            sync()
            elapsed.append(perf_counter() - start)
        timings[name] = statistics.median(elapsed)
    torch.testing.assert_close(values[0], values[1], rtol=3e-5, atol=3e-6)
    print(f"device={device}, batch={args.bs}, points={args.n_points}, instances/cloud={min(args.instances, args.n_points)}")
    print(f"loss forward+backward median: original={timings['original']:.6f}s optimized={timings['optimized']:.6f}s")
    print(f"loss-only speedup={timings['original'] / timings['optimized']:.2f}x (not epoch speedup)")


if __name__ == "__main__":
    main()
