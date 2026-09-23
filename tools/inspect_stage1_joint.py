"""Compare two trusted Stage 1 checkpoints without data, training, or W&B."""

import argparse
import json
from pathlib import Path

import torch


def distribution(values):
    values = values.detach().double().reshape(-1)
    finite = torch.isfinite(values)
    result = {"count": values.numel(), "nonfinite": int((~finite).sum())}
    if finite.any():
        quantiles = torch.quantile(values[finite], torch.tensor(
            [0., .1, .5, .9, 1.], dtype=torch.float64))
        result.update(zip(("min", "p10", "median", "p90", "max"), quantiles.tolist()))
    return result


def read_checkpoint(path):
    # Only load checkpoints from your own trusted training runs: these can
    # contain Python metadata in addition to tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError(f"Missing model/state_dict in {path}")
    return checkpoint, state


def metadata(checkpoint):
    args = checkpoint.get("args", {})
    optimizer = checkpoint.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "config": {key: args.get(key) for key in (
            "model", "train_phase", "bs", "lr", "decay_rate", "grad_clip",
            "training_recipe", "data_root", "use_extra_features",
            "w_pmt", "w_cluster", "w_mad", "w_dim", "w_loc",
        )},
        "optimizer_groups": [{key: group.get(key) for key in (
            "group_name", "lr", "weight_decay", "eps", "betas",
        )} for group in groups],
        "optimizer_nonfinite_values": sum(
            int((~torch.isfinite(value)).sum())
            for state in optimizer.get("state", {}).values()
            for value in state.values() if torch.is_tensor(value)
        ),
    }


def compare(geometry, joint):
    directions = {}
    blocks = {}
    incompatible = []
    nonfinite = {}
    for name, before in geometry.items():
        after = joint.get(name)
        if not torch.is_tensor(before) or not before.is_floating_point():
            continue
        if not torch.is_tensor(after) or before.shape != after.shape:
            incompatible.append(name)
            continue
        before, after = before.double(), after.double()
        bad = int((~torch.isfinite(after)).sum())
        if bad:
            nonfinite[name] = bad
        if name.endswith(".directions"):
            norm_before, norm_after = before.norm(dim=0), after.norm(dim=0)
            valid = (norm_before > 0) & (norm_after > 0)
            cosine = ((before[:, valid] / norm_before[valid]) *
                      (after[:, valid] / norm_after[valid])).sum(dim=0)
            directions[name] = {
                "geometry_norm": distribution(norm_before),
                "joint_norm": distribution(norm_after),
                "joint_over_geometry_norm": distribution(
                    norm_after[norm_before > 0] / norm_before[norm_before > 0]),
                "direction_change_deg": distribution(
                    cosine.clamp(-1, 1).acos() * (180. / torch.pi)),
            }
        # Compare parameter tensors separately from BatchNorm running buffers.
        if name.endswith(("running_mean", "running_var")):
            continue
        parts = name.split(".")
        block = ".".join(parts[:2]) if parts[0] == "embedding" else parts[0]
        totals = blocks.setdefault(block, [0., 0., 0.])
        totals[0] += before.square().sum().item()
        totals[1] += after.square().sum().item()
        totals[2] += (after - before).square().sum().item()
    return {
        "normalized_directions": directions,
        "blocks": {name: {
            "geometry_norm": old ** .5,
            "joint_norm": new ** .5,
            "relative_change": delta ** .5 / max(old ** .5, 1e-30),
        } for name, (old, new, delta) in blocks.items()},
        "incompatible_geometry_keys": incompatible,
        "joint_nonfinite_tensors": nonfinite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("joint", type=Path)
    args = parser.parse_args()
    geometry_checkpoint, geometry = read_checkpoint(args.geometry)
    joint_checkpoint, joint = read_checkpoint(args.joint)
    report = {
        "note": "Parameter comparison only; does not prove a cause or measure training gradients.",
        "geometry": metadata(geometry_checkpoint),
        "joint": metadata(joint_checkpoint),
        **compare(geometry, joint),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
