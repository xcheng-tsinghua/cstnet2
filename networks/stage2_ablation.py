"""Classification-only input ablations over frozen Stage 1 constraints."""
from torch import nn
from functional.stage2_ablation_config import COMPONENTS, COMPONENT_SLICES, EXPERIMENTS


def intervene_inputs(constraints, components):
    """Zero removed values before every encoder, token and statistics path."""
    if constraints.ndim != 3 or constraints.shape[-1] != 12:
        raise ValueError("constraints must have shape [B, N, 12]")
    constraints = constraints.clone()
    for name, (start, stop) in zip(COMPONENTS, COMPONENT_SLICES):
        if name not in components:
            constraints[..., start:stop] = 0
    return constraints


class Stage2AblationModel(nn.Module):
    def __init__(self, model, experiment):
        super().__init__()
        self.model = model
        self.spec = EXPERIMENTS[experiment]

    def forward(self, xyz, constraints, return_aux=False):
        constraints = intervene_inputs(constraints, self.spec.components)
        if return_aux:
            return self.model(xyz, constraints, return_aux=True)
        return self.model(xyz, constraints)


def build_ablation_model(task, num_classes, experiment, model_name="constraint_aware", config=None):
    if task != "cls" or model_name != "constraint_aware":
        raise ValueError("ablation entry supports only CSTNet2 classification")
    from networks.classification_models import build_classification_model
    model = build_classification_model(num_classes, config or {"model": model_name})
    return Stage2AblationModel(model, experiment)
