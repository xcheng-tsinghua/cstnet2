"""Serializable, torch-free Stage 2 experiment definitions."""
from dataclasses import asdict, dataclass

COMPONENTS = ("primitive_type", "direction", "dimension", "location")
COMPONENT_SLICES = ((0, 5), (5, 8), (8, 9), (9, 12))


@dataclass(frozen=True)
class AblationSpec:
    components: tuple[str, ...] = COMPONENTS
    description: str = "Component input ablation"

    def to_dict(self):
        result = asdict(self)
        result["components"] = list(self.components)
        return result


EXPERIMENTS = {"xyz_only": AblationSpec(
    components=(), description="Zero all constraints; retain the five-stream capacity")}
for name in COMPONENTS:
    EXPERIMENTS[f"no_{name}"] = AblationSpec(
        components=tuple(c for c in COMPONENTS if c != name),
        description=f"Zero {name} before every Stage 2 input path")
    EXPERIMENTS[f"only_{name}"] = AblationSpec(
        components=(name,), description=f"Keep only {name} and XYZ")
SUITES = {
    "all": tuple(EXPERIMENTS),
}


def expand_experiments(names):
    result = []
    for name in names:
        for experiment in SUITES.get(name, (name,)):
            if experiment not in EXPERIMENTS:
                raise ValueError(f"unknown experiment/suite: {experiment}")
            if experiment not in result:
                result.append(experiment)
    return result
