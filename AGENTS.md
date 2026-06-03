# Global Rules for the Point Cloud Constraint Project

## 1. Project Overview

This project is a point cloud learning project for mechanical parts. Its core idea is to use explicit geometric constraints extracted from CAD-derived point clouds to improve point cloud classification and point cloud generation.

The project follows a strict two-stage architecture:

1. **Stage 1: Constraint Extraction**

   * Input: raw or noisy 3D point cloud.
   * Output: per-point constraint representation.
   * Stage 1 learns to predict per-point primitive types and per-point clustering features.
   * Points belonging to the same geometric primitive are grouped through clustering.
   * Geometric primitive fitting is then performed for each cluster.
   * The fitted primitives are used to compute the final per-point constraint representation.

2. **Stage 2: Constraint-Aware Classification and Generation**

   * Input: point cloud coordinates and per-point constraint representation extracted by Stage 1.
   * Stage 2 uses the original point cloud and the extracted constraints for downstream tasks.
   * For classification, Stage 2 encodes the point cloud and constraints to predict the mechanical part category.
   * For generation, Stage 2 uses a diffusion-based structure to predict noise from noisy point clouds, constraints, and diffusion timesteps.

Stage 1 and Stage 2 must be trained separately. Stage 1 is trained first as a reusable constraint extraction model. After Stage 1 training is completed, all Stage 1 parameters must be frozen. For both classification and generation, only Stage 2 should be trained.

---

## 2. Per-Point Constraint Representation

The constraint of a point cloud is represented in a per-point format. Each point is expressed as:

```text
(x, y, z, constraint)
```

where `constraint` contains five components:

```text
constraint = {
    primitive_type,
    direction,
    dimension,
    continuity,
    location
}
```

Each component is defined as follows.

---

### 2.1 Primitive Type

`primitive_type` represents the type of the geometric primitive on which the point is located.

For example, if a point is sampled from a cylindrical face of a CAD model, its primitive type is `cylinder`.

The primitive type is represented as a five-dimensional one-hot vector:

```text
plane:                   (1, 0, 0, 0, 0)
cylinder:                (0, 1, 0, 0, 0)
cone:                    (0, 0, 1, 0, 0)
sphere:                  (0, 0, 0, 1, 0)
free-form surface/other: (0, 0, 0, 0, 1)
```

The fifth category is used for free-form surfaces and all other unsupported or unknown primitive types.

---

### 2.2 Direction

`direction` represents the main direction of the primitive on which the point is located. It is represented as a 3D vector.

The direction definition is:

```text
plane:                   surface normal
cylinder:                rotation axis
cone:                    rotation axis
sphere:                  (0, 0, -1), indicating no valid main direction
free-form surface/other: (0, 0, -1), indicating no valid main direction
```

For `plane`, `cylinder`, and `cone`, the direction must be processed by the `dir_unify` function to ensure a unique and consistent direction.

The direction unification rule is:

```python
def dir_unify(direction):
    ax_x = direction.X()
    ax_y = direction.Y()
    ax_z = direction.Z()

    zero_lim = precision.Confusion()

    if ax_z < -zero_lim:
        # If z < 0, reverse the direction.
        direction *= -1.0

    elif abs(ax_z) <= zero_lim and ax_y < -zero_lim:
        # If z is approximately zero and y < 0, reverse the direction.
        direction *= -1.0

    elif abs(ax_z) <= zero_lim and abs(ax_y) <= zero_lim and ax_x < -zero_lim:
        # If z and y are approximately zero and x < 0, reverse the direction.
        direction *= -1.0

    else:
        # No reversal is needed.
        pass

    return direction
```

This rule ensures that equivalent opposite directions are mapped to a unique representation.

---

### 2.3 Dimension

`dimension` represents the main size parameter of the primitive on which the point is located. It is represented as a floating-point scalar.

The dimension definition is:

```text
plane:                   -1.0, indicating no valid main dimension
cylinder:                radius
cone:                    semi-angle
sphere:                  radius
free-form surface/other: -1.0, indicating no valid main dimension
```

---

### 2.4 Continuity

`continuity` represents local surface continuity information at the point.

In this project, it is represented by the normal vector of the point on the mechanical part:

```text
continuity = point normal
```

It is represented as a 3D vector.

---

### 2.5 Location

`location` represents the position parameter of the primitive on which the point is located. It is represented as a 3D vector.

The location definition is:

```text
plane:
    The foot point of the perpendicular projection from the origin to the plane.

cylinder:
    The foot point of the perpendicular projection from the origin to the cylinder rotation axis.

cone:
    The cone apex coordinate.

sphere:
    The sphere center coordinate.

free-form surface/other:
    (0, 0, 0), indicating no valid primitive location.
```

---

## 3. Overall Model Workflow

The project follows the workflow below:

```text
Input point cloud
    ↓
Stage 1: extract per-point constraints
    ↓
Stage 2: use point cloud + constraints for classification or generation
```

Stage 1 is a pretrained and frozen constraint extractor during Stage 2 training.

Stage 2 is the task-specific model. It is trained for classification or generation while Stage 1 remains frozen.

---

## 4. Stage 1: Constraint Extraction

### 4.1 Purpose of Stage 1

Stage 1 is responsible for learning how to extract per-point constraint representations from raw point clouds.

Stage 1 should be trained independently on a very large-scale point cloud dataset. Its role is similar to a foundation model for constraint extraction, but the model size should remain practical for the available hardware.

The target hardware is a single NVIDIA RTX 4090 GPU with 24 GB VRAM. Therefore, the Stage 1 model should not be excessively large. The model architecture, batch size, number of points, and training strategy should be designed with this hardware constraint in mind.

After Stage 1 training is completed, all Stage 1 parameters must be frozen.

Stage 1 should then be used only as a fixed constraint extractor during Stage 2 training and downstream tasks.

Do not jointly train Stage 1 and Stage 2 unless explicitly requested.

---

### 4.2 Stage 1 Input and Output

The input of the Stage 1 model is a 3D point cloud tensor:

```python
point_cloud.shape == [batch_size, num_points, 3]
```

The model outputs two types of data:

1. **Per-point primitive type**

   * Predicts the primitive type of each point.
   * The primitive type follows the five-class definition:

     * `plane`
     * `cylinder`
     * `cone`
     * `sphere`
     * `free-form surface/other`

2. **Per-point clustering feature**

   * Used to cluster points that belong to the same geometric primitive.
   * The training objective is similar to metric learning or clustering-based training:

     * Points belonging to the same primitive should have close clustering features.
     * Points belonging to different primitives should have distant clustering features.

---

### 4.3 Dataset Fields for Stage 1

The training dataset can provide the following fields. See the `CstNet2Dataset` class in:

```text
data_utils/datasets.py
```

Dataset fields:

```text
xyz:
    Point coordinates.

cls:
    Mechanical part category label.

pmt:
    Primitive type.

mad:
    Main axis direction or main direction.

dim:
    Main primitive dimension.

nor:
    Point normal.

loc:
    Primitive location.

affiliate_idx:
    The primitive instance index to which each point belongs.
    This field can be used as supervision for clustering feature learning.
```

The `affiliate_idx` field should be treated as the primitive instance label and can be used for clustering supervision.

---

### 4.4 Computing the Per-Point Constraint Representation

After obtaining the per-point primitive type and per-point clustering feature, the following procedure should be used:

1. Cluster the points according to their clustering features.
2. Treat each point cluster as one primitive instance.
3. Determine the primitive type of each point cluster using the predicted per-point primitive types.
4. Fit the corresponding geometric primitive for each cluster.
5. According to the constraint definitions in Section 2, compute the per-point constraint representation:

   * `primitive_type`
   * `direction`
   * `dimension`
   * `continuity`
   * `location`

The final output of Stage 1 is the per-point constraint representation for the entire point cloud.

---

## 5. Stage 2: Constraint-Aware Classification and Generation

### 5.1 Purpose of Stage 2

Stage 2 is responsible for downstream constraint-aware learning tasks, including:

1. Mechanical part point cloud classification.
2. Mechanical part point cloud generation.

For both classification and generation, only Stage 2 should be trained. Stage 1 must remain frozen.

During Stage 2 training, the input point cloud should first be passed through the frozen Stage 1 model to obtain the per-point constraint representation. Then, the point cloud and its constraint representation are used as input to Stage 2.

---

### 5.2 Stage 2 Input

Stage 2 uses both the original point cloud coordinates and the extracted per-point constraints.

The input consists of:

```text
point coordinates:
    x, y, z

constraint components:
    primitive_type
    direction
    dimension
    continuity
    location
```

The model should separately extract features from the five constraint components:

```text
xyz + primitive_type -> primitive_type feature
xyz + direction -> direction feature
xyz + dimension -> dimension feature
xyz + continuity -> continuity feature
xyz + location -> location feature
```

When merging the five constraint features, an attention mechanism should be used so that different points can adaptively focus on different constraint components.

---

### 5.3 Stage 2 Encoder-Decoder Design

The general Stage 2 encoder-decoder design is:

```text
Input: point cloud + per-point constraints
    ↓
Constraint feature extraction
    ↓
Attention-based constraint feature fusion
    ↓
Point cloud encoder with downsampling
    ↓
Global or latent feature
    ↓
Point cloud decoder with upsampling, if needed
    ↓
Task-specific output
```

For classification, Stage 2 uses an encoder and classification head.

For generation, Stage 2 uses a diffusion-based network that receives:

```text
noisy point cloud at timestep t
per-point constraint representation extracted by frozen Stage 1
diffusion timestep t
```

and predicts the noise.

---

## 6. Training Strategy

### 6.1 Separate Training of Stage 1 and Stage 2

The project must strictly follow a separated two-stage training strategy.

1. Train Stage 1 first.
2. Freeze Stage 1 after training.
3. Train Stage 2 for downstream tasks.
4. Do not update Stage 1 during Stage 2 training.

Stage 1 and Stage 2 must not be jointly trained unless explicitly requested.

---

### 6.2 Stage 1 Training

Stage 1 should be trained independently on a very large-scale point cloud dataset.

Its purpose is to learn a reusable constraint extraction capability.

The Stage 1 model should learn:

```text
input point cloud
    ↓
per-point primitive type
per-point clustering feature
```

Then clustering and primitive fitting are used to compute the final per-point constraint representation.

The Stage 1 model should be designed for training on a single NVIDIA RTX 4090 GPU with 24 GB VRAM. The model should be powerful enough to learn constraint extraction, but it should not be unnecessarily large.

During Stage 2 training, Stage 1 must be used as a frozen pretrained constraint extractor.

---

### 6.3 Stage 2 Training

Stage 2 should be trained after Stage 1 has been trained and frozen.

For classification and generation, only Stage 2 should be trained.

---

## 7. Correct Workflow for Classification

For point cloud classification, the workflow is:

```text
input point cloud: [batch_size, num_points, 3]
    ↓
frozen Stage 1 constraint extractor
    ↓
per-point constraint representation
    ↓
Stage 2 encoder
    ↓
constraint-aware global feature
    ↓
classification head
    ↓
class prediction
```

---

## 8. Correct Workflow for Generation

For point cloud generation, use a diffusion-based training strategy.

At diffusion timestep `t`, the model receives a noisy point cloud:

```python
noisy_point_cloud_t.shape == [batch_size, num_points, 3]
```

The noisy point cloud at timestep `t` should first be passed through the frozen Stage 1 model to obtain the per-point constraint representation.

Then, Stage 2 receives:

```text
noisy point cloud at timestep t
per-point constraint representation extracted by frozen Stage 1
diffusion timestep t
```

Stage 2 predicts the noise added to the point cloud.

The generation training workflow is:

```text
clean point cloud
    ↓
add noise according to diffusion timestep t
    ↓
noisy point cloud at timestep t: [batch_size, num_points, 3]
    ↓
frozen Stage 1 constraint extractor
    ↓
per-point constraint representation
    ↓
Stage 2 encoder-decoder / diffusion network
    ↓
predicted noise
    ↓
diffusion loss
```
