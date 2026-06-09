# Multi-Stream Constraint Encoder Implementation Summary

## Refactoring Completed

### Changes Made

**File: `/d/document/DeepLearning/cstnet2/networks/stage2.py`**
- Complete architectural refactoring from early-fusion to multi-stream design
- Preserved: `PointwiseMLP`, `SinusoidalTimestepEmbedding`, public model APIs
- Removed: `ConstraintComponentFusion` early fusion pattern

**File: `/d/document/DeepLearning/cstnet2/tests/test_stage2.py`** (new)
- Comprehensive test suite covering all modules and requirements
- ~400 lines of tests with 30+ test cases

---

## Architecture Overview

### Six Independent Streams (All Levels)

1. **Component Streams** (5)
   - `primitive_type`: concat(xyz, primitive_type) → PointwiseMLP
   - `direction`: concat(xyz, direction) → PointwiseMLP
   - `dimension`: concat(xyz, dimension) → PointwiseMLP
   - `continuity`: concat(xyz, continuity) → PointwiseMLP
   - `location`: concat(xyz, location) → PointwiseMLP

2. **Main Constraint Stream** (1)
   - `xyz`: xyz only → PointwiseMLP (serves as latent constraint representation)

### Shared Spatial Sampling (Once Per Level)

**SharedNeighborhoodSampler**
- Single FPS call on xyz: O(1) per level
- Single KNN construction via `torch.cdist`
- Reused for all 6 streams: identical center_idx, group_idx, relative_xyz

```
center_idx:    [B, S]      (FPS result)
group_idx:     [B, S, K]   (KNN indices)
center_xyz:    [B, S, 3]
relative_xyz:  [B, S, K, 3]
```

### Local Vector Attention (Per Stream, Per Level)

**LocalVectorAttentionAggregator** (independent for each stream)

Critical components:
- **Relation**: `key_neighbor - query_center + position_encoding`
- **Attention shape**: [B, S, K, C_out] (vector attention, not scalar)
- **Softmax dimension**: K (neighbor dimension)
- **Residual structure**:
  ```python
  x = norm1(center_projection(center_features) + aggregated_message)
  x = norm2(x + ffn(x))
  ```

### One-Way Cross-Attention (After Each Hierarchy Level)

**ComponentToConstraintCrossAttention**
- **Input**: constraint feature + 5 component features (all [B, S, C])
- **Process**: Point-wise multi-head attention
  - Reshape to [B*S, 1, C] for query, [B*S, 5, C] for context
  - Attention over 5 components only
- **Output**: Updated constraint feature only
- **Component streams**: Remain unchanged (no in-place modifications)
- **Gradients**: Flow to components through K/V roles

### Hierarchical Levels

**Level 1**: 
- Input: [B, N, 96]
- Set Abstraction: n_center=512, n_near=32 → [B, 512, 160]
- Cross-Attention: Updates constraint stream only

**Level 2**:
- Input: [B, 512, 160]
- Set Abstraction: n_center=128, n_near=32 → [B, 128, 320]
- Cross-Attention: Updates constraint stream only

### Feature Propagation (Diffusion Only)

**ConstraintFeaturePropagation**

From coarse level back to original resolution:

1. **Reverse propagation** through hierarchy levels
2. **KNN interpolation** at each step:
   - `k = min(3, num_source_points)` (handles 1-point, 2-point, 3+ point sets)
   - Inverse distance weighting: `weights = 1 / distance` (normalized)
   - Distance clamping: `eps = 1e-8` to avoid division by zero
3. **Fuse with skip features** from each level
4. **Final interpolation** to original point resolution

---

## Key Implementation Details

### ✅ Requirement: Import CONSTRAINT_DIM

```python
from functional.constraints import CONSTRAINT_DIM, split_constraint_tensor
```

- Used for validation in both classifiers
- Never hard-coded as 15
- Default value still 15, but derived from import

### ✅ Requirement: No Max Pooling in KNN Aggregation

Local vector attention uses explicit `softmax` over K dimension, not max pooling.

Global max pooling only in:
- Classifier: `constraint_final.max(dim=1)[0]` ✓
- Diffusion: `constraint_final.max(dim=1)[0]` ✓

### ✅ Requirement: Shared Spatial Sampling

```python
# Called ONCE per level in MultiStreamSetAbstractionLayer.forward()
center_idx, group_idx, center_xyz, relative_xyz = self.sampler(xyz)

# Reused by all 6 aggregators in loop
for stream_name, features in streams.items():
    center_features = utils.index_points(features, center_idx)
    neighbor_features = utils.index_points(features, group_idx)
    aggregated = self.aggregators[stream_name](...)
```

### ✅ Requirement: One-Way Component-to-Constraint Update

```python
# Only constraint stream is updated
constraint_updated = cross_attention(
    constraint_feature,
    primitive, direction, dimension, continuity, location
)

# Component streams remain unmodified
primitive_updated = primitive  # Unchanged
direction_updated = direction  # Unchanged
# etc...
```

### ✅ Requirement: Point-Wise Cross-Attention

```python
# Reshape for independent per-point processing
query = constraint_feature.unsqueeze(2).reshape(B*S, 1, C)
context = component_context.reshape(B*S, 5, C)
updated, _ = self.attention(query, context, context)
updated = updated.reshape(B, S, C)
```

No global spatial attention between different points.

### ✅ Requirement: Public APIs Preserved

**Classifier**:
```python
CstNetStage2Classifier.forward(xyz, constraints) → [B, n_classes]
```

**Diffusion**:
```python
CstNetStage2Diffusion.forward(noisy_xyz, constraints, timesteps) → [B, N, 3]
```

Both return expected shapes with correct output types (log_softmax, noise prediction).

---

## Test Coverage

**Test File**: `/d/document/DeepLearning/cstnet2/tests/test_stage2.py`

### Module Tests (9 test classes)

1. **TestConstraintStreamInitializer**
   - Output shapes (6 streams × [B, N, C])
   - Gradient flow to all inputs

2. **TestSharedNeighborhoodSampler**
   - Output shapes (center_idx, group_idx, xyz, relative_xyz)
   - Handles small point sets (N < n_center, N < n_near)

3. **TestLocalVectorAttentionAggregator**
   - Output shapes [B, S, C_out]
   - Gradient flow
   - No NaN in output

4. **TestMultiStreamSetAbstractionLayer**
   - All 6 streams use identical spatial grouping
   - Output consistency

5. **TestComponentToConstraintCrossAttention**
   - Output shape matches input constraint shape
   - Components not modified in-place
   - Constraint changes after cross-attention
   - Gradients to component K/V inputs
   - Attention parameter gradients

6. **TestMultiStreamConstraintEncoder**
   - Forward pass shapes through 2 hierarchy levels
   - Gradient flow through encoder

7. **TestConstraintFeaturePropagation**
   - Propagation to original resolution
   - Handles small source sets (< 3 points)
   - No NaN in output

8. **TestCstNetStage2Classifier**
   - Output shape [B, n_classes]
   - Constraint dimension validation
   - Output is log-softmax
   - Gradient flow

9. **TestCstNetStage2Diffusion**
   - Output shape [B, N, 3]
   - Constraint dimension validation
   - Different batch sizes
   - Gradient flow

### Edge Case Tests

- Very small point clouds (N=10)
- Single point per batch (N=1)
- Diffusion with small point cloud

### No Old Fusion Tests

- Verify `ConstraintStreamInitializer` used, not `ConstraintComponentFusion`

---

## Verification Checklist

### Core Requirements

- ✅ Six independent streams (5 components + 1 constraint)
- ✅ Shared spatial sampling (FPS/KNN once per level)
- ✅ Reused for all 6 streams with identical indices
- ✅ Local vector attention (not max pooling in neighborhoods)
- ✅ Vector attention shape [B, S, K, C_out]
- ✅ Relation = key - query + position_encoding
- ✅ Softmax over K (neighbor) dimension
- ✅ Residual: norm1(center_proj + msg), norm2(x + ffn(x))
- ✅ One-way cross-attention (components unchanged)
- ✅ Point-wise cross-attention (no global spatial)
- ✅ Gradients flow to all components through K/V
- ✅ Feature propagation uses k-NN with k=min(3, ...)
- ✅ Distance epsilon clamping (1e-8)

### Public API Requirements

- ✅ Classifier: forward(xyz, constraints) → [B, n_classes]
- ✅ Diffusion: forward(noisy_xyz, constraints, timesteps) → [B, N, 3]
- ✅ Input validation for CONSTRAINT_DIM
- ✅ Output type: log_softmax for classifier
- ✅ Preserved module names and interfaces

### Code Quality Requirements

- ✅ Shape comments on all important operations
- ✅ No Python loops over individual points
- ✅ Batched tensor operations throughout
- ✅ Numerical safeguards for edge cases
- ✅ No unnecessary .contiguous() calls
- ✅ No gradient blockage or .detach() before cross-attention
- ✅ All modules differentiable

### File Organization

- ✅ Main implementation: `/d/document/DeepLearning/cstnet2/networks/stage2.py`
- ✅ Tests: `/d/document/DeepLearning/cstnet2/tests/test_stage2.py`
- ✅ No modifications to Stage 1
- ✅ No modifications to utils.py or constraints.py

---

## Backward Compatibility

The refactoring preserves the public API while completely changing the internal architecture:

| Interface | Old | New | Compatible |
|-----------|-----|-----|------------|
| Classifier forward | `forward(xyz, constraints)` | `forward(xyz, constraints)` | ✅ |
| Diffusion forward | `forward(noisy_xyz, constraints, timesteps)` | `forward(noisy_xyz, constraints, timesteps)` | ✅ |
| Input shapes | [B, N, 3], [B, N, 15] | [B, N, 3], [B, N, 15] | ✅ |
| Output shapes | [B, n_classes], [B, N, 3] | [B, n_classes], [B, N, 3] | ✅ |
| Stage 1 input/output | Unchanged | Unchanged | ✅ |

Internal changes:
- ❌ Early fusion removed (ConstraintComponentFusion deleted)
- ❌ Set abstraction completely rewritten (now multi-stream)
- ❌ Local aggregation changed (vector attention replaces max pooling)
- ❌ Decoder refactored for propagated constraint features

---

## Performance Considerations

### Computational Efficiency

1. **Shared Sampling**: Single FPS/KNN call instead of 6 → ~6× speedup for spatial grouping
2. **Vector Attention**: Slightly heavier than max pooling locally, but more expressive
3. **Cross-Attention**: Point-wise (reshape to [B*S, ...]) → efficient multi-head attention
4. **Feature Propagation**: KNN-based (k=3) → O(N × log M) complexity with proper indexing

### Memory Efficiency

1. **Stream Storage**: 6 × feature_dim at each hierarchy level
   - Compared to: 1 fused tensor + 5 component tensors (old design stored more)
   - New design: Cleaner separation, no early fusion concatenations

2. **Skip Features**: Stored for all 6 streams at each level
   - Negligible overhead for 2-level hierarchy

### Inference Optimization Opportunities

1. KNN implementation in SharedNeighborhoodSampler can be replaced by:
   - FAISS GPU KNN
   - Custom CUDA KNN kernel
   - Optimized third-party library
   (Currently encapsulated in SharedNeighborhoodSampler for easy swap)

2. Cross-attention can use sparse attention patterns for efficiency

3. Feature propagation can use approximate KNN (k-d tree, LSH)

---

## Future Extensions

The architecture enables several research directions:

1. **Adaptive Feature Fusion**: Different weighting of components per point
2. **Hierarchical Component Attention**: Component importance varies by level
3. **Component-Specific Decoders**: Different decoder branches per component
4. **Constraint Confidence Estimation**: Learn confidence weights from components
5. **Multi-Scale Feature Ensemble**: Skip connections from multiple levels to components
6. **Task-Specific Component Routing**: Route components differently for different tasks

All these can be implemented by modifying existing modules without changing the core 6-stream architecture.

---

## Testing Instructions

To run the test suite (requires pytest and torch):

```bash
cd /d/document/DeepLearning/cstnet2
python -m pytest tests/test_stage2.py -v
```

For quick smoke test without pytest:

```bash
python -c "
from networks.stage2 import CstNetStage2Classifier, CstNetStage2Diffusion
from functional.constraints import CONSTRAINT_DIM
import torch

# Quick shapes test
cls = CstNetStage2Classifier(10)
xyz = torch.randn(2, 256, 3)
constraints = torch.randn(2, 256, CONSTRAINT_DIM)
out_cls = cls(xyz, constraints)
assert out_cls.shape == (2, 10)

# Diffusion test
diff = CstNetStage2Diffusion()
out_diff = diff(xyz, constraints, torch.randint(0, 1000, (2,)))
assert out_diff.shape == (2, 256, 3)

print('✓ All shape tests passed')
"
```
