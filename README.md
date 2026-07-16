# CstNet2: Constraint-Aware Point Cloud Learning

CstNet2 is a two-stage point cloud learning project for mechanical parts. The
project extracts explicit geometric constraints from CAD-derived point clouds
and uses those constraints for downstream point cloud classification and
diffusion-based generation.

The project is designed around a strict training rule:

1. Train Stage 1 as a reusable constraint extractor.
2. Freeze all Stage 1 parameters.
3. Train Stage 2 for classification or generation.
4. Do not jointly train Stage 1 and Stage 2 unless you intentionally change the
   research setting.

## Implementation Principle

### Per-Point Constraint Representation

Each point is represented as:

```text
(x, y, z, constraint)
```

The constraint vector has 15 dimensions:

```text
primitive_type: 5D one-hot
direction:      3D vector
dimension:      1D scalar
continuity:     3D point normal
location:       3D vector
```

Primitive types:

```text
0: plane
1: cylinder
2: cone
3: sphere
4: free-form surface / other
```

Direction, dimension, continuity, and location follow the definitions in
`AGENTS.md`. For plane, cylinder, and cone directions, opposite directions are
canonicalized so equivalent axes map to a unique representation.

### Stage 1: Constraint Extraction

Stage 1 receives point coordinates:

```python
xyz.shape == [batch_size, num_points, 3]
```

It predicts:

```text
per-point primitive type
per-point clustering embedding
```

The primitive type is trained with per-point classification loss. The clustering
embedding is trained with a discriminative instance loss using `affiliate_idx` as
the primitive-instance label.

During inference, Stage 1 post-processing performs:

1. Cluster points in Stage 1 embedding space.
2. Assign each cluster a primitive type by majority vote.
3. Fit a primitive to each cluster.
4. Convert fitted primitives into the 15D per-point constraint tensor.

The reusable implementation is in:

```text
networks/stage1_extractor.py
functional/constraints.py
```

### Stage 2: Constraint-Aware Learning

Stage 2 receives the original or noisy point cloud plus the constraint tensor.
It extracts separate features for each constraint component:

```text
xyz + primitive_type
xyz + direction
xyz + dimension
xyz + continuity
xyz + location
```

These five feature streams are fused with attention so each point can adaptively
focus on different geometric constraints.

Implemented Stage 2 models:

```text
networks/stage2.py
    CstNetStage2Classifier
    CstNetStage2Diffusion
```

Classification uses an encoder and classification head. Generation uses a
diffusion denoiser that predicts added Gaussian noise from:

```text
noisy point cloud at timestep t
frozen Stage 1 constraints
diffusion timestep t
```

## Directory Structure

```text
cstnet2/
|-- AGENTS.md                      Project rules and model specification
|-- README.md                      This file
|-- train_cst_pred.py              Stage 1 training entry point
|-- train_cls.py                   Stage 2 classification training
|-- train_gen.py                   Stage 2 diffusion generation training
|-- vis_cstpred.py                 Optional point cloud visualization
|-- func_test.py                   Small local test script
|-- eval_model.py                  EvalScope helper script
|-- cmd.txt                        Historical command notes
|-- data_utils/
|   |-- datasets.py                Dataset loaders and constraint utilities
|   |-- convert_txt_to_npy.py       TXT-to-NPY cache helper
|   |-- cone_gen.py                Synthetic cone data helper
|   |-- vis.py                     Visualization helpers
|-- functional/
|   |-- constraints.py             Constraint assembly and primitive fitting
|   |-- cst_pred_trainer.py        Stage 1 training loop
|   |-- loss.py                    Stage 1 and geometry losses
|-- networks/
|   |-- cst_pred_wrapper.py        Stage 1 model wrapper
|   |-- stage1_extractor.py        Frozen Stage 1 constraint extractor
|   |-- stage2.py                  Stage 2 classifier and diffusion model
|   |-- point_net.py               PointNet modules
|   |-- point_net2.py              PointNet++ modules
|   |-- attn_3dgcn.py              Attention 3DGCN modules
|   |-- dgcnn_gn.py                DGCNN-style modules
|   |-- utils.py                   Shared network utilities
|-- imgs/
|   |-- overall.png                Project figure
|-- records/
|   |-- agent_records.md           Development notes
```

## Data Format

`CstNet2Dataset` expects this directory layout:

```text
dataset_root/
|-- train/
|   |-- class_name_a/
|   |   |-- sample_000.txt
|   |   |-- sample_001.txt
|   |-- class_name_b/
|-- test/
|   |-- class_name_a/
|   |-- class_name_b/
```

Each point file should contain one point per row with 15 columns:

```text
x y z
pmt
mad_x mad_y mad_z
dim
nor_x nor_y nor_z
loc_x loc_y loc_z
affiliate_idx
```

Column meaning:

```text
xyz            point coordinates
pmt            primitive type index, 0-4
mad            main axis direction / main direction
dim            main primitive dimension
nor            point normal
loc            primitive location
affiliate_idx  primitive instance label for clustering supervision
```

The dataset loader automatically caches `*.txt` files as `*.txt.npy` for faster
future loading.

## Deployment Method

### 1. Create or Activate the Python Environment

The project is developed with a local Conda environment named `dp`.

```bash
conda activate dp
```

If you need to create a new environment, install PyTorch according to your CUDA
version first, then install the common Python dependencies:

```bash
conda create -n dp python=3.11
conda activate dp
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn tqdm colorama einops matplotlib tensorboard
```

Optional dependencies:

```bash
pip install wandb          # optional experiment logging
pip install open3d         # optional visualization
pip install pymeshlab      # optional CAD point sampling utilities
pip install pythonocc-core # optional OpenCascade CAD processing, usually via conda-forge
```

For Linux servers, `pythonocc-core` is usually installed with:

```bash
conda install -c conda-forge pythonocc-core
```

### 2. Prepare the Dataset

Place the dataset under a path such as:

```text
data/pcd_cstnet2/Param20K_Extend
```

or use the local Windows path configured in the scripts:

```text
D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend
```

Use `--local True` to select the local Windows path, or override the root
directly:

```bash
python train_cst_pred.py --root_sever data/pcd_cstnet2/Param20K_Extend
python train_cls.py --root_local D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend --local True
```

### 3. Recommended Hardware

The default design targets one NVIDIA RTX 4090 with 24 GB VRAM. If memory is
limited, reduce:

```text
--bs
--n_points
--workers
```

## Running Methods

### Train Stage 1 Constraint Extractor

Train Stage 1 first:

```bash
python train_cst_pred.py --model pointnet2 --bs 128 --n_points 2000 --use_wandb False
```

Useful options:

```text
--model pointnet2|pointnet|attn_3dgcn
--is_sample True         run a small sampled dataloader for debugging
--is_load_weight True    resume from model_trained/<save_name>.pth
--use_wandb True         enable wandb if installed
```

Default Stage 1 checkpoint path:

```text
model_trained/pointnet2_pmt_prim_cluster.pth
```

### Train Stage 2 Classification

After Stage 1 is trained, run classification with Stage 1 frozen:

```bash
python train_cls.py ^
  --constraint_source stage1 ^
  --stage1_model pointnet2 ^
  --stage1_ckpt model_trained/pointnet2_pmt_prim_cluster.pth ^
  --bs 32 ^
  --n_points 2000
```

On Linux:

```bash
python train_cls.py \
  --constraint_source stage1 \
  --stage1_model pointnet2 \
  --stage1_ckpt model_trained/pointnet2_pmt_prim_cluster.pth \
  --bs 32 \
  --n_points 2000
```

For ablation or debugging with ground-truth constraints:

```bash
python train_cls.py --constraint_source gt --is_sample True
```

### Train Stage 2 Diffusion Generation

After Stage 1 is trained, run diffusion denoising training:

```bash
python train_gen.py ^
  --constraint_source stage1 ^
  --stage1_model pointnet2 ^
  --stage1_ckpt model_trained/pointnet2_pmt_prim_cluster.pth ^
  --bs 16 ^
  --n_points 2000 ^
  --timesteps 1000
```

For a small debug run:

```bash
python train_gen.py --constraint_source gt --is_sample True --epoch 1 --bs 2 --n_points 128
```

### Train Stage 2 MFCAD++ Segmentation

The MFCAD++ point files must use the 17-column format documented in
`data_utils/mfcad_seg_dataset.py`. Constraints are assumed to have already been
predicted by the frozen Stage 1 extractor. The class count and stable colors are
read from `data_utils/mfcad_label_map.json`; they are not hard-coded in the
network.

```bash
python train_stage2_seg.py --data_root D:\document\DeepLearning\DataSet\pcd_cstnet2\mfcad_pcd
```

Training accepts either `val/` or `validation/` and can start before a `test/`
directory is present. It writes `class_statistics.json` using the training split
only, plus `last.pth` and `best_point_miou.pth` under the output directory.

Resume all optimizer, scheduler, AMP, epoch, metric, and RNG state with:

```bash
python train_stage2_seg.py --resume model_trained/stage2_mfcad_seg/last.pth
```

Evaluate point-level and Face-level metrics, and optionally export NPZ/PLY views:

```bash
python eval_stage2_seg.py model_trained/stage2_mfcad_seg/best_point_miou.pth --split test --prediction_dir predictions/mfcad_seg
```

### Visualization

Visualization is optional and requires `open3d`.

```bash
python vis_cstpred.py --root_dataset path/to/pointclouds --num_point 2500
```

## Testing Methods

### 1. Syntax and Import Checks

Run compile checks:

```bash
python -B -m compileall functional networks cst_pred cst_fea data_utils train_cls.py train_cst_pred.py train_gen.py vis_cstpred.py
```

Run import checks:

```bash
python -B -c "import train_cst_pred; import train_cls; import train_gen; import vis_cstpred; print('imports ok')"
```

### 2. Tiny Forward Smoke Test

This verifies Stage 1 extraction, Stage 2 classification, and Stage 2 diffusion
without needing the full dataset:

```bash
python -B -c "import torch; from functional.constraints import ground_truth_constraints_to_tensor; from networks.stage1_extractor import FrozenStage1ConstraintExtractor; from networks.stage2 import CstNetStage2Classifier, CstNetStage2Diffusion; B,N=2,64; xyz=torch.rand(B,N,3); pmt=torch.randint(0,5,(B,N)); cst=ground_truth_constraints_to_tensor(pmt, torch.randn(B,N,3), torch.rand(B,N), torch.randn(B,N,3), torch.randn(B,N,3)); print(CstNetStage2Classifier(6).eval()(xyz,cst).shape); print(CstNetStage2Diffusion().eval()(xyz,cst,torch.randint(0,1000,(B,))).shape); print(FrozenStage1ConstraintExtractor(model_name='pointnet2', checkpoint=None).eval()(xyz[:1]).shape)"
```

Expected output shapes:

```text
classification: [2, 6]
diffusion:      [2, 64, 3]
stage1 cst:     [1, 64, 15]
```

### 3. Small Dataset Debug Runs

Use sampled dataloaders before long training:

```bash
python train_cst_pred.py --is_sample True --epoch 1 --bs 2 --n_points 128 --workers 0 --use_wandb False
python train_cls.py --constraint_source gt --is_sample True --epoch 1 --bs 2 --n_points 128 --workers 0
python train_gen.py --constraint_source gt --is_sample True --epoch 1 --bs 2 --n_points 128 --workers 0
```

After these pass, switch `--constraint_source stage1` for Stage 2 and use the
trained Stage 1 checkpoint.

## Output Files

Training creates:

```text
model_trained/    model checkpoints
log/              training logs and tensorboard files
*.txt.npy         dataset cache files next to source TXT files
```

These generated files are intentionally ignored by Git.

## Notes

- Stage 2 scripts freeze Stage 1 automatically through
  `FrozenStage1ConstraintExtractor`.
- `--constraint_source gt` is intended for debugging and ablations, not the main
  two-stage workflow.
- If Stage 1 checkpoint loading fails, train Stage 1 first or pass the correct
  `--stage1_ckpt` path.
- On Windows PowerShell, use `^` for multi-line commands. On Linux/macOS shells,
  use `\`.
