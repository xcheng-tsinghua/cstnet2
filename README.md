# CstNet2: Constraint-Aware Point Cloud Learning

CstNet2 is a two-stage point cloud learning project for mechanical parts. The
project extracts explicit geometric constraints from CAD-derived point clouds
and uses those constraints for downstream point cloud classification and
segmentation.

The project is designed around a strict training rule:

1. Train Stage 1 as a reusable constraint extractor.
2. Freeze all Stage 1 parameters.
3. Train Stage 2 for classification or segmentation.
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

The reusable implementation is used by the offline dataset-preprocessing step:

```text
networks/stage1_extractor.py
functional/constraints.py
```

### Stage 2: Constraint-Aware Learning

Stage 2 receives the original point cloud plus the constraint tensor.
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
networks/stage2_segmentation.py
    ConstraintAwareSegmentationNet
```

Classification uses an encoder and classification head. Segmentation uses an
encoder-decoder with feature propagation and a per-point segmentation head.
The constraint-aware classifier has a single token-fusion architecture: tokens
pooled from every constraint stream and encoder level are fused by a Transformer.

## Directory Structure

```text
cstnet2/
|-- AGENTS.md                      Project rules and model specification
|-- README.md                      This file
|-- .env.example                   WandB API-key template (copy to .env)
|-- train_cst_pred.py              Stage 1 training entry point
|-- gen_cst_pred.py                Offline Stage 1 constraint generation
|-- train_cls.py                   Stage 2 classification training
|-- train_seg.py                   Stage 2 MFCAD++ segmentation training
|-- eval_seg.py                    Stage 2 MFCAD++ segmentation evaluation
|-- vis_cstpred.py                 Optional point cloud visualization
|-- func_test.py                   Small local test script
|-- eval_model.py                  EvalScope helper script
|-- cmd.txt                        Historical command notes
|-- data_utils/
|   |-- datasets.py                Dataset loaders and constraint utilities
|   |-- mfcad_seg_dataset.py       MFCAD++ segmentation dataset loader
|   |-- mfcad_label_map.json       MFCAD++ label names and colors
|   |-- convert_txt_to_npy.py       TXT-to-NPY cache helper
|   |-- cone_gen.py                Synthetic cone data helper
|   |-- vis.py                     Visualization helpers
|-- functional/
|   |-- checkpoint_io.py           Shared fault-tolerant checkpoint saving
|   |-- constraints.py             Constraint assembly and primitive fitting
|   |-- cst_pred_trainer.py        Stage 1 training loop
|   |-- loss.py                    Stage 1 and geometry losses
|-- networks/
|   |-- cst_pred_wrapper.py        Stage 1 model wrapper
|   |-- stage1_extractor.py        Offline frozen Stage 1 inference helper
|   |-- stage2.py                  Stage 2 classifier
|   |-- stage2_segmentation.py     Stage 2 segmentation model
|   |-- feature_propagation.py     Segmentation decoder propagation
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
pip install numpy scipy scikit-learn tqdm colorama einops matplotlib wandb
```

Optional dependencies:

```bash
pip install open3d         # optional visualization
pip install pymeshlab      # optional CAD point sampling utilities
pip install pythonocc-core # optional OpenCascade CAD processing, usually via conda-forge
```

For Linux servers, `pythonocc-core` is usually installed with:

```bash
conda install -c conda-forge pythonocc-core
```

All three training entry points require online WandB logging. Copy the tracked
template to the ignored `.env` file and insert your API key before training:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```dotenv
WANDB_API_KEY=your_real_wandb_api_key
```

The API key is read directly from the project-root `.env`; it is never passed
as a command-line argument or copied into the WandB run config. Training stops
with an error if `.env`, the key, or the `wandb` package is missing.

### 2. Prepare the Dataset

Place the dataset under a path such as:

```text
data/pcd_cstnet2/Param20K_Extend
```

or use the local Windows path configured in the scripts:

```text
D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend
```

Use `--local` to select the local Windows path, or override the root
directly:

```bash
python train_cst_pred.py --root_sever data/pcd_cstnet2/Param20K_Extend
python train_cls.py --root_local D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend --local
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
python train_cst_pred.py --model pointnet2 --bs 128 --n_points 2000
```

Useful options:

```text
--model pointnet2|pointnet|attn_3dgcn
--is_sample              run a small sampled dataloader for debugging
--resume_checkpoint PATH resume model, optimizer, scheduler, and epoch state
--init_from_checkpoint PATH load model weights only for a new training run
```

All training boolean options are value-less flags. For example, use `--local`,
`--is_sample`, `--resume`, or `--use_amp` to enable an option; do not append
`True` or `False`. Stage 1 losses and gradient diagnostics are enabled by
default and can be turned off with flags such as `--disable_mad_loss` and
`--disable_grad_diagnostics`.

Stage 1, classification, and segmentation always log every epoch to WandB.
This includes all losses, learning rates, aggregate metrics, per-class metrics,
class histograms, and confusion matrices produced by each task.

All three training tasks use the same atomic checkpoint writer. A storage I/O
failure is retried three times; if all attempts fail, the temporary file is
removed, the previous valid checkpoint is retained, and training continues.
WandB checkpoint fields ending in `_saved` record whether each attempted epoch
checkpoint was written successfully.

Default Stage 1 checkpoint path:

```text
model_trained/pointnet2_pmt_prim_cluster.pth
```

### Generate Offline Stage 1 Constraints

`gen_cst_pred.py` recursively runs a trained Stage 1 checkpoint over a point-cloud
directory. Only the first three values in every row are used as model input. The
relative directory and file paths are reproduced below the output directory.

The first 15 output columns always use the same layout as ground truth:

```text
xyz(3), pmt(1), mad(3), dim(1), nor(3), loc(3), affiliate_idx(1)
```

Any unknown input columns after `xyz` are not read by Stage 1. They are copied
unchanged after the predicted 15-column core so task-specific fields such as
MFCAD++ face ids and segmentation labels are retained. With the default
`--input_layout auto`, existing 15-column GT constraints are replaced instead
of duplicated. Use `--input_layout raw` if a raw file happens to resemble the
GT column layout, or `--input_layout gt` to force replacement.

```bash
python gen_cst_pred.py \
  --input_dir /opt/data/private/data_set/pcd_cstnet2/MFCAD_raw \
  --output_dir /opt/data/private/data_set/pcd_cstnet2/MFCAD_predicted_constraints \
  --checkpoint model_trained/pointnet2_pmt_prim_cluster/best_constraint_score.pth
```

The checkpoint argument can also be its containing directory. The generator
then selects `best_constraint_score.pth`, `best_pmt_miou.pth`,
`best_cluster_ari.pth`, or `last.pth`, in that order. Model name, Stage 1 mode,
feature settings, and clustering bandwidth are read from current checkpoint
metadata. For a legacy weights-only checkpoint, specify `--model` and
`--stage1_mode` explicitly. Existing outputs are skipped unless `--overwrite`
is supplied. TXT is processed by default; use for example
`--extensions .txt,.npy` when required.

### Train Stage 2 Classification

Stage 1 must first traverse the classification point-cloud dataset and write
its predicted `pmt/mad/dim/nor/loc` values into the point files using the same
columns and data types as the ground-truth fields. Classification never loads
or invokes a Stage 1 model. Select predicted or ground-truth constraints by
pointing the classification loader at the corresponding dataset root.

On Windows, train with a predicted-constraint dataset:

```bash
python train_cls.py ^
  --local ^
  --root_local D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_predicted_constraints ^
  --bs 32 ^
  --n_points 2000
```

On Linux, select the corresponding server dataset directory:

```bash
python train_cls.py \
  --root_sever /opt/data/private/data_set/pcd_cstnet2/Param20K_predicted_constraints \
  --bs 32 \
  --n_points 2000
```

For an oracle ablation, point the same command at the ground-truth dataset:

```bash
python train_cls.py --root_sever /path/to/Param20K_ground_truth --is_sample
```

Classification supports the same baseline families as segmentation:

```bash
python train_cls.py --model constraint_aware
python train_cls.py --model pointnet
python train_cls.py --model pointnet2
python train_cls.py --model dgcnn
python train_cls.py --model attn3dgcn
python train_cls.py --model pointtransformer
python train_cls.py --model pointmamba
python train_cls.py --model pointnext
python train_cls.py --model pointmlp
```

`constraint_aware` always selects the token-fusion classifier. There is no
classification variant switch. Its run and weight name retains the
`constraint_aware_token_fusion` suffix so existing token-fusion weights remain
separate from incompatible checkpoints produced by the removed architectures.

Baselines use XYZ only by default. Add `--baseline_use_constraints` to feed
the 15D per-point constraint stored in the selected dataset as an additional
point feature:

```bash
python train_cls.py --model pointnet2 --baseline_use_constraints \
  --root_sever /path/to/Param20K_predicted_constraints

python train_cls.py --model dgcnn --baseline_use_constraints \
  --root_sever /path/to/Param20K_ground_truth
```

With the default `--save_name stage2_cls`, weights are isolated automatically
as `model_trained/stage2_cls_<model_name>.pth`; constraint-enabled baselines use
the `_constraints` suffix. Training starts fresh unless `--resume` is set.

### Train Stage 2 MFCAD++ Segmentation

The MFCAD++ point files must use the 17-column format documented in
`data_utils/mfcad_seg_dataset.py`. Constraints are assumed to have already been
written into those files by offline Stage 1 preprocessing. The class count and
stable colors are read from `data_utils/mfcad_label_map.json`; they are not
hard-coded in the network.

```bash
python train_seg.py --data_root D:\document\DeepLearning\DataSet\pcd_cstnet2\mfcad_pcd
```

Select the segmentation architecture with `--model`:

```bash
python train_seg.py --model constraint_aware
python train_seg.py --model pointnet
python train_seg.py --model pointnet2
python train_seg.py --model dgcnn
python train_seg.py --model attn3dgcn
python train_seg.py --model pointtransformer
python train_seg.py --model pointmamba
python train_seg.py --model pointnext
python train_seg.py --model pointmlp
```

The four newer baselines are kept in one self-contained file per method:
`networks/point_transformer.py`,
`networks/point_mamba.py`, `networks/pointnext.py`, and
`networks/pointmlp.py`. Each file contains both its classifier and
segmenter. They are pure-PyTorch adaptations of the official architectures,
so no `pointops`, `pointnet2_ops`, KNN-CUDA, OpenPoints, or `mamba_ssm`
installation is required. Baseline training exposes only the common
`--baseline_use_constraints` switch; architecture defaults are edited directly
in each model file and are intentionally not duplicated in the training CLI.
Architecture references are the Point Transformer
[paper](https://arxiv.org/abs/2012.09164) and
[repository](https://github.com/POSTECH-CVLab/point-transformer), the official
[PointMamba](https://github.com/LMD0311/PointMamba),
[PointNeXt](https://github.com/guochengqian/PointNeXt), and
[pointMLP](https://github.com/ma-xu/pointMLP-pytorch) repositories.

Each baseline uses XYZ only by default. Add `--baseline_use_constraints` to
concatenate the full 15D constraint vector as point attributes, producing an
18-channel `XYZ + constraints` input:

```bash
python train_seg.py --model pointnet2 --baseline_use_constraints
python train_seg.py --model dgcnn --baseline_use_constraints
```

All variants share the same dataset, loss, and metrics. Checkpoints are written
to model-named subdirectories under `model_trained/seg/`; constraint-enabled baselines use names such
as `pointnet2_constraints/` so they cannot overwrite XYZ-only results. All
experiments reuse the training-only `class_statistics.json` cache. Evaluation
reconstructs the model family and input mode from the checkpoint; baseline
architecture defaults are read from the corresponding model file.

Training starts fresh by default. Set `--resume` to load `last.pth` from
the selected model's fixed output directory. A missing checkpoint is reported
as an error instead of silently starting over.

Training accepts either `val/` or `validation/` and can start before a `test/`
directory is present. It writes the shared `class_statistics.json` using the
training split only, plus `last.pth` and `best_point_miou.pth` under each model's
output directory.

Resume all optimizer, scheduler, AMP, epoch, metric, and RNG state with:

```bash
python train_seg.py --model constraint_aware --resume
```

Evaluate point-level and Face-level metrics, and optionally export NPZ/PLY views:

```bash
python eval_seg.py model_trained/seg/constraint_aware/best_point_miou.pth --split test --prediction_dir predictions/mfcad_seg
```

Evaluation also requires `.env` and uploads the complete selected-split loss,
point/Face metrics, per-class values, and confusion matrices to WandB while
retaining the local JSON result.

### Visualization

Visualization is optional and requires `open3d`.

```bash
python vis_cstpred.py --root_dataset path/to/pointclouds --num_point 2500
```

## Testing Methods

### 1. Syntax and Import Checks

Run compile checks:

```bash
python -B -m compileall functional networks cst_pred data_utils train_cls.py train_cst_pred.py train_seg.py eval_seg.py vis_cstpred.py
```

Run import checks:

```bash
python -B -c "import train_cst_pred; import train_cls; import train_seg; import eval_seg; import vis_cstpred; print('imports ok')"
```

### 2. Tiny Forward Smoke Test

This verifies construction of the stored constraint representation and Stage 2
classification without loading a Stage 1 model:

```bash
python -B -c "import torch; from functional.constraints import ground_truth_constraints_to_tensor; from networks.stage2 import CstNetStage2Classifier; B,N=2,64; xyz=torch.rand(B,N,3); pmt=torch.randint(0,5,(B,N)); cst=ground_truth_constraints_to_tensor(pmt, torch.randn(B,N,3), torch.rand(B,N), torch.randn(B,N,3), torch.randn(B,N,3)); print(CstNetStage2Classifier(6).eval()(xyz,cst).shape)"
```

Expected output shapes:

```text
classification: [2, 6]
```

### 3. Small Dataset Debug Runs

Use sampled dataloaders before long training:

```bash
python train_cst_pred.py --is_sample --epoch 1 --bs 2 --n_points 128 --workers 0
python train_cls.py --root_sever /path/to/precomputed_dataset --is_sample --epoch 1 --bs 2 --n_points 128 --workers 0
```

After these pass, run Stage 1 preprocessing over the complete dataset and point
Stage 2 at the generated dataset root.

## Output Files

Training creates:

```text
model_trained/    model checkpoints
log/              Stage 1 local JSON training-history files
*.txt.npy         dataset cache files next to source TXT files
```

These generated files are intentionally ignored by Git.

## Notes

- Stage 2 classification and segmentation never load Stage 1. Stage 1 inference
  is an offline dataset-preprocessing step.
- Predicted and ground-truth constraints share the same point-file schema; the
  dataset root determines which constraint source is used.
- On Windows PowerShell, use `^` for multi-line commands. On Linux/macOS shells,
  use `\`.
