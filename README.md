# Kvasir-SEG + TinyUNet + CKKS Protection Experiment

This is the medical RGB/binary-segmentation version of the experiment.

## Dataset

Kvasir-SEG:
- 1000 RGB colonoscopy images
- corresponding binary polyp masks
- automatically downloaded from the official Simula dataset server

Default fixed split:
- Train: 700
- Validation: 150
- Test: 150
- Seed: 42

## Five experiments

1. No protection
2. 25% spatial protection
3. 50% spatial protection
4. 75% spatial protection
5. 100% spatial protection

Each condition is preprocessed and saved to its own folder before training.

## Protection interpretation

The percentage is spatial image coverage, not encryption strength.

Inside the protected area:
1. RGB values are genuinely encrypted using CKKS/TenSEAL.
2. Raw ciphertext cannot be fed directly into a conventional TinyUNet.
3. Therefore a deterministic ciphertext-derived noise representation replaces
   the protected image values for CNN training/testing.
4. The binary ground-truth mask is never changed.

Use --save-ciphertext if you want serialized .ckks ciphertext chunks stored too.

## Install

pip install -r requirements.txt

## Recommended first run

### Step 1: preprocess

python prepare_data.py --image-size 256 --overwrite

### Step 2: verify the binary masks

python inspect_masks.py

### Step 3: visually inspect preprocessing

python show_preprocessing.py --sample-index 0

### Step 4: train all five models and evaluate

python run_all.py --skip-preprocess --epochs 35 --batch-size 16

## One-command version

python run_all.py --image-size 256 --epochs 35 --batch-size 16 --overwrite

## Main outputs

runs/comparison/results.csv
runs/comparison/metrics_vs_protection.png
runs/comparison/dice_drop.png
runs/comparison/protection_inputs.png
runs/comparison/prediction_comparison.png

The prediction comparison is arranged as:

25% | 50% | No Protection | 75% | 100%

Row 1: protected RGB input
Row 2: binary ground truth
Row 3: TinyUNet prediction

## Extra robustness experiment

cross_condition_eval.py evaluates all 5 models on all 5 test conditions.

This produces a 5 x 5 matrix:

              TEST
TRAIN     0  25  50  75  100
0
25
50
75
100

The clean-trained row (Train=0%) directly shows how a normal TinyUNet degrades
when increasing protection is introduced at deployment.

Outputs:

runs/cross_condition/dice_matrix.csv
runs/cross_condition/iou_matrix.csv
runs/cross_condition/dice_heatmap.png
runs/cross_condition/iou_heatmap.png
runs/cross_condition/all_cross_condition_results.csv

## Metrics

Primary:
- Dice
- IoU

Diagnostic:
- Precision
- Recall
- Pixel accuracy
- predicted foreground fraction
- ground-truth foreground fraction

Foreground-fraction metrics are included specifically to detect trivial model
collapse such as predicting the whole image as foreground.

## Coordinate-preserving RMT segmentation adaptation

The RMT conditions are an adaptation inspired by DisguisedNets. They use
block-wise orthogonal transforms at fixed image coordinates and intentionally
omit block permutation. Segmentation masks are saved unchanged. RMT transforms
the whole image; it does not implement encrypted training or inference.

Conditions:

- `rmt_block16_n0`
- `rmt_block16_n005`
- `rmt_block16_n010`
- `rmt_block8_n005`
- `rmt_block32_n005`

RMT preprocessing uses the fixed split seed 42 and image size 256. It writes
only new RMT condition folders, refuses to overwrite an existing RMT folder,
and records transformation details in each condition's
`rmt_metadata.json`, plus the exact fixed matrices in `rmt_matrices.npy`. RMT
is not applied to PBUNet. The transformation seed is independent of training
seeds, and its matrices/noise are stable across runs. The transform itself does not clip; processed RGB PNGs use the existing
deterministic `[0,1]` clipping and 8-bit serialization convention.

Run the in-memory deterministic sanity tests:

```bash
python3 -m unittest test_rmt_protection.py
```

Create RMT processed data without touching the existing clean/CKKS folders:

```bash
python3 prepare_data.py --rmt-only --image-size 256 --seed 42
```

The RMT training conditions can then be selected explicitly (this starts
training, so run it only when ready):

```bash
python3 run_multi_seed.py --models tinyunet --conditions rmt_block16_n0 rmt_block16_n005 rmt_block16_n010 rmt_block8_n005 rmt_block32_n005 --seeds 42 101 202 303
```
