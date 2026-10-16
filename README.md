# Rethinking Diffusion-Conditioned Dual-Encoder U-Nets: A Leakage-Free Benchmark for Histopathology Image Segmentation

Official implementation of the paper *"Rethinking Diffusion-Conditioned Dual-Encoder U-Nets: A Leakage-Free Benchmark for Histopathology Image Segmentation."*

**Authors:** Talha Khalid, Ashfaq Ahmad, Iqra Javed, Waseem Iqbal, Syed Baqar Hussain, Amber Sultan
University of Management and Technology (UMT), Lahore, Pakistan

---

## Overview

D3EU-Net (Pani et al., SIPAIM 2025) proposed a diffusion-conditioned, dual-encoder U-Net for histopathology image segmentation. It fuses transformer-based text embeddings and a convolutional spatial-mask encoder through cross-attention, and reports strong Dice and IoU on TNBC, MoNuSeg and GlaS.

This repository provides an **independent re-implementation and critical audit** of that architecture. It is a critical benchmark, not a new architecture. We:

1. document undisclosed or inconsistent elements in the published methodology and its reference implementation;
2. fix them in a leakage-free, early-stopped, fully seeded training and evaluation pipeline;
3. benchmark the corrected dual-encoder model against two independent non-diffusion baselines on four public datasets, including PanNuke, which the original paper did not use.

In the code, our corrected re-implementation of D3EU-Net is named **LUCID-Net** (variant `lucidnet`).

## Issues identified in the published method

1. **Undisclosed test-time mask conditioning.** The spatial encoder takes a segmentation mask, but the paper never states what mask is supplied at test time.
2. **No model-selection criterion.** Training runs a fixed 1000 epochs, with no validation split, early stopping or checkpoint rule.
3. **Diffusion objective inconsistent with the network output.** A DDPM noise-prediction objective is stated, but the network outputs a bounded, sigmoid-activated mask.
4. **Diffusion step count fragile to misconfiguration.** The step count was never ablated, and setting T = 1 silently disables noising.
5. **Dataset-level text conditioning.** One generated description is used per dataset, not per image.
6. **No hardware, memory or runtime figures.**
7. **Missing dataset provenance and split documentation.**
8. **A single fixed split with no repeated-seed evaluation.**

## Fixes applied in this implementation

| # | Problem in the reference implementation | Fix |
|---|---|---|
| 1 | `evaluate()` fed the **real test mask** into the spatial encoder (label leakage) | The spatial encoder receives a **zero mask** in evaluation mode |
| 2 | Diffusion instantiated with `T=1`, so almost no noise was injected | `T = 1000` restored (linear β schedule, 1×10⁻⁴ to 0.02) |
| 3 | Double sigmoid: inside the model and again in `evaluate()` | A single sigmoid, inside the model only |
| 4 | Trained with plain `BCELoss()` only | Composite BCE + Dice loss |
| 5 | Every image received the literal placeholder text `"{Prompt}"` | The real dataset-level DeepSeek-R1 descriptions are used |
| 6 | No checkpointing over 1000 epochs | The best-validation-Dice checkpoint is saved on every improvement |
| 7 | No validation split or early stopping | Train/val/test splits, with early stopping on validation Dice |
| 8 | No seeding | `random`, NumPy, PyTorch and CUDA seeded (seed 42) |
| 9 | Image/mask pairing by sorted directory listing (misaligns on stray files) | Pairing by filename stem, filtered to image extensions, with an explicit mismatch check |
| 10 | Empty, hard-coded data paths | Prepared train/val/test folders for all four datasets |
| 11 | A dead `mask` argument in the model's forward pass | Removed |
| 12 | No data augmentation | Synchronized image/mask flips and ±15° rotation (training only) |
| 13 | No preprocessing for MoNuSeg, GlaS or PanNuke | A dedicated data builder for each dataset |

Because the network outputs a bounded mask rather than a noise residual, diffusion is used as a **stochastic input-augmentation regularizer** during training: a random timestep is drawn per sample and the image is noised. No noise is injected at evaluation (t = 0), and no iterative DDPM sampling is performed.

## Architectures

| Variant | Model | Conditioning |
|---|---|---|
| `lucidnet` | Corrected D3EU-Net re-implementation (LUCID-Net) | Text + spatial mask via cross-attention, with diffusion-noised input |
| `attn_unet` | Attention U-Net with additive attention gates (Oktay et al., 2018) at each skip connection | None |
| `timm_unet` | U-Net decoder over an ImageNet-pretrained timm encoder (ResNet-34) | None |

**LUCID-Net details:**
- **Text encoder.** Word embedding followed by a 2-layer Transformer encoder (4 heads), mean pooling and a linear layer, giving a 128-D embedding.
  - The dataset prompt is tokenized with a simple word-level vocabulary and padded or truncated to 20 tokens.
- **Spatial encoder.** Four 3×3 convolutional layers (64, 128, 256, 512 channels) with global average pooling, giving a 512-D embedding.
- **Conditioned U-Net.**
  - A four-level encoder (64–512 channels) with a 1024-channel bottleneck.
  - At each of the four decoder stages, a residual cross-attention block attends to the concatenated 640-D conditioning vector.
  - The output is computed as `sigmoid(conv1x1(x − conv3x3(dec)))`.

**Training (all variants):**
- loss: 0.5·BCE + 0.5·Dice
- optimizer: Adam, learning rate 1×10⁻³
- scheduler: ReduceLROnPlateau on validation Dice (factor 0.5, patience 15)
- gradient-norm clipping at 1.0
- maximum 300 epochs with early-stopping patience 40; for PanNuke, 80 epochs with patience 12
- batch size 4 (16 for PanNuke)
- input resolution 128×128

**Evaluation:**
- **Per-image metrics.** Dice, IoU, pixel accuracy, precision, recall, specificity and F1, reported as mean ± std over test images at a 0.5 threshold.
- **Pixel-level metrics.** A confusion matrix, ROC-AUC and average precision. Pixels are subsampled to 200k per batch for these curves.

## Datasets

| Dataset | Task | Train | Val | Test | Split notes |
|---|---|---:|---:|---:|---|
| [TNBC](https://zenodo.org/record/1175282) | Breast, nuclei | 35 | 10 | 5 | Patient/slide-level split |
| [MoNuSeg](https://monuseg.grand-challenge.org/Data/) | Multi-organ, nuclei | 30 | 7 | 14 | All 14 official test images used for testing; XML polygons rasterized to masks |
| [GlaS](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) | Colorectal, gland | 68 | 17 | 80 | 85 official training images split 68/17, stratified by `Grade.csv`; testA + testB used for testing; instance masks binarized |
| [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) | Multi-organ, nuclei | 4196 | 524 | 525 | Folds 2 and 3 combined (80/10/10); 6-channel instance/class arrays decoded to binary masks |

Preprocessing:
- Images are resized and normalized to [−1, 1].
- Masks are binarized at 0.5.

The datasets are not redistributed. Download them from the official sources and follow their licenses.

## Results

### Main benchmark (as reported in the paper)

These results come from `code/run_all.py`: 128×128 input, ResNet-34 encoder for `timm_unet`, a single seeded run per combination, test-set metrics in %.

| Dataset | Architecture | Dice | IoU | Acc. | Prec. | Recall |
|---|---|---:|---:|---:|---:|---:|
| TNBC | LUCID-Net (dual-encoder) | 70.85 | 55.27 | 96.81 | 82.11 | 62.98 |
| TNBC | Attention U-Net | **82.37** | **70.23** | **97.82** | 80.66 | 84.90 |
| TNBC | ResNet-34 U-Net | 78.54 | 64.77 | 97.11 | 75.34 | 82.49 |
| MoNuSeg | LUCID-Net (dual-encoder) | 55.26 | 38.92 | 67.21 | 40.16 | 94.52 |
| MoNuSeg | Attention U-Net | **72.69** | **58.02** | 86.33 | 63.01 | 89.12 |
| MoNuSeg | ResNet-34 U-Net | 72.07 | 56.42 | **86.53** | 63.60 | 84.00 |
| GlaS | LUCID-Net (dual-encoder) | 65.87 | 51.01 | 51.99 | 51.62 | 97.83 |
| GlaS | Attention U-Net | **88.10** | **79.99** | **88.48** | 88.44 | 89.74 |
| GlaS | ResNet-34 U-Net | 87.80 | 79.31 | 88.32 | 89.50 | 87.68 |
| PanNuke | LUCID-Net (dual-encoder) | 54.19 | 39.88 | 88.82 | 78.29 | 46.79 |
| PanNuke | Attention U-Net | **79.64** | **69.18** | **94.86** | 78.85 | 81.72 |
| PanNuke | ResNet-34 U-Net | 79.56 | 69.09 | 94.81 | 78.35 | 82.16 |

Once the evaluation-time leakage is removed and early stopping is enforced, both simpler baselines outperform the dual-encoder architecture on every dataset. This is consistent with the leakage hypothesis. It is an inference from our re-implementation, not a claim about how the originally published numbers were produced.

### Notebook extension: 256×256, ResNet-50, TTA and ensembling

The notebook re-runs all 12 combinations with these settings:
- 256×256 input
- a ResNet-50 encoder for `timm_unet`
- batch size 4 (8 for PanNuke)

It then adds inference-time refinement:
- **Test-time augmentation:** horizontal and vertical flips, averaged.
- **Post-processing:** a 3×3 morphological opening.
- **Ensemble:** a probability-level average of `attn_unet` and `timm_unet`.

Test-set Dice / IoU (%):

| Dataset | LUCID-Net | Attention U-Net | ResNet-50 U-Net | Ensemble (Attn + ResNet-50) |
|---|---:|---:|---:|---:|
| TNBC | 74.80 / 59.89 | **83.88 / 72.42** | 81.96 / 69.67 | 83.60 / 71.99 |
| MoNuSeg | 75.11 / 60.21 | 75.25 / 61.90 | **78.48 / 64.67** | 78.42 / **64.67** |
| GlaS | 68.31 / 53.85 | 89.84 / 82.52 | 90.64 / 83.59 | **91.50 / 85.12** |
| PanNuke | 68.04 / 54.41 | 80.14 / 69.83 | 80.62 / 70.59 | **81.01 / 71.13** |

The ranking is unchanged at the higher resolution: the leakage-free LUCID-Net stays below both baselines on every dataset.

## Requirements

- Python 3.12
- PyTorch with CUDA (developed with PyTorch 2.11 and CUDA 12.8) and torchvision
- timm, NumPy, pandas, Pillow, SciPy, scikit-learn, matplotlib, tqdm

```bash
pip install torch torchvision timm numpy pandas pillow scipy scikit-learn matplotlib tqdm
```

Experiments were run on an AMD Ryzen 7 5800X CPU, 60 GB RAM and one NVIDIA GeForce RTX 3080 (10 GB) under Ubuntu Linux.

## Usage

### Scripts (reproduces the paper's main benchmark)

**1. Download the raw datasets** into a `dataset/` directory next to `code/`, using these folder names:
- `TNBC - Nuclei Segmentation`
- `MoNuSeg - Training Data` and `MoNuSeg - Test Data`
- `Warwick QU`
- `PanNuke`

**2. Build the paired train/val/test splits.** The script is idempotent; add `--force` to rebuild.

```bash
python code/prepare_data.py --datasets TNBC MoNuSeg GlaS PanNuke
```

**3. Train and evaluate every dataset × architecture combination:**

```bash
python code/run_all.py --datasets TNBC MoNuSeg GlaS PanNuke \
                       --variants lucidnet attn_unet timm_unet \
                       --max-epochs 300 --patience 40 --batch-size 4
```

For each run, the script saves:
- the best-validation checkpoint (`best.pt`)
- the per-epoch history (`history.json`)
- test metrics (`test_metrics.json`)
- plots: confusion matrix, ROC/PR curves, training curves and qualitative samples
- a cross-dataset `comparison_table.csv` / `.json`

**4. Rebuild the comparison table** from whichever runs have finished. This is safe to run while a sweep is still in progress.

```bash
python code/aggregate_results.py
```

### Notebook

`Rethinking_Diffusion-Conditioned_Dual-Encoder_UNets_A_Leakage-Free_Benchmark.ipynb` is a self-contained walkthrough of the whole pipeline. It contains:
- review notes on the original notebook
- model definitions and a diffusion demonstration
- dataset preparation
- training and evaluation for all four datasets
- test-time augmentation, post-processing and ensembling
- comparison tables

If a checkpoint already exists under `checkpoints_v2/<dataset>/<variant>/best.pt`, the notebook loads it instead of retraining.

## Limitations

- Every result comes from a single fixed seed per (dataset, architecture). The values are point estimates, especially on the 5–14-image TNBC and MoNuSeg test sets.
- The leakage hypothesis is inferred from our own re-implementation and the implementation artifact we worked from. It cannot be verified directly against the original evaluation code.
- The comparison with D3EU-Net's originally reported figures is not a controlled comparison, because those figures come from a different, undisclosed protocol.

## Citation

If you use this code, please cite the accompanying paper:

> T. Khalid, A. Ahmad, I. Javed, W. Iqbal, S. B. Hussain, and A. Sultan, "Rethinking Diffusion-Conditioned Dual-Encoder U-Nets: A Leakage-Free Benchmark for Histopathology Image Segmentation," 2026.

## License

The source code is released under the [MIT License](LICENSE). TNBC, MoNuSeg, GlaS, PanNuke and the ImageNet-pretrained encoder weights are subject to their own licenses and terms of use.

## Contact

Talha Khalid — tk839587@gmail.com