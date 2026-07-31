"""Build paired image/mask train/val/test folders for all 4 datasets.

Fixes "loophole #13" from ANALYSIS.md: the original notebook's SegmentationDataset
assumes ready-made, paired grayscale mask PNGs, but only TNBC ships that way.
MoNuSeg is `.tif` + polygon `.xml` (needs rasterization), GlaS/Warwick QU is
*instance*-labeled `.bmp` (needs binarizing), and PanNuke is a 6-channel
instance+class `.npy` array (needs its own decoder). This script builds all four
into a common `prepared/<Dataset>/{train,val,test}/{images,masks}/*.png` layout,
matching (where the raw data allows it) the paper's reported split sizes:

  TNBC     : 35 / 10 / 5   (patient/slide-level split, not random image-level)
  MoNuSeg  : 30 / 7 / 14   (paper says 30/7/7; we use ALL 14 official test images
             as test, not half of them -- a more robust, still non-leaky
             evaluation set; documented deviation, see ANALYSIS.md)
  GlaS     : 68 / 17 / 80  (matches paper exactly: 85 official train images ->
             68/17, testA+testB (60+20=80) -> test, stratified by Grade.csv)
  PanNuke  : fold_2 + fold_3 combined (fold_1 has no masks.npy on disk) -> 80/10/10

Idempotent: re-running skips a split that already has files, unless --force.
"""
import argparse
import csv
import os
import random
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "prepared")

SEED = 42


def _reset_dir(d, force):
    if os.path.isdir(d):
        if force:
            shutil.rmtree(d)
        else:
            return False
    os.makedirs(d, exist_ok=True)
    return True


def _save_pair(img: Image.Image, mask: Image.Image, out_root, split, stem):
    img_dir = os.path.join(out_root, split, "images")
    mask_dir = os.path.join(out_root, split, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    img.convert("RGB").save(os.path.join(img_dir, f"{stem}.png"))
    mask.convert("L").save(os.path.join(mask_dir, f"{stem}.png"))


# --------------------------------------------------------------------------- #
# TNBC — already paired PNGs; split by slide (patient-level) to avoid leakage.
# --------------------------------------------------------------------------- #

def prepare_tnbc(force=False):
    out_root = os.path.join(OUT, "TNBC")
    if not _reset_dir(out_root, force):
        print("TNBC already prepared, skipping (use --force to rebuild)")
        return
    raw = os.path.join(RAW, "TNBC - Nuclei Segmentation")
    slide_dirs = sorted(d for d in os.listdir(raw) if d.startswith("Slide_"))

    # Slide sizes sum to 50 (paper: 50 total, 35/10/5). Chosen so the *slide*-level
    # split lands on those exact counts (see ANALYSIS.md for the arithmetic):
    test_slides = {"Slide_03"}
    val_slides = {"Slide_02", "Slide_06", "Slide_08"}

    n = {"train": 0, "val": 0, "test": 0}
    for slide in slide_dirs:
        split = "test" if slide in test_slides else "val" if slide in val_slides else "train"
        gt_dir = os.path.join(raw, f"GT_{slide.split('_')[1]}")
        slide_dir = os.path.join(raw, slide)
        for fname in sorted(os.listdir(slide_dir)):
            if not fname.lower().endswith(".png"):
                continue
            stem = os.path.splitext(fname)[0]
            img = Image.open(os.path.join(slide_dir, fname))
            mask = Image.open(os.path.join(gt_dir, fname))
            _save_pair(img, mask, out_root, split, stem)
            n[split] += 1
    print("TNBC:", n)


# --------------------------------------------------------------------------- #
# MoNuSeg — rasterize polygon-vertex XML annotations into binary masks.
# --------------------------------------------------------------------------- #

def _rasterize_monuseg_xml(xml_path, size):
    tree = ET.parse(xml_path)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for region in tree.getroot().iter("Region"):
        pts = [(float(v.get("X")), float(v.get("Y"))) for v in region.iter("Vertex")]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return mask


def prepare_monuseg(force=False):
    out_root = os.path.join(OUT, "MoNuSeg")
    if not _reset_dir(out_root, force):
        print("MoNuSeg already prepared, skipping (use --force to rebuild)")
        return

    train_pool_dir = os.path.join(RAW, "MoNuSeg - Training Data", "Tissue Images")
    train_ann_dir = os.path.join(RAW, "MoNuSeg - Training Data", "Annotations")
    test_dir = os.path.join(RAW, "MoNuSeg - Test Data")

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(train_pool_dir) if f.endswith(".tif"))
    rng = random.Random(SEED)
    rng.shuffle(stems)
    val_stems = set(stems[:7])
    train_stems = set(stems[7:])

    n = {"train": 0, "val": 0, "test": 0}
    for stem in stems:
        split = "val" if stem in val_stems else "train"
        img = Image.open(os.path.join(train_pool_dir, f"{stem}.tif")).convert("RGB")
        mask = _rasterize_monuseg_xml(os.path.join(train_ann_dir, f"{stem}.xml"), img.size)
        _save_pair(img, mask, out_root, split, stem)
        n[split] += 1

    test_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(test_dir) if f.endswith(".tif"))
    for stem in test_stems:
        img = Image.open(os.path.join(test_dir, f"{stem}.tif")).convert("RGB")
        mask = _rasterize_monuseg_xml(os.path.join(test_dir, f"{stem}.xml"), img.size)
        _save_pair(img, mask, out_root, "test", stem)
        n["test"] += 1
    print("MoNuSeg:", n, "(paper reports 30/7/7; we keep all 14 official test images as test)")


# --------------------------------------------------------------------------- #
# GlaS / Warwick QU — binarize instance-labeled annotation bitmaps.
# --------------------------------------------------------------------------- #

def prepare_glas(force=False):
    out_root = os.path.join(OUT, "GlaS")
    if not _reset_dir(out_root, force):
        print("GlaS already prepared, skipping (use --force to rebuild)")
        return
    raw = os.path.join(RAW, "Warwick QU")

    grade = {}
    with open(os.path.join(raw, "Grade.csv"), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            grade[name] = row[" grade (GlaS)"].strip()

    train_stems = sorted({os.path.splitext(f)[0].replace("_anno", "")
                           for f in os.listdir(raw) if f.startswith("train_") and f.endswith(".bmp")})

    # Stratify the held-back validation slice by benign/malignant grade.
    benign = [s for s in train_stems if grade.get(s) == "benign"]
    malignant = [s for s in train_stems if grade.get(s) == "malignant"]
    rng = random.Random(SEED)
    rng.shuffle(benign)
    rng.shuffle(malignant)
    n_val = 17
    n_val_benign = round(n_val * len(benign) / len(train_stems))
    val_stems = set(benign[:n_val_benign] + malignant[: n_val - n_val_benign])

    n = {"train": 0, "val": 0, "test": 0}
    for stem in train_stems:
        split = "val" if stem in val_stems else "train"
        img = Image.open(os.path.join(raw, f"{stem}.bmp")).convert("RGB")
        anno = np.array(Image.open(os.path.join(raw, f"{stem}_anno.bmp")))
        mask = Image.fromarray(((anno > 0) * 255).astype(np.uint8))
        _save_pair(img, mask, out_root, split, stem)
        n[split] += 1

    test_stems = sorted(os.path.splitext(f)[0].replace("_anno", "")
                         for f in os.listdir(raw)
                         if (f.startswith("testA_") or f.startswith("testB_")) and f.endswith(".bmp")
                         and "_anno" not in f)
    for stem in test_stems:
        img = Image.open(os.path.join(raw, f"{stem}.bmp")).convert("RGB")
        anno = np.array(Image.open(os.path.join(raw, f"{stem}_anno.bmp")))
        mask = Image.fromarray(((anno > 0) * 255).astype(np.uint8))
        _save_pair(img, mask, out_root, "test", stem)
        n["test"] += 1
    print("GlaS:", n)


# --------------------------------------------------------------------------- #
# PanNuke — decode the 6-channel instance+class mask array; fold_1 has no
# masks.npy on disk, so only fold_2 + fold_3 (5,245 images) are usable.
# --------------------------------------------------------------------------- #

def prepare_pannuke(force=False):
    out_root = os.path.join(OUT, "PanNuke")
    if not _reset_dir(out_root, force):
        print("PanNuke already prepared, skipping (use --force to rebuild)")
        return

    # mmap_mode='r': fold_2+fold_3 float64 arrays are ~25 GB combined, so we
    # slice lazily from disk rather than loading/concatenating them in RAM.
    fold_arrays = {}
    global_index = []  # list of (fold, local_idx)
    for fold in (2, 3):
        base = os.path.join(RAW, "PanNuke", f"fold_{fold}", f"Fold {fold}")
        imgs = np.load(os.path.join(base, "images", f"fold{fold}", "images.npy"), mmap_mode="r")
        masks = np.load(os.path.join(base, "masks", f"fold{fold}", "masks.npy"), mmap_mode="r")
        fold_arrays[fold] = (imgs, masks)
        global_index.extend((fold, i) for i in range(imgs.shape[0]))

    n_total = len(global_index)
    order = list(range(n_total))
    rng = random.Random(SEED)
    rng.shuffle(order)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    split_of = {}
    for pos in order[:n_train]:
        split_of[pos] = "train"
    for pos in order[n_train:n_train + n_val]:
        split_of[pos] = "val"
    for pos in order[n_train + n_val:]:
        split_of[pos] = "test"

    n = {"train": 0, "val": 0, "test": 0}
    for pos, (fold, local_idx) in enumerate(global_index):
        imgs, masks = fold_arrays[fold]
        split = split_of[pos]
        img = Image.fromarray(np.asarray(imgs[local_idx]).astype(np.uint8))
        fg = (np.asarray(masks[local_idx])[..., :5].sum(axis=-1) > 0)
        mask = Image.fromarray((fg * 255).astype(np.uint8))
        _save_pair(img, mask, out_root, split, f"pannuke_f{fold}_{local_idx:05d}")
        n[split] += 1
    print("PanNuke:", n, f"(out of {n_total}; fold_1 skipped, no masks.npy on disk)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if a split already exists")
    parser.add_argument("--datasets", nargs="+", default=["TNBC", "MoNuSeg", "GlaS", "PanNuke"])
    args = parser.parse_args()

    dispatch = {
        "TNBC": prepare_tnbc,
        "MoNuSeg": prepare_monuseg,
        "GlaS": prepare_glas,
        "PanNuke": prepare_pannuke,
    }
    for name in args.datasets:
        print(f"\n=== Preparing {name} ===")
        dispatch[name](force=args.force)


if __name__ == "__main__":
    main()
