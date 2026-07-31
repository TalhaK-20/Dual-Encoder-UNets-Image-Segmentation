"""Driver: train + evaluate every (dataset x model variant) combination and
write a final cross-dataset comparison table.

Usage:
    python run_all.py --datasets TNBC MoNuSeg GlaS PanNuke \
                       --variants lucidnet attn_unet timm_unet \
                       --max-epochs 300 --patience 40
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.engine import train_and_evaluate
from src.report import save_all
from src.text_prompts import DATASET_PROMPTS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREPARED = os.path.join(ROOT, "prepared")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")

# PanNuke is ~65x bigger than the paper's tiny datasets; cap its epoch budget
# so a full sweep finishes in a reasonable time (early stopping will usually
# trigger well before this on the tiny sets anyway).
MAX_EPOCHS_OVERRIDE = {"PanNuke": 80}
PATIENCE_OVERRIDE = {"PanNuke": 12}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["TNBC", "MoNuSeg", "GlaS", "PanNuke"])
    parser.add_argument("--variants", nargs="+", default=["lucidnet", "attn_unet", "timm_unet"])
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rows = []
    for dataset in args.datasets:
        data_root = os.path.join(PREPARED, dataset)
        if not os.path.isdir(data_root):
            print(f"!! {dataset} not found in {PREPARED}, skipping (run prepare_data.py first)")
            continue
        text_prompt = DATASET_PROMPTS[dataset]
        max_epochs = MAX_EPOCHS_OVERRIDE.get(dataset, args.max_epochs)
        patience = PATIENCE_OVERRIDE.get(dataset, args.patience)
        batch_size = 16 if dataset == "PanNuke" else args.batch_size

        for variant in args.variants:
            out_dir = os.path.join(RESULTS, dataset, variant)
            ckpt_dir = os.path.join(CHECKPOINTS, dataset, variant)
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"\n{'=' * 80}\n{dataset} / {variant}\n{'=' * 80}")

            result = train_and_evaluate(
                dataset_name=dataset, data_root=data_root, text_prompt=text_prompt,
                variant=variant, out_dir=ckpt_dir, max_epochs=max_epochs, patience=patience,
                batch_size=batch_size, device=device,
            )
            save_all(result, out_dir, dataset, variant)

            summary = result["test_summary"]
            rows.append({
                "dataset": dataset, "variant": variant,
                "dice": summary["dice_mean"], "dice_std": summary["dice_std"],
                "iou": summary["iou_mean"], "iou_std": summary["iou_std"],
                "accuracy": summary["accuracy_mean"], "precision": summary["precision_mean"],
                "recall": summary["recall_mean"], "f1": summary["f1_mean"],
                "n_test_images": summary["n_images"], "best_epoch": summary["best_epoch"],
            })
            print(f"-> {dataset}/{variant} TEST  Dice={summary['dice_mean']*100:.2f}%  "
                  f"IoU={summary['iou_mean']*100:.2f}%  Acc={summary['accuracy_mean']*100:.2f}%")

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, "comparison_table.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote comparison table -> {csv_path}")

    with open(os.path.join(RESULTS, "comparison_table.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
