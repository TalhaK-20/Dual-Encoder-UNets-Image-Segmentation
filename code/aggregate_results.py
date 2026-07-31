"""Scan checkpoints/<dataset>/<variant>/test_metrics.json for whatever the sweep has
finished so far and (re)write results/comparison_table.{csv,json}. Safe to run at any
time, including while run_all.py is still running other combinations.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")


def main():
    rows = []
    if os.path.isdir(CHECKPOINTS):
        for dataset in sorted(os.listdir(CHECKPOINTS)):
            for variant in sorted(os.listdir(os.path.join(CHECKPOINTS, dataset))):
                metrics_path = os.path.join(CHECKPOINTS, dataset, variant, "test_metrics.json")
                if not os.path.exists(metrics_path):
                    continue
                with open(metrics_path) as f:
                    s = json.load(f)
                rows.append({
                    "dataset": dataset, "variant": variant,
                    "dice": s["dice_mean"], "dice_std": s["dice_std"],
                    "iou": s["iou_mean"], "iou_std": s["iou_std"],
                    "accuracy": s["accuracy_mean"], "precision": s["precision_mean"],
                    "recall": s["recall_mean"], "f1": s["f1_mean"],
                    "n_test_images": s["n_images"], "best_epoch": s["best_epoch"],
                })

    os.makedirs(RESULTS, exist_ok=True)
    if rows:
        with open(os.path.join(RESULTS, "comparison_table.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(RESULTS, "comparison_table.json"), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {len(rows)} rows -> results/comparison_table.csv")
        for r in rows:
            print(f"  {r['dataset']:10s} {r['variant']:10s}  dice={r['dice']*100:5.2f}%  iou={r['iou']*100:5.2f}%")
    else:
        print("No completed runs found yet.")


if __name__ == "__main__":
    main()
