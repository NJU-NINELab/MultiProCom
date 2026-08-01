from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "multiprocom": "MultiProCom", "vision_only": "V-Only", "radar_only": "R-Only",
    "wo_ramf": "w/o RAMF", "wo_afsp": "w/o AFSP",
}


def main():
    parser = argparse.ArgumentParser(description="Summarize full-data deployment training metrics.")
    parser.add_argument("--runs-root", default="experiments/multiprocom/training_runs")
    parser.add_argument("--output-dir", default="experiments/multiprocom/metrics")
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, scene_rows, step_rows = [], [], []
    for method, label in LABELS.items():
        path = runs_root / method / "full_data_metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["full_data_metrics"]
        rows.append({
            "method": label, "epoch": payload["epoch"], "windows": payload["num_train_windows"],
            "top1": metrics["top1_mean"], "top3": metrics["top3_mean"], "top5": metrics["top5_mean"],
            "beam_mae": metrics["beam_mae"], "within_one": metrics["within_one_accuracy"],
        })
        for scene, scene_metrics in sorted(metrics["per_scenario"].items()):
            scene_rows.append({
                "method": label, "scenario": scene, "top1": scene_metrics["top1_mean"],
                "top3": scene_metrics["top3_mean"], "top5": scene_metrics["top5_mean"],
                "beam_mae": scene_metrics["beam_mae"], "within_one": scene_metrics["within_one_accuracy"],
            })
            for horizon in range(1, 9):
                step_rows.append({
                    "method": label, "scenario": scene, "horizon": horizon,
                    "top1": scene_metrics["step_metrics"][f"Top1@t+{horizon}"],
                    "top3": scene_metrics["step_metrics"][f"Top3@t+{horizon}"],
                    "top5": scene_metrics["step_metrics"][f"Top5@t+{horizon}"],
                })
    for filename, data in (
        ("fulltrain_summary.csv", rows),
        ("fulltrain_by_scenario.csv", scene_rows),
        ("fulltrain_by_scenario_step.csv", step_rows),
    ):
        if not data:
            continue
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines = [
        "# Actual-data full training results", "",
        "> These are training-data resubstitution metrics, not held-out generalization metrics.", "",
        "| Method | Epoch | Windows | Top-1 | Top-3 | Top-5 | Beam MAE | Within ±1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['epoch']} | {row['windows']} | {row['top1']:.4f} | "
            f"{row['top3']:.4f} | {row['top5']:.4f} | {row['beam_mae']:.4f} | {row['within_one']:.4f} |"
        )
    (output_dir / "fulltrain_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} full-data method summaries under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
