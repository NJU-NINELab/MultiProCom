from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from actual_data_utils import ACTUAL_SCENARIOS
from dataset import MAP_KEYS, ActualMultimodalDataset, TemporalSequenceDataset, precomputed_map_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute global radar normalization stats from training windows.")
    parser.add_argument("--root", default="/home/ybpeng/Data/ActualMulData/dataset_multimodal_data")
    parser.add_argument(
        "--precomputed-radar-root",
        default="experiments/multiprocom/assets/precomputed_radar_maps",
    )
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reservoir-size", type=int, default=2000000)
    parser.add_argument("--output-json", default="experiments/multiprocom/assets/radar_normalization.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    scenarios = list(ACTUAL_SCENARIOS)
    precomputed_root = Path(args.precomputed_radar_root).expanduser().resolve()
    if not precomputed_root.exists():
        raise FileNotFoundError(f"precomputed radar root not found: {precomputed_root}")

    base = ActualMultimodalDataset(
        root=args.root,
        scenarios=scenarios,
        precomputed_radar_root=str(precomputed_root),
        cache_radar_maps=False,
        radar_norm_mode="frame_logminmax",
    )
    temporal = TemporalSequenceDataset(
        base_dataset=base,
        history_len=args.history_len,
        future_steps=args.future_steps,
        window_stride=1,
        require_contiguous=True,
        include_future_maps=False,
    )

    train_idx = list(range(len(temporal)))

    hist_sample_ids = set()
    for wid in train_idx:
        hist_sample_ids.update(int(i) for i in temporal.windows[wid]["hist_indices"])

    if not hist_sample_ids:
        raise RuntimeError("No training history samples found for stats computation.")

    stats = {
        k: {
            "count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "reservoir": np.empty((0,), dtype=np.float32),
        }
        for k in MAP_KEYS
    }

    reservoir_size = max(10000, int(args.reservoir_size))
    num_samples = 0

    for sample_id in sorted(hist_sample_ids):
        sample = base.samples[sample_id]
        map_path = precomputed_map_path(precomputed_root, sample["scenario"], sample["radar_rel"])
        if not map_path.exists():
            raise FileNotFoundError(
                "Missing precomputed map during stats computation: "
                f"{map_path}. Please run precompute_radar_maps.py first."
            )

        with np.load(map_path, allow_pickle=False) as maps:
            for key in MAP_KEYS:
                if key not in maps:
                    raise KeyError(f"Missing key '{key}' in {map_path}")
                arr = np.asarray(maps[key], dtype=np.float32).reshape(-1)
                st = stats[key]
                st["count"] += int(arr.size)
                st["sum"] += float(arr.sum())
                st["sumsq"] += float(np.square(arr, dtype=np.float32).sum())
                res = st["reservoir"]
                if res.size < reservoir_size:
                    take = min(reservoir_size - res.size, arr.size)
                    if take > 0:
                        st["reservoir"] = np.concatenate([res, arr[:take]], axis=0)
                        arr = arr[take:]
                        res = st["reservoir"]
                if arr.size > 0:
                    sample_n = min(int(arr.size), reservoir_size)
                    pick = np.random.choice(arr.size, size=sample_n, replace=False)
                    candidate = np.concatenate([res, arr[pick]], axis=0)
                    if candidate.size > reservoir_size:
                        keep = np.random.choice(candidate.size, size=reservoir_size, replace=False)
                        candidate = candidate[keep]
                    st["reservoir"] = candidate.astype(np.float32, copy=False)

        num_samples += 1
        if num_samples % 500 == 0:
            print(f"Processed {num_samples}/{len(hist_sample_ids)} history samples...")

    out_maps = {}
    for key in MAP_KEYS:
        st = stats[key]
        count = max(int(st["count"]), 1)
        mean = float(st["sum"] / count)
        var = max(float(st["sumsq"] / count - mean * mean), 0.0)
        std = float(var ** 0.5)
        res = st["reservoir"]
        if res.size == 0:
            raise RuntimeError(f"Reservoir for key={key} is empty.")
        p01 = float(np.percentile(res, 1.0))
        p995 = float(np.percentile(res, 99.5))
        out_maps[key] = {
            "count": int(st["count"]),
            "mean": mean,
            "std": std,
            "p01": p01,
            "p995": p995,
            "reservoir_size": int(res.size),
        }

    payload = {
        "mode": "global_stats",
        "root": str(Path(args.root).expanduser().resolve()),
        "precomputed_radar_root": str(precomputed_root),
        "scenarios": scenarios,
        "history_len": int(args.history_len),
        "future_steps": int(args.future_steps),
        "sampling": "all_random_raw_candidate_windows",
        "candidate_window_stride": 1,
        "seed": int(args.seed),
        "num_train_windows": int(len(train_idx)),
        "num_hist_samples": int(len(hist_sample_ids)),
        "maps": out_maps,
    }

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved radar global stats: {out_path}")


if __name__ == "__main__":
    main()
