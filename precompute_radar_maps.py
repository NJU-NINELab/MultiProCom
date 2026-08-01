from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from actual_data_utils import ACTUAL_SCENARIOS
from dataset import ActualMultimodalDataset, build_processed_radar_maps_from_raw, precomputed_map_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the four actual-data radar response maps.")
    parser.add_argument("--root", default="/home/ybpeng/Data/ActualMulData/dataset_multimodal_data")
    parser.add_argument("--output-root", default="experiments/multiprocom/assets/precomputed_radar_maps")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = ActualMultimodalDataset(
        root=args.root,
        scenarios=ACTUAL_SCENARIOS,
        load_modalities=("vision",),
        cache_radar_maps=False,
    )
    output_root = Path(args.output_root)
    completed = skipped = 0
    for sample in dataset.samples:
        destination = precomputed_map_path(
            output_root, sample["scenario"], sample["radar_rel"]
        )
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        maps = build_processed_radar_maps_from_raw(sample["radar_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **maps)
        completed += 1
    print(f"Precomputed {completed} radar frames; skipped {skipped}; output={output_root.resolve()}")


if __name__ == "__main__":
    main()
