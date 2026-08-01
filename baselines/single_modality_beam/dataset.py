from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from dataset import ActualMultimodalDataset, TemporalSequenceDataset


class VehicleTrackTemporalDataset(Dataset):
    """Actual-data temporal windows backed by offline motion-component sets."""

    def __init__(
        self,
        base_dataset: ActualMultimodalDataset,
        history_len: int,
        future_steps: int,
        vehicle_track_json: str,
        max_vehicles_per_frame: int = 8,
    ):
        self.temporal = TemporalSequenceDataset(
            base_dataset=base_dataset,
            history_len=history_len,
            future_steps=future_steps,
            window_stride=1,
            require_contiguous=True,
            include_future_maps=False,
        )
        track_path = Path(vehicle_track_json).expanduser().resolve()
        if not track_path.exists():
            raise FileNotFoundError(f"Motion-component cache not found: {track_path}")
        payload = json.loads(track_path.read_text(encoding="utf-8"))
        self.frame_detections = payload.get("frame_detections", {})
        if not self.frame_detections:
            raise ValueError(f"No motion components found in: {track_path}")
        self.max_vehicles_per_frame = int(max_vehicles_per_frame)
        self.windows = self.temporal.windows
        self.group_to_window_indices = self.temporal.group_to_window_indices
        self.base = base_dataset

    @staticmethod
    def _frame_key(sample: dict) -> str:
        return f"{sample['scenario']}::{sample['seq_index']}::{sample['index_num']}"

    def __len__(self):
        return len(self.temporal)

    def __getitem__(self, idx):
        window = self.windows[idx]
        frame_boxes = []
        for base_idx in window["hist_indices"]:
            sample = self.base.samples[base_idx]
            detections = list(self.frame_detections.get(self._frame_key(sample), []))[
                : self.max_vehicles_per_frame
            ]
            detections.extend(
                [[0.0] * 6 for _ in range(self.max_vehicles_per_frame - len(detections))]
            )
            frame_boxes.append(detections)
        return {
            "hist_vehicle_bboxes": torch.tensor(frame_boxes, dtype=torch.float32),
            "future_beam_labels": torch.tensor(
                [int(self.base.samples[i]["label"]) for i in window["future_indices"]],
                dtype=torch.long,
            ),
            "meta_window_id": torch.tensor(int(idx), dtype=torch.long),
            "meta_scenario": window["scenario"],
            "meta_seq_index": window["seq_index"],
            "meta_group_key": window["group_key"],
        }
