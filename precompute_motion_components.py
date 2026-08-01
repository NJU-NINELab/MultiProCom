from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from actual_data_utils import ACTUAL_SCENARIOS
from dataset import ActualMultimodalDataset


def motion_boxes(previous, current, max_boxes=8):
    diff = cv2.absdiff(current, previous)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    # A large foreground ratio is normally caused by exposure changes or camera
    # vibration, not by a robot.  Dropping the whole frame is safer than feeding
    # those components to the visual branch as confident targets.
    if float(np.count_nonzero(mask)) / float(mask.size) > 0.18:
        return []
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    height, width = current.shape
    candidates = []
    for component in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[component]]
        cx, cy = centers[component]
        if area < 35 or area > 5000 or cy < height * 0.25:
            continue
        confidence = min(1.0, max(0.05, area / 800.0))
        candidates.append((area, [cx / width, cy / height, w / width, h / height, confidence, 1.0]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:max_boxes]]


def _temporally_matches(box, neighbor):
    cx, cy, width, height = box[:4]
    nx, ny, nwidth, nheight = neighbor[:4]
    area = max(width * height, 1e-6)
    neighbor_area = max(nwidth * nheight, 1e-6)
    area_ratio = neighbor_area / area
    if not 0.2 <= area_ratio <= 5.0:
        return False
    center_distance = float(np.hypot(cx - nx, cy - ny))
    motion_radius = max(0.10, 1.5 * max(width, height, nwidth, nheight))
    return center_distance <= motion_radius


def filter_temporally_consistent(candidate_sequence, max_boxes=8):
    """Suppress one-frame motion noise and retain spatially associated components."""
    filtered = []
    last_index = len(candidate_sequence) - 1
    for frame_index, candidates in enumerate(candidate_sequence):
        previous = candidate_sequence[frame_index - 1] if frame_index > 0 else []
        following = candidate_sequence[frame_index + 1] if frame_index < last_index else []
        retained = []
        for box in candidates:
            matches_previous = any(_temporally_matches(box, other) for other in previous)
            matches_following = any(_temporally_matches(box, other) for other in following)
            if frame_index == 0:
                consistent = matches_following
            elif frame_index == last_index:
                consistent = matches_previous
            else:
                consistent = matches_previous or matches_following
            if not consistent:
                continue
            confidence_scale = 1.0 if matches_previous and matches_following else 0.55
            stable_box = list(box)
            stable_box[4] = float(np.clip(stable_box[4] * confidence_scale, 0.0, 1.0))
            retained.append(stable_box)
        retained.sort(key=lambda item: item[2] * item[3] * item[4], reverse=True)
        filtered.append(retained[:max_boxes])
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Build label-free motion boxes for actual robot frames.")
    parser.add_argument("--root", default="/home/ybpeng/Data/ActualMulData/dataset_multimodal_data")
    parser.add_argument("--output", default="experiments/multiprocom/assets/motion_components.json")
    parser.add_argument("--max-boxes", type=int, default=8)
    args = parser.parse_args()
    base = ActualMultimodalDataset(root=args.root, scenarios=ACTUAL_SCENARIOS, load_modalities=("vision",))
    frame_detections, groups = {}, {}
    for scene in ACTUAL_SCENARIOS:
        samples = sorted((s for s in base.samples if s["scenario"] == scene), key=lambda s: s["index_num"])
        unique = list(dict.fromkeys(str(s["image_path"]) for s in samples))
        candidate_sequence, previous = [], None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for image_path in unique:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            frame = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
            frame = cv2.GaussianBlur(clahe.apply(frame), (5, 5), 0)
            candidate_sequence.append([] if previous is None else motion_boxes(previous, frame, args.max_boxes))
            previous = frame
        stable_sequence = filter_temporally_consistent(candidate_sequence, args.max_boxes)
        boxes_by_path = dict(zip(unique, stable_sequence))
        available = 0
        for sample in samples:
            key = f"{scene}::{sample['seq_index']}::{sample['index_num']}"
            boxes = boxes_by_path[str(sample["image_path"])]
            frame_detections[key] = boxes
            available += bool(boxes)
        groups[f"{scene}::session"] = {
            "frames": len(samples), "unique_camera_frames": len(unique),
            "available_features": available, "coverage": available / max(len(samples), 1),
        }
        print(f"{scene}: frames={len(samples)} unique={len(unique)} coverage={available/max(len(samples),1):.4f}")
    payload = {
        "format": "actual_motion_bbox_set_v2",
        "feature_order": ["cx", "cy", "width", "height", "confidence", "valid"],
        "normalized_coordinates": True,
        "tracker": "clahe_pairwise_motion_with_temporal_association",
        "groups": groups,
        "frame_detections": frame_detections, "frame_features": {},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
