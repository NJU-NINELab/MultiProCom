import csv
import json
import random
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MAP_KEYS = ("power_map", "range_angle_map", "range_doppler_map", "delay_doppler_map")
RADAR_NORM_MODES = ("frame_logminmax", "global_stats")


def _parse_int(text: str):
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _extract_first_int(text: str):
    if text is None:
        return None
    match = re.search(r"(\d+)", str(text))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _sample_sort_key(sample: dict, fallback_row_id: int):
    index_num = sample.get("index_num")
    if index_num is not None:
        return (0, int(index_num), fallback_row_id)
    frame_num = sample.get("frame_num")
    if frame_num is not None:
        return (1, int(frame_num), fallback_row_id)
    time_stamp = str(sample.get("time_stamp", "")).strip()
    if time_stamp:
        return (2, time_stamp, fallback_row_id)
    return (3, fallback_row_id)


def _window_order_index(sample: dict):
    index_num = sample.get("index_num")
    if index_num is not None:
        return int(index_num)
    frame_num = sample.get("frame_num")
    if frame_num is not None:
        return int(frame_num)
    return None

def _log_minmax(array, eps=1e-6):
    x = np.log1p(np.maximum(np.asarray(array, dtype=np.float32), 0.0))
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def _global_percentile_norm(array, stats_dict: dict, eps=1e-6):
    lo = float(stats_dict.get("p01", 0.0))
    hi = float(stats_dict.get("p995", 1.0))
    if hi <= lo:
        raise ValueError(f"Invalid global stats percentiles: p01={lo}, p995={hi}")
    x = np.asarray(array, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def _resize_2d(array, target_hw):
    h, w = target_hw
    return cv2.resize(np.asarray(array, dtype=np.float32), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _load_image(image_path, image_size):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def infer_axes(data):
    if data.ndim != 3:
        raise ValueError(f"Expected 3D radar tensor, got shape {data.shape}.")
    small_axes = [idx for idx, size in enumerate(data.shape) if size <= 16]
    antenna_axis = small_axes[0] if small_axes else 0
    remaining = [idx for idx in range(3) if idx != antenna_axis]
    range_axis = max(remaining, key=lambda idx: data.shape[idx])
    chirp_axis = min(remaining, key=lambda idx: data.shape[idx])
    return antenna_axis, range_axis, chirp_axis


def canonicalize_radar(data):
    if not np.iscomplexobj(data):
        data = data.astype(np.float32).astype(np.complex64)
    if data.ndim != 3:
        raise ValueError(f"Expected radar data with shape (ant, range, chirp), got {data.shape}.")

    antenna_axis, range_axis, chirp_axis = infer_axes(data)
    return np.moveaxis(data, [chirp_axis, antenna_axis, range_axis], [0, 1, 2]).astype(np.complex64)


def compute_radar_maps(radar_data, n_angles=64, use_window=True):
    num_chirps, _, num_ranges = radar_data.shape

    power_map = np.sum(np.abs(radar_data) ** 2, axis=1).astype(np.float32)

    cube = radar_data
    if use_window:
        range_window = np.hanning(num_ranges).astype(np.float32)
        doppler_window = np.hanning(num_chirps).astype(np.float32)
        cube = cube * range_window[None, None, :]
        cube = cube * doppler_window[:, None, None]

    delay_doppler_fft = np.fft.fft(cube, axis=0)
    delay_doppler_fft = np.fft.fftshift(delay_doppler_fft, axes=0)
    delay_doppler_map = np.sum(np.abs(delay_doppler_fft) ** 2, axis=1).astype(np.float32)

    range_fft = np.fft.fft(cube, axis=2)
    range_fft = range_fft - np.mean(range_fft, axis=0, keepdims=True)

    doppler_fft = np.fft.fft(range_fft, axis=0)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
    range_doppler_map = np.sum(np.abs(doppler_fft) ** 2, axis=1).astype(np.float32)

    angle_fft = np.fft.fft(range_fft, n=n_angles, axis=1)
    angle_fft = np.fft.fftshift(angle_fft, axes=1)
    range_angle_map = np.max(np.abs(angle_fft) ** 2, axis=0).astype(np.float32)

    return {
        "power_map": power_map,
        "range_angle_map": range_angle_map,
        "range_doppler_map": range_doppler_map,
        "delay_doppler_map": delay_doppler_map,
    }


def preprocess_radar_maps(
    radar_maps,
    power_size=(32, 64),
    rd_size=(32, 64),
    ra_size=(64, 64),
    dd_size=(32, 64),
    apply_logminmax: bool = True,
):
    def _norm(x):
        return _log_minmax(x) if apply_logminmax else np.asarray(x, dtype=np.float32)

    return {
        "power_map": _resize_2d(_norm(radar_maps["power_map"]), power_size),
        "range_angle_map": _resize_2d(_norm(radar_maps["range_angle_map"]), ra_size),
        "range_doppler_map": _resize_2d(_norm(radar_maps["range_doppler_map"]), rd_size),
        "delay_doppler_map": _resize_2d(_norm(radar_maps["delay_doppler_map"]), dd_size),
    }


def build_processed_radar_maps_from_raw(
    radar_path,
    power_size=(32, 64),
    rd_size=(32, 64),
    ra_size=(64, 64),
    dd_size=(32, 64),
    n_angles=64,
    use_window=True,
):
    radar_path = Path(radar_path)
    if radar_path.suffix.lower() == ".bin":
        radar_cube = load_actual_radar_cube(radar_path)
        radar_maps = compute_actual_radar_maps(radar_cube, n_angles=n_angles)
    else:
        radar_raw = np.load(radar_path, allow_pickle=False)
        radar_cube = canonicalize_radar(radar_raw)
        radar_maps = compute_radar_maps(radar_cube, n_angles=n_angles, use_window=use_window)
    return preprocess_radar_maps(
        radar_maps=radar_maps,
        power_size=power_size,
        rd_size=rd_size,
        ra_size=ra_size,
        dd_size=dd_size,
        apply_logminmax=True,
    )


def load_actual_radar_cube(radar_path, num_chirps=32, num_tx=3, num_rx=4, num_samples=64, num_iq=2):
    """Decode one real-capture frame using the acquisition-side RadarParser layout."""
    adc_data = np.fromfile(radar_path, dtype="<i2")
    ints_per_frame = num_chirps * num_tx * num_rx * (num_samples // 2) * num_iq * 2
    if adc_data.size != ints_per_frame:
        raise ValueError(
            f"Actual radar frame must contain {ints_per_frame} int16 values, "
            f"got {adc_data.size}: {radar_path}"
        )
    adc_data = adc_data.reshape(
        1, num_chirps, num_tx, num_rx, num_samples // 2, num_iq, 2
    )
    adc_data = adc_data.transpose(0, 1, 2, 3, 4, 6, 5)
    adc_data = adc_data.reshape(1, num_chirps, num_tx, num_rx, num_samples, num_iq)
    adc_complex = (1j * adc_data[..., 0] + adc_data[..., 1]).astype(np.complex64)
    return adc_complex.reshape(num_chirps, num_tx * num_rx, num_samples)


def compute_actual_radar_maps(frame_data, n_angles=64):
    """Build the four map tensors expected by the existing radar encoder."""
    frame_data = np.asarray(frame_data, dtype=np.complex64)
    if frame_data.shape != (32, 12, 64):
        raise ValueError(f"Expected actual radar cube [32,12,64], got {frame_data.shape}")
    power_map = np.sum(np.abs(frame_data) ** 2, axis=1, dtype=np.float32)
    range_fft = np.fft.fft(frame_data, axis=2)
    range_fft = range_fft - np.mean(range_fft, axis=0, keepdims=True)
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
    range_doppler_map = np.sum(np.abs(doppler_fft) ** 2, axis=1, dtype=np.float32)
    angle_fft = np.fft.fftshift(np.fft.fft(doppler_fft, n=n_angles, axis=1), axes=1)
    range_angle_map = np.max(np.abs(angle_fft) ** 2, axis=0).T.astype(np.float32)
    delay_centered = frame_data - np.mean(frame_data, axis=0, keepdims=True)
    delay_fft = np.fft.fftshift(np.fft.fft(delay_centered, axis=0), axes=0)
    delay_doppler_map = np.sum(np.abs(delay_fft) ** 2, axis=1, dtype=np.float32)
    return {
        "power_map": power_map,
        "range_angle_map": range_angle_map,
        "range_doppler_map": range_doppler_map,
        "delay_doppler_map": delay_doppler_map,
    }


def precomputed_map_path(precomputed_root, scenario, radar_rel_path):
    radar_rel = Path(radar_rel_path)
    return Path(precomputed_root) / str(scenario) / radar_rel.with_suffix(".npz")


class DeepSense6GDataset(Dataset):
    """
    Minimal camera-radar dataset for deepsense6G scenarios.
    Required columns in CSV: unit1_rgb, unit1_radar, unit1_beam.
    """

    def __init__(
        self,
        root="/home/ybpeng/Data/deepsense",
        scenarios: Iterable[str] = ("Scenario31",),
        image_size=(224, 224),
        power_size=(32, 64),
        rd_size=(32, 64),
        ra_size=(64, 64),
        dd_size=(32, 64),
        cache_radar_maps: bool = True,
        max_radar_cache_items: int = 0,
        precomputed_radar_root: str = "",
        radar_norm_mode: str = "frame_logminmax",
        radar_norm_stats: str = "",
        load_modalities: Iterable[str] = ("vision", "radar"),
        sample_format: str = "deepsense",
    ):
        self.root = Path(root)
        self.image_size = tuple(image_size)
        self.power_size = tuple(power_size)
        self.rd_size = tuple(rd_size)
        self.ra_size = tuple(ra_size)
        self.dd_size = tuple(dd_size)
        self.cache_radar_maps = bool(cache_radar_maps)
        self.max_radar_cache_items = int(max_radar_cache_items)
        self.precomputed_radar_root = Path(precomputed_radar_root).expanduser().resolve() if precomputed_radar_root else None
        self.load_modalities = frozenset(str(m).strip().lower() for m in load_modalities)
        invalid_modalities = self.load_modalities.difference({"vision", "radar"})
        if invalid_modalities or not self.load_modalities:
            raise ValueError(
                "load_modalities must contain one or both of {'vision', 'radar'}, "
                f"got {sorted(self.load_modalities)}."
            )
        self.radar_norm_mode = str(radar_norm_mode).strip()
        if self.radar_norm_mode not in RADAR_NORM_MODES:
            raise ValueError(f"Unsupported radar_norm_mode: {self.radar_norm_mode}")
        self.radar_norm_stats_path = Path(radar_norm_stats).expanduser().resolve() if radar_norm_stats else None
        self.radar_norm_stats = None
        if "radar" in self.load_modalities and self.radar_norm_mode == "global_stats":
            if self.radar_norm_stats_path is None or not self.radar_norm_stats_path.exists():
                raise FileNotFoundError(
                    "global_stats mode requires --radar-norm-stats JSON file. "
                    f"Got: {self.radar_norm_stats_path}"
                )
            payload = json.loads(self.radar_norm_stats_path.read_text(encoding="utf-8"))
            maps = payload.get("maps", {})
            missing = [k for k in MAP_KEYS if k not in maps]
            if missing:
                raise ValueError(f"radar norm stats missing map keys: {missing}")
            self.radar_norm_stats = maps
        self._warned_precomputed_missing = False
        self._radar_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self.sample_format = str(sample_format).strip().lower()
        if self.sample_format not in {"deepsense", "actual"}:
            raise ValueError(f"Unsupported sample_format: {sample_format}")
        self.scenarios = (
            [self._normalize_scenario_name(s) for s in scenarios]
            if self.sample_format == "deepsense"
            else [str(s).strip() for s in scenarios if str(s).strip()]
        )
        if not self.scenarios:
            raise ValueError("At least one scenario must be provided.")
        self.csv_paths = []
        for scenario in self.scenarios:
            csv_path = (
                self._resolve_scenario_csv(scenario)
                if self.sample_format == "deepsense"
                else self.root / scenario / "multimodal_index.csv"
            )
            if csv_path is None or not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV not found for {scenario}. Expected one of: "
                    f"{(self.root / scenario / f'{scenario.lower()}_dev.csv')}, "
                    f"{(self.root / scenario / 'scenario_dev.csv')}, "
                    f"or any '*_dev.csv' under {(self.root / scenario)}."
                )
            self.csv_paths.append(csv_path)

        self.samples = []
        for csv_path in self.csv_paths:
            scenario = csv_path.parent.name
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                required_cols = (
                    {"unit1_rgb", "unit1_radar", "unit1_beam"}
                    if self.sample_format == "deepsense"
                    else {"index", "scene", "camera_path", "radar_path", "best_beam_index"}
                )
                if reader.fieldnames is None:
                    raise ValueError(f"CSV has no header: {csv_path}")
                missing = required_cols.difference(reader.fieldnames)
                if missing:
                    raise ValueError(f"Missing CSV columns in {csv_path}: {sorted(missing)}")

                row_id = 0
                for row in reader:
                    image_col = "unit1_rgb" if self.sample_format == "deepsense" else "camera_path"
                    radar_col = "unit1_radar" if self.sample_format == "deepsense" else "radar_path"
                    image_path = (csv_path.parent / str(row[image_col])).resolve()
                    radar_path = (csv_path.parent / str(row[radar_col])).resolve()
                    power_rel_text = str(row.get("unit1_pwr_60ghz", "")).strip()
                    power_path = (csv_path.parent / power_rel_text).resolve() if power_rel_text else None
                    radar_rel = Path(str(row[radar_col]))
                    if not image_path.exists() or not radar_path.exists():
                        row_id += 1
                        continue

                    label = (
                        int(float(row["unit1_beam"])) - 1
                        if self.sample_format == "deepsense"
                        else int(float(row["best_beam_index"]))
                    )
                    seq_index = row.get("seq_index", "") if self.sample_format == "deepsense" else "session"
                    time_stamp = row.get("time_stamp", "")
                    index_num = _parse_int(row.get("index", ""))
                    frame_num = _extract_first_int(row.get(image_col, ""))
                    if frame_num is None:
                        frame_num = _extract_first_int(row.get(radar_col, ""))
                    self.samples.append(
                        {
                            "image_path": image_path,
                            "radar_path": radar_path,
                            "power_path": power_path,
                            "radar_rel": radar_rel,
                            "label": label,
                            "scenario": scenario,
                            "seq_index": str(seq_index).strip() if seq_index is not None else "",
                            "time_stamp": str(time_stamp).strip() if time_stamp is not None else "",
                            "index_num": index_num,
                            "frame_num": frame_num,
                            "row_id": row_id,
                        }
                    )
                    row_id += 1

        if not self.samples:
            raise ValueError(f"No usable samples found under: {self.root} for scenarios={self.scenarios}")

    def _normalize_scenario_name(self, scenario: str) -> str:
        text = str(scenario).strip()
        if not text:
            raise ValueError("Scenario name cannot be empty.")
        if text.lower().startswith("scenario"):
            suffix = text[len("scenario") :]
        else:
            suffix = text
        if not suffix.isdigit():
            raise ValueError(f"Invalid scenario name: {scenario}")
        return f"Scenario{int(suffix)}"

    def _resolve_scenario_csv(self, scenario: str):
        scenario_dir = self.root / scenario
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        candidates = [
            scenario_dir / f"{scenario.lower()}_dev.csv",
            scenario_dir / "scenario_dev.csv",
        ]
        for p in candidates:
            if p.exists():
                return p

        fallback = sorted(scenario_dir.glob("*_dev.csv"))
        if fallback:
            return fallback[0]
        return None

    def __len__(self):
        return len(self.samples)

    def _get_precomputed_map_path(self, sample):
        if self.precomputed_radar_root is None:
            return None
        return precomputed_map_path(
            precomputed_root=self.precomputed_radar_root,
            scenario=sample["scenario"],
            radar_rel_path=sample["radar_rel"],
        )

    def _load_precomputed_maps(self, map_path):
        target_sizes = {
            "power_map": self.power_size,
            "range_angle_map": self.ra_size,
            "range_doppler_map": self.rd_size,
            "delay_doppler_map": self.dd_size,
        }
        with np.load(map_path, allow_pickle=False) as maps:
            loaded = {}
            for key in MAP_KEYS:
                if key not in maps:
                    raise KeyError(f"Missing key '{key}' in precomputed file: {map_path}")
                arr = np.asarray(maps[key], dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Expected 2D map for key '{key}' in {map_path}, got shape {arr.shape}")
                if tuple(arr.shape) != tuple(target_sizes[key]):
                    arr = _resize_2d(arr, target_sizes[key])
                loaded[key] = arr
        return loaded

    def _normalize_maps(self, radar_maps: dict[str, np.ndarray], from_precomputed: bool) -> dict[str, np.ndarray]:
        if self.radar_norm_mode == "frame_logminmax":
            if from_precomputed:
                return {k: np.asarray(v, dtype=np.float32) for k, v in radar_maps.items()}
            return {k: _log_minmax(v) for k, v in radar_maps.items()}
        if self.radar_norm_mode == "global_stats":
            if self.radar_norm_stats is None:
                raise RuntimeError("global_stats mode requires loaded radar_norm_stats.")
            return {k: _global_percentile_norm(v, self.radar_norm_stats[k]) for k, v in radar_maps.items()}
        raise ValueError(f"Unsupported radar_norm_mode: {self.radar_norm_mode}")

    def __getitem__(self, idx):
        sample = self.samples[idx]

        item = {"label": torch.tensor(sample["label"], dtype=torch.long)}
        if "vision" in self.load_modalities:
            item["image"] = _load_image(sample["image_path"], self.image_size)

        if "radar" in self.load_modalities:
            if self.cache_radar_maps and idx in self._radar_cache:
                cached = self._radar_cache.pop(idx)
                self._radar_cache[idx] = cached
                power_map, range_angle_map, range_doppler_map, delay_doppler_map = cached
            else:
                precomputed_path = self._get_precomputed_map_path(sample)
                radar_maps = None
                if precomputed_path is not None and precomputed_path.exists():
                    radar_maps = self._load_precomputed_maps(precomputed_path)
                    radar_maps = self._normalize_maps(radar_maps, from_precomputed=True)
                else:
                    if precomputed_path is not None and (not self._warned_precomputed_missing):
                        print(
                            f"[DeepSense6GDataset] Precomputed map not found, fallback to online compute: {precomputed_path}"
                        )
                        self._warned_precomputed_missing = True
                    if Path(sample["radar_path"]).suffix.lower() == ".bin":
                        radar_cube = load_actual_radar_cube(sample["radar_path"])
                        radar_maps_raw = compute_actual_radar_maps(radar_cube, n_angles=64)
                    else:
                        radar_raw = np.load(sample["radar_path"], allow_pickle=False)
                        radar_cube = canonicalize_radar(radar_raw)
                        radar_maps_raw = compute_radar_maps(radar_cube, n_angles=64, use_window=True)
                    radar_maps = preprocess_radar_maps(
                        radar_maps=radar_maps_raw,
                        power_size=self.power_size,
                        rd_size=self.rd_size,
                        ra_size=self.ra_size,
                        dd_size=self.dd_size,
                        apply_logminmax=False,
                    )
                    radar_maps = self._normalize_maps(radar_maps, from_precomputed=False)

                power_map = radar_maps["power_map"]
                range_angle_map = radar_maps["range_angle_map"]
                range_doppler_map = radar_maps["range_doppler_map"]
                delay_doppler_map = radar_maps["delay_doppler_map"]

                if self.cache_radar_maps:
                    self._radar_cache[idx] = (
                        power_map,
                        range_angle_map,
                        range_doppler_map,
                        delay_doppler_map,
                    )
                    if self.max_radar_cache_items > 0:
                        while len(self._radar_cache) > self.max_radar_cache_items:
                            self._radar_cache.popitem(last=False)

            item.update(
                {
                    "power_map": torch.from_numpy(power_map[None, :, :]).float(),
                    "range_angle_map": torch.from_numpy(range_angle_map[None, :, :]).float(),
                    "range_doppler_map": torch.from_numpy(range_doppler_map[None, :, :]).float(),
                    "delay_doppler_map": torch.from_numpy(delay_doppler_map[None, :, :]).float(),
                }
            )
        return item


class ActualMultimodalDataset(DeepSense6GDataset):
    """Camera/radar/0-based-beam dataset produced by the real acquisition system."""

    def __init__(self, *args, **kwargs):
        kwargs["sample_format"] = "actual"
        super().__init__(*args, **kwargs)


def build_base_dataset(dataset_type="deepsense", **kwargs):
    dataset_type = str(dataset_type).strip().lower()
    if dataset_type == "actual":
        return ActualMultimodalDataset(**kwargs)
    if dataset_type == "deepsense":
        return DeepSense6GDataset(**kwargs)
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


class TemporalSequenceDataset(Dataset):
    """
    Strict temporal window dataset.
    One sample contains history frames and future targets:
    - history: [t-N+1, ..., t]
    - future:  [t+1, ..., t+M]
    Sliding stride is configurable via window_stride.
    """

    def __init__(
        self,
        base_dataset: DeepSense6GDataset,
        history_len: int = 5,
        future_steps: int = 8,
        window_stride: int = 1,
        require_contiguous: bool = True,
        include_future_maps: bool = True,
        vehicle_track_json: str = "",
        max_vehicles_per_frame: int = 8,
    ):
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        if future_steps < 1:
            raise ValueError("future_steps must be >= 1.")
        if window_stride < 1:
            raise ValueError("window_stride must be >= 1.")
        if max_vehicles_per_frame < 1:
            raise ValueError("max_vehicles_per_frame must be >= 1.")
        self.base = base_dataset
        self.history_len = int(history_len)
        self.future_steps = int(future_steps)
        self.window_stride = int(window_stride)
        self.require_contiguous = bool(require_contiguous)
        self.include_future_maps = bool(include_future_maps)
        self.max_vehicles_per_frame = int(max_vehicles_per_frame)
        self.frame_detections = {}
        if vehicle_track_json:
            track_path = Path(vehicle_track_json).expanduser().resolve()
            if not track_path.exists():
                raise FileNotFoundError(f"Vehicle bbox cache not found: {track_path}")
            payload = json.loads(track_path.read_text(encoding="utf-8"))
            self.frame_detections = payload.get("frame_detections", {})
            if not self.frame_detections:
                raise ValueError(f"No frame_detections found in vehicle bbox cache: {track_path}")

        grouped = defaultdict(list)
        for base_idx, sample in enumerate(self.base.samples):
            scenario = str(sample.get("scenario", "")).strip()
            seq_index = str(sample.get("seq_index", "")).strip()
            if not scenario or not seq_index:
                continue
            group_key = f"{scenario}::{seq_index}"
            grouped[group_key].append((base_idx, sample))

        self.windows = []
        self.group_to_window_indices: dict[str, list[int]] = defaultdict(list)
        for group_key in sorted(grouped.keys()):
            entries = sorted(
                grouped[group_key],
                key=lambda p: _sample_sort_key(p[1], p[1].get("row_id", p[0])),
            )
            n = len(entries)
            if n < self.history_len + self.future_steps:
                continue
            min_center = self.history_len - 1
            max_center = n - self.future_steps - 1
            for center in range(min_center, max_center + 1, self.window_stride):
                start = center - self.history_len + 1
                end = center + self.future_steps
                window_entries = entries[start : end + 1]
                if self.require_contiguous:
                    order_index = [_window_order_index(item[1]) for item in window_entries]
                    if any(v is None for v in order_index):
                        continue
                    if any(order_index[i + 1] - order_index[i] != 1 for i in range(len(order_index) - 1)):
                        continue
                else:
                    order_index = [_window_order_index(item[1]) for item in window_entries]
                hist_indices = [item[0] for item in window_entries[: self.history_len]]
                fut_indices = [item[0] for item in window_entries[self.history_len :]]
                first_sample = window_entries[0][1]
                last_sample = window_entries[-1][1]
                window = {
                    "group_key": group_key,
                    "scenario": str(first_sample.get("scenario", "")).strip(),
                    "seq_index": str(first_sample.get("seq_index", "")).strip(),
                    "center_pos": center,
                    "hist_indices": hist_indices,
                    "future_indices": fut_indices,
                    "start_index_num": _window_order_index(first_sample),
                    "end_index_num": _window_order_index(last_sample),
                    "center_index_num": _window_order_index(window_entries[self.history_len - 1][1]),
                    "order_index": order_index,
                }
                wid = len(self.windows)
                self.windows.append(window)
                self.group_to_window_indices[group_key].append(wid)

        if not self.windows:
            raise ValueError(
                "No valid temporal windows found. "
                f"Check history_len={self.history_len}, future_steps={self.future_steps}, "
                f"window_stride={self.window_stride}, "
                "seq_index availability, and index continuity."
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        hist_samples = [self.base[i] for i in window["hist_indices"]]
        fut_samples = [self.base[i] for i in window["future_indices"]] if self.include_future_maps else None

        item = {
            "future_beam_labels": torch.tensor(
                [int(self.base.samples[i]["label"]) for i in window["future_indices"]],
                dtype=torch.long,
            ),
            "meta_start_index": torch.tensor(
                int(window["start_index_num"]) if window["start_index_num"] is not None else -1,
                dtype=torch.long,
            ),
            "meta_end_index": torch.tensor(
                int(window["end_index_num"]) if window["end_index_num"] is not None else -1,
                dtype=torch.long,
            ),
            "meta_window_id": torch.tensor(int(idx), dtype=torch.long),
            "meta_center_pos": torch.tensor(int(window["center_pos"]), dtype=torch.long),
            "meta_scenario": window["scenario"],
            "meta_seq_index": window["seq_index"],
            "meta_group_key": window["group_key"],
        }
        if "image" in hist_samples[0]:
            item["hist_image"] = torch.stack([s["image"] for s in hist_samples], dim=0)
        for key in MAP_KEYS:
            if key in hist_samples[0]:
                item[f"hist_{key}"] = torch.stack([s[key] for s in hist_samples], dim=0)
            if self.include_future_maps and fut_samples is not None and key in fut_samples[0]:
                item[f"future_{key}"] = torch.stack([s[key] for s in fut_samples], dim=0)
        if self.frame_detections:
            frame_boxes = []
            for base_idx in window["hist_indices"]:
                sample = self.base.samples[base_idx]
                frame_key = f"{sample['scenario']}::{sample['seq_index']}::{sample['index_num']}"
                detections = list(self.frame_detections.get(frame_key, []))[: self.max_vehicles_per_frame]
                detections.extend([[0.0] * 6 for _ in range(self.max_vehicles_per_frame - len(detections))])
                frame_boxes.append(detections)
            item["hist_vehicle_bboxes"] = torch.tensor(frame_boxes, dtype=torch.float32)
        return item


def build_random_segment_plan(
    temporal_dataset: TemporalSequenceDataset,
    segments_per_group: int = 3,
    segment_len: int = 20,
    seed: int = 42,
):
    if segments_per_group < 1:
        raise ValueError("segments_per_group must be >= 1.")
    if segment_len < 1:
        raise ValueError("segment_len must be >= 1.")

    rng = random.Random(seed)
    segments = []
    group_summary = {}
    skipped_groups_short = []

    for group_key in sorted(temporal_dataset.group_to_window_indices.keys()):
        group_window_ids = list(temporal_dataset.group_to_window_indices[group_key])
        n = len(group_window_ids)
        if n <= 0:
            continue

        if n < int(segment_len):
            skipped_groups_short.append(group_key)
            group_summary[group_key] = {
                "total_windows": int(n),
                "effective_segment_len": 0,
                "requested_segments": int(segments_per_group),
                "selected_segments": 0,
                "skipped_short_group": True,
            }
            continue

        eff_len = int(segment_len)
        max_start = n - eff_len
        candidate_starts = list(range(max_start + 1))
        rng.shuffle(candidate_starts)
        target_segments = min(int(segments_per_group), len(candidate_starts))

        chosen_ranges = []
        for start_pos in candidate_starts:
            end_pos = start_pos + eff_len - 1
            if all(end_pos < s0 or start_pos > e0 for s0, e0 in chosen_ranges):
                chosen_ranges.append((start_pos, end_pos))
                if len(chosen_ranges) >= target_segments:
                    break

        # Fill remaining slots (allow overlap) when disjoint sampling is insufficient.
        if len(chosen_ranges) < target_segments:
            chosen_starts = {s for s, _ in chosen_ranges}
            for start_pos in candidate_starts:
                if start_pos in chosen_starts:
                    continue
                chosen_ranges.append((start_pos, start_pos + eff_len - 1))
                if len(chosen_ranges) >= target_segments:
                    break

        chosen_ranges = sorted(chosen_ranges, key=lambda x: x[0])
        for seg_id, (start_pos, end_pos) in enumerate(chosen_ranges):
            segment_window_ids = group_window_ids[start_pos : end_pos + 1]
            first_window = temporal_dataset.windows[segment_window_ids[0]]
            last_window = temporal_dataset.windows[segment_window_ids[-1]]

            segment = {
                "group_key": group_key,
                "scenario": first_window["scenario"],
                "seq_index": first_window["seq_index"],
                "segment_id_in_group": int(seg_id),
                "start_pos": int(start_pos),
                "end_pos": int(end_pos),
                "length": int(len(segment_window_ids)),
                "start_window_id": int(segment_window_ids[0]),
                "end_window_id": int(segment_window_ids[-1]),
                "start_center_pos": int(first_window["center_pos"]),
                "end_center_pos": int(last_window["center_pos"]),
                "start_center_index_num": first_window.get("center_index_num"),
                "end_center_index_num": last_window.get("center_index_num"),
                "window_ids": [int(x) for x in segment_window_ids],
            }
            segments.append(segment)

        group_summary[group_key] = {
            "total_windows": int(n),
            "effective_segment_len": int(eff_len),
            "requested_segments": int(segments_per_group),
            "selected_segments": int(len(chosen_ranges)),
            "skipped_short_group": False,
        }

    return {
        "seed": int(seed),
        "segments_per_group": int(segments_per_group),
        "segment_len": int(segment_len),
        "history_len": int(temporal_dataset.history_len),
        "future_steps": int(temporal_dataset.future_steps),
        "window_stride": int(temporal_dataset.window_stride),
        "num_groups": int(len(group_summary)),
        "num_segments": int(len(segments)),
        "skipped_groups_short": skipped_groups_short,
        "num_skipped_groups_short": int(len(skipped_groups_short)),
        "groups": group_summary,
        "segments": segments,
    }


def segment_plan_to_window_indices(segment_plan: dict) -> list[int]:
    if "segments" not in segment_plan:
        raise ValueError("Segment plan missing key: segments")
    indices = []
    for seg in segment_plan["segments"]:
        for wid in seg.get("window_ids", []):
            indices.append(int(wid))
    if not indices:
        raise ValueError("Segment plan produced empty window indices.")
    return indices


def split_temporal_indices_chrono(
    temporal_dataset: TemporalSequenceDataset,
    val_ratio: float,
):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"Invalid val_ratio: {val_ratio}. Expected 0 < val_ratio < 1.")

    train_idx = []
    val_idx = []
    for group_key in sorted(temporal_dataset.group_to_window_indices.keys()):
        indices = temporal_dataset.group_to_window_indices[group_key]
        n = len(indices)
        if n == 0:
            continue
        if n == 1:
            train_idx.extend(indices)
            continue
        val_count = int(round(n * val_ratio))
        val_count = min(max(val_count, 1), n - 1)
        train_count = n - val_count
        train_idx.extend(indices[:train_count])
        val_idx.extend(indices[train_count:])

    if not train_idx or not val_idx:
        raise ValueError(
            "Chronological split produced empty train/val subset. "
            "Please check temporal window counts or adjust val_ratio."
        )
    return train_idx, val_idx


def build_dataloader(
    root="/home/ybpeng/Data/deepsense",
    scenarios=("Scenario31",),
    image_size=(224, 224),
    power_size=(32, 64),
    rd_size=(32, 64),
    ra_size=(64, 64),
    dd_size=(32, 64),
    precomputed_radar_root="",
):
    dataset = DeepSense6GDataset(
        root=root,
        scenarios=scenarios,
        image_size=image_size,
        power_size=power_size,
        rd_size=rd_size,
        ra_size=ra_size,
        dd_size=dd_size,
        precomputed_radar_root=precomputed_radar_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


if __name__ == "__main__":
    loader, dataset = build_dataloader(scenarios=("Scenario31", "Scenario33"))
    batch = next(iter(loader))
    print(f"CSVs: {[str(p) for p in dataset.csv_paths]}")
    print(f"Scenarios: {dataset.scenarios}")
    print(f"Usable samples: {len(dataset)}")
    print(f"image:             {tuple(batch['image'].shape)}")
    print(f"power_map:         {tuple(batch['power_map'].shape)}")
    print(f"range_angle_map:   {tuple(batch['range_angle_map'].shape)}")
    print(f"range_doppler_map: {tuple(batch['range_doppler_map'].shape)}")
    print(f"delay_doppler_map: {tuple(batch['delay_doppler_map'].shape)}")
    print(f"label:             {tuple(batch['label'].shape)}")
