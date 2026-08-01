from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset

from actual_data_utils import ACTUAL_SCENARIOS
from dataset import ActualMultimodalDataset, TemporalSequenceDataset
from multiprocom import MultiProCom

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def rss_mb() -> float:
    if psutil is not None:
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0**2)
    # ru_maxrss is KB on Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def cpu_times() -> tuple[float, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime), float(usage.ru_stime)


def parse_gpu_ids(text: str) -> list[int]:
    ids = []
    for token in text.split(','):
        token = token.strip()
        if token:
            ids.append(int(token))
    return ids


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def model_forward(model: torch.nn.Module, batch: Dict[str, Any], pred_horizon: int) -> Dict[str, torch.Tensor]:
    return model(
        hist_vehicle_bboxes=batch['hist_vehicle_bboxes'],
        hist_range_doppler_map=batch['hist_range_doppler_map'],
        hist_range_angle_map=batch['hist_range_angle_map'],
        hist_delay_doppler_map=batch['hist_delay_doppler_map'],
        hist_power_map=batch['hist_power_map'],
        pred_horizon=pred_horizon,
        target_tokens=None,
        teacher_forcing=False,
    )


def safe_cuda_sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def query_nvidia_smi() -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw',
                '--format=csv,noheader,nounits',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 7:
            rows.append(
                {
                    'index': parts[0],
                    'name': parts[1],
                    'memory_total_mb': parts[2],
                    'memory_used_mb': parts[3],
                    'gpu_util_percent': parts[4],
                    'memory_util_percent': parts[5],
                    'power_w': parts[6],
                }
            )
    return rows


def estimate_flops(model: torch.nn.Module, batch: Dict[str, Any], pred_horizon: int, device: torch.device) -> dict[str, Any]:
    # torch.profiler reports FLOPs for supported aten ops (mainly conv/mm/addmm/bmm).
    # It is therefore a conservative approximate count for the current PyTorch graph.
    try:
        with torch.inference_mode():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if device.type == 'cuda' else []),
                with_flops=True,
                profile_memory=True,
                record_shapes=False,
            ) as prof:
                _ = model_forward(model, batch, pred_horizon)
                safe_cuda_sync(device)
        total_flops = 0
        by_op = []
        for evt in prof.key_averages():
            flops = int(getattr(evt, 'flops', 0) or 0)
            if flops > 0:
                total_flops += flops
                by_op.append((evt.key, flops))
        by_op = sorted(by_op, key=lambda x: x[1], reverse=True)[:20]
        return {
            'status': 'ok',
            'flops_per_batch': int(total_flops),
            'flops_per_sample': float(total_flops) / float(batch['hist_vehicle_bboxes'].shape[0]),
            'top_ops': [{'op': k, 'flops': int(v)} for k, v in by_op],
            'note': 'Approximate PyTorch profiler FLOPs for supported operations; unsupported ops are not counted.',
        }
    except Exception as exc:
        return {'status': 'failed', 'error': repr(exc)}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return float(xs[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile MultiProCom inference cost on real temporal windows.')
    parser.add_argument('--root', default='/home/ybpeng/Data/ActualMulData/dataset_multimodal_data')
    parser.add_argument('--precomputed-radar-root', default='experiments/multiprocom/assets/precomputed_radar_maps')
    parser.add_argument('--radar-norm-stats', default='experiments/multiprocom/assets/radar_normalization.json')
    parser.add_argument('--motion-components', default='experiments/multiprocom/assets/motion_components.json')
    parser.add_argument('--checkpoint', default='experiments/multiprocom/checkpoints/multiprocom.pt')
    parser.add_argument('--output-json', default='experiments/multiprocom/complexity/profile_multiprocom.json')
    parser.add_argument('--output-md', default='experiments/multiprocom/complexity/profile_multiprocom.md')
    parser.add_argument('--history-len', type=int, default=5)
    parser.add_argument('--pred-horizon-max', type=int, default=8)
    parser.add_argument('--pred-horizon', type=int, default=8)
    parser.add_argument('--num-beams', type=int, default=9)
    parser.add_argument('--feat-dim', type=int, default=256)
    parser.add_argument('--max-vehicles-per-frame', type=int, default=8)
    parser.add_argument('--max-vision-weight', type=float, default=0.35)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--warmup-iters', type=int, default=20)
    parser.add_argument('--profile-iters', type=int, default=100)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--gpu-ids', default='', help='Optional DataParallel GPU ids within current CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--flops-batch-size', type=int, default=1, help='Use a small batch for profiler FLOPs to reduce overhead.')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False.')

    start_rss = rss_mb()
    scenario_list = list(ACTUAL_SCENARIOS)
    base = ActualMultimodalDataset(
        root=args.root,
        scenarios=scenario_list,
        precomputed_radar_root=args.precomputed_radar_root,
        radar_norm_mode='global_stats',
        radar_norm_stats=args.radar_norm_stats,
        load_modalities=('radar',),
    )
    temporal = TemporalSequenceDataset(
        base_dataset=base,
        history_len=args.history_len,
        future_steps=args.pred_horizon_max,
        window_stride=1,
        require_contiguous=True,
        include_future_maps=False,
        vehicle_track_json=args.motion_components,
        max_vehicles_per_frame=args.max_vehicles_per_frame,
    )
    dataset_rss = rss_mb()

    indices = list(range(min(len(temporal), max(args.batch_size, args.flops_batch_size))))
    loader = DataLoader(
        Subset(temporal, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    batch_cpu = next(iter(loader))

    model = MultiProCom(
        feat_dim=args.feat_dim,
        num_beams=args.num_beams,
        history_len=args.history_len,
        pred_horizon_max=args.pred_horizon_max,
        max_vision_weight=args.max_vision_weight,
    )
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[Profile] load_state_dict missing={len(missing)} unexpected={len(unexpected)}')

    param_total = sum(p.numel() for p in model.parameters())
    param_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_total = sum(b.numel() for b in model.buffers())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    if args.gpu_ids.strip():
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    else:
        gpu_ids = []
    if device.type == 'cuda':
        torch.cuda.set_device(0)
    model = model.to(device).eval()
    if device.type == 'cuda' and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    model_rss = rss_mb()

    batch = move_batch(batch_cpu, device)
    batch_size = int(batch['hist_vehicle_bboxes'].shape[0])
    input_shapes = {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v) and k.startswith('hist_')}

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    safe_cuda_sync(device)
    before_alloc = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    before_reserved = torch.cuda.memory_reserved(device) if device.type == 'cuda' else 0

    with torch.inference_mode():
        for _ in range(max(0, args.warmup_iters)):
            _ = model_forward(model, batch, args.pred_horizon)
    safe_cuda_sync(device)

    latencies_ms = []
    cpu_user0, cpu_sys0 = cpu_times()
    wall0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(1, args.profile_iters)):
            t0 = time.perf_counter()
            _ = model_forward(model, batch, args.pred_horizon)
            safe_cuda_sync(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
    wall1 = time.perf_counter()
    cpu_user1, cpu_sys1 = cpu_times()

    after_rss = rss_mb()
    gpu_peak_alloc = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    gpu_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    after_alloc = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    after_reserved = torch.cuda.memory_reserved(device) if device.type == 'cuda' else 0

    # Use a separate small batch for FLOPs profiling to keep profiler overhead manageable.
    flops_n = min(args.flops_batch_size, batch_size)
    flops_batch = {k: (v[:flops_n] if torch.is_tensor(v) and v.shape[0] >= flops_n else v) for k, v in batch.items()}
    flops = estimate_flops(model, flops_batch, args.pred_horizon, device)

    wall_s = wall1 - wall0
    cpu_s = (cpu_user1 - cpu_user0) + (cpu_sys1 - cpu_sys0)
    throughput_samples_s = (float(batch_size) * float(max(1, args.profile_iters))) / max(wall_s, 1e-9)
    latency_batch_ms_mean = statistics.mean(latencies_ms)
    latency_batch_ms_std = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    latency_sample_ms_mean = latency_batch_ms_mean / float(batch_size)

    result: Dict[str, Any] = {
        'profile_scope': 'current MultiProCom forward pass with bbox-track vision encoder, four-branch radar encoder, RAMF fusion, and AFSP decoder',
        'checkpoint': str(ckpt_path),
        'device': str(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'dataset': {
            'type': 'actual_multimodal',
            'root': str(Path(args.root).expanduser()),
            'scenarios': scenario_list,
            'temporal_windows': len(temporal),
            'history_len': args.history_len,
            'pred_horizon_max': args.pred_horizon_max,
            'pred_horizon_profiled': args.pred_horizon,
            'window_stride': 1,
            'input_shapes': input_shapes,
        },
        'parameters': {
            'total': int(param_total),
            'trainable': int(param_trainable),
            'buffers': int(buffer_total),
            'parameter_size_mb_fp32': param_bytes / (1024.0**2),
            'buffer_size_mb_fp32': buffer_bytes / (1024.0**2),
            'checkpoint_file_mb': ckpt_path.stat().st_size / (1024.0**2),
        },
        'flops': flops,
        'latency': {
            'batch_size': batch_size,
            'warmup_iters': args.warmup_iters,
            'profile_iters': args.profile_iters,
            'batch_ms_mean': latency_batch_ms_mean,
            'batch_ms_std': latency_batch_ms_std,
            'batch_ms_p50': percentile(latencies_ms, 50),
            'batch_ms_p90': percentile(latencies_ms, 90),
            'batch_ms_p95': percentile(latencies_ms, 95),
            'batch_ms_min': min(latencies_ms),
            'batch_ms_max': max(latencies_ms),
            'sample_ms_mean': latency_sample_ms_mean,
            'throughput_samples_per_s': throughput_samples_s,
        },
        'cpu': {
            'process_rss_mb_start': start_rss,
            'process_rss_mb_after_dataset': dataset_rss,
            'process_rss_mb_after_model': model_rss,
            'process_rss_mb_after_profile': after_rss,
            'process_rss_mb_delta_total': after_rss - start_rss,
            'cpu_time_s_total_profile_loop': cpu_s,
            'cpu_time_ms_per_sample_profile_loop': (cpu_s * 1000.0) / max(float(batch_size * max(1, args.profile_iters)), 1.0),
            'cpu_util_estimate_percent_of_one_core': 100.0 * cpu_s / max(wall_s, 1e-9),
        },
        'gpu': {
            'available': bool(torch.cuda.is_available()),
            'device_count_visible': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else '',
            'memory_allocated_before_forward_mb': before_alloc / (1024.0**2),
            'memory_reserved_before_forward_mb': before_reserved / (1024.0**2),
            'memory_allocated_after_profile_mb': after_alloc / (1024.0**2),
            'memory_reserved_after_profile_mb': after_reserved / (1024.0**2),
            'peak_memory_allocated_mb': gpu_peak_alloc / (1024.0**2),
            'peak_memory_reserved_mb': gpu_peak_reserved / (1024.0**2),
            'nvidia_smi_snapshot': query_nvidia_smi(),
        },
        'load_state': {'missing_keys': len(missing), 'unexpected_keys': len(unexpected)},
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    flops_sample = result['flops'].get('flops_per_sample') if result['flops'].get('status') == 'ok' else None
    flops_text = 'N/A'
    if flops_sample is not None:
        flops_text = f"{flops_sample / 1e9:.3f} GFLOPs/sample"
    md = f"""# MultiProCom Compute Profile\n\n- Checkpoint: `{result['checkpoint']}`\n- Device: `{result['device']}` ({result['gpu']['device_name'] or 'CPU'})\n- Input: N={args.history_len}, M={args.pred_horizon}, stride=1, batch={batch_size}\n- Parameters: {param_total:,} ({result['parameters']['parameter_size_mb_fp32']:.2f} MB fp32)\n- FLOPs: {flops_text}\n- Latency: {latency_sample_ms_mean:.4f} ms/sample, {latency_batch_ms_mean:.4f} ms/batch, throughput {throughput_samples_s:.2f} samples/s\n- CPU RSS after profile: {after_rss:.2f} MB\n- CPU time: {result['cpu']['cpu_time_ms_per_sample_profile_loop']:.4f} ms/sample, estimated {result['cpu']['cpu_util_estimate_percent_of_one_core']:.1f}% of one CPU core during timed loop\n- GPU peak allocated: {result['gpu']['peak_memory_allocated_mb']:.2f} MB\n- GPU peak reserved: {result['gpu']['peak_memory_reserved_mb']:.2f} MB\n\nNote: FLOPs are estimated with `torch.profiler(with_flops=True)` and count supported PyTorch operations.\n"""
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding='utf-8')

    print(json.dumps({
        'output_json': str(out_json.resolve()),
        'output_md': str(out_md.resolve()),
        'params': param_total,
        'flops_per_sample': flops_sample,
        'latency_ms_per_sample': latency_sample_ms_mean,
        'throughput_samples_per_s': throughput_samples_s,
        'cpu_rss_mb_after_profile': after_rss,
        'gpu_peak_allocated_mb': result['gpu']['peak_memory_allocated_mb'],
        'gpu_peak_reserved_mb': result['gpu']['peak_memory_reserved_mb'],
    }, indent=2))


if __name__ == '__main__':
    main()
