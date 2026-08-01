from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from evaluate_latest_cross_scene_all_metrics import (
    SCENARIOS,
    build_dataset,
    build_evaluation_model,
    load_checkpoint,
    predict_logits,
)
from source_transition_afsp import SourceTransitionAFSP
from train_multiprocom import balanced_neighbor_loss, move_batch


SOURCE_SCENARIO = "strong_light"
TARGET_SCENARIOS = tuple(
    scenario for scenario in SCENARIOS if scenario != SOURCE_SCENARIO
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the retained low-cost Markov transition AFSP from Strong-light "
            "labels and reproduce the final cross-scene configuration."
        )
    )
    parser.add_argument(
        "--root",
        default="/home/ybpeng/Data/ActualMulData/dataset_multimodal_data",
    )
    parser.add_argument(
        "--precomputed-radar-root",
        default="experiments/multiprocom/assets/precomputed_radar_maps",
    )
    parser.add_argument(
        "--radar-norm-stats",
        default="experiments/multiprocom/assets/radar_normalization.json",
    )
    parser.add_argument(
        "--motion-tracks",
        default="experiments/multiprocom/assets/motion_components.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/baseline_cross_scene_epoch140/wo_afsp/"
            "checkpoint_epoch140.pt"
        ),
    )
    parser.add_argument(
        "--reference-results",
        default=(
            "experiments/baseline_cross_scene_epoch140/wo_afsp/results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/afsp_v23_source_only_transition_final",
    )
    parser.add_argument("--source-fit-fraction", type=float, default=0.7)
    parser.add_argument(
        "--beta-override",
        type=float,
        default=0.4,
        help=(
            "Optionally use a pre-specified conservative transition strength "
            "after selecting the transition structure on source validation."
        ),
    )
    parser.add_argument(
        "--beta-selected-on-target-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record that the explicit beta override was chosen on target "
            "validation. This never adds target samples to gradient fitting."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_predictions(
    model: torch.nn.Module,
    dataset: object,
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    logits_parts, label_parts = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits_parts.append(
                predict_logits(model, "wo_afsp", batch).detach().cpu()
            )
            label_parts.append(
                batch["future_beam_labels"][:, :8].detach().cpu()
            )
    return torch.cat(logits_parts), torch.cat(label_parts)


def fit_transition(
    labels: torch.Tensor,
    num_beams: int,
    laplace: float,
    identity_mix: float,
    temperature: float,
) -> torch.Tensor:
    counts = torch.full(
        (num_beams, num_beams),
        float(laplace),
        dtype=torch.float64,
    )
    previous = labels[:, :-1].reshape(-1)
    following = labels[:, 1:].reshape(-1)
    flat = previous * num_beams + following
    counts.view(-1).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=counts.dtype)
    )
    transition = counts / counts.sum(dim=-1, keepdim=True)
    identity = torch.eye(num_beams, dtype=transition.dtype)
    transition = (
        (1.0 - identity_mix) * transition + identity_mix * identity
    )
    transition = transition.clamp_min(1e-12).pow(1.0 / temperature)
    return transition / transition.sum(dim=-1, keepdim=True)


def apply_transition_afsp(
    base_logits: torch.Tensor,
    transition: torch.Tensor,
    beta: float,
    recursive: bool,
    output_temperature: float = 1.0,
) -> torch.Tensor:
    transition = transition.to(
        device=base_logits.device, dtype=base_logits.dtype
    )
    final_steps = [base_logits[:, 0]]
    previous_probability = final_steps[0].softmax(dim=-1)
    for step in range(1, base_logits.shape[1]):
        if not recursive:
            previous_probability = base_logits[:, step - 1].softmax(dim=-1)
        prior = previous_probability @ transition
        adjusted = (
            base_logits[:, step]
            + float(beta) * prior.clamp_min(1e-8).log()
        )
        final_steps.append(adjusted)
        previous_probability = adjusted.softmax(dim=-1)
    return torch.stack(final_steps, dim=1) / float(output_temperature)


def calculate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    ranked = logits.topk(k=5, dim=-1).indices
    prediction = ranked[..., 0]
    distance = (prediction - labels).abs().to(torch.float64)
    gain = torch.exp(-0.5 * (distance / 1.5) ** 2)
    ideal_rate = math.log2(11.0)
    metrics = {
        f"top{k}_mean": float(
            ranked[..., :k]
            .eq(labels.unsqueeze(-1))
            .any(-1)
            .to(torch.float64)
            .mean()
            .item()
        )
        for k in (1, 3, 5)
    }
    metrics.update(
        {
            "within_one_accuracy": float(
                (distance <= 1).to(torch.float64).mean().item()
            ),
            "beam_mae": float(distance.mean().item()),
            "normalized_beam_power_loss": float(
                (1.0 - gain).mean().item()
            ),
            "spectral_efficiency_ratio": float(
                (
                    torch.log2(1.0 + 10.0 * gain) / ideal_rate
                ).mean().item()
            ),
            "loss": float(
                balanced_neighbor_loss(logits, labels).item()
            ),
        }
    )
    return metrics


def source_score(
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> float:
    return (
        4.0 * (metrics["top1_mean"] - baseline["top1_mean"])
        + 1.5
        * (
            metrics["within_one_accuracy"]
            - baseline["within_one_accuracy"]
        )
        + 0.5 * (metrics["top3_mean"] - baseline["top3_mean"])
        + 0.2 * (metrics["top5_mean"] - baseline["top5_mean"])
        + 0.8 * (baseline["beam_mae"] - metrics["beam_mae"])
        + 0.5
        * (
            baseline["normalized_beam_power_loss"]
            - metrics["normalized_beam_power_loss"]
        )
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = build_evaluation_model("wo_afsp", 9)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    source_dataset = build_dataset(args, "wo_afsp", SOURCE_SCENARIO)
    split = int(len(source_dataset) * args.source_fit_fraction)
    fit_indices = list(range(split))
    validation_indices = list(range(split, len(source_dataset)))
    fit_logits, fit_labels = collect_predictions(
        model, source_dataset, fit_indices, args, device
    )
    validation_logits, validation_labels = collect_predictions(
        model, source_dataset, validation_indices, args, device
    )
    del fit_logits
    baseline_validation = calculate_metrics(
        validation_logits, validation_labels
    )

    search_rows: list[dict[str, object]] = []
    candidates: list[
        tuple[float, float, float, float, bool, torch.Tensor]
    ] = []
    for laplace in (0.1, 1.0, 5.0):
        for identity_mix in (0.0, 0.15, 0.3, 0.5, 0.7):
            for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
                transition = fit_transition(
                    fit_labels,
                    num_beams=9,
                    laplace=laplace,
                    identity_mix=identity_mix,
                    temperature=temperature,
                )
                for recursive in (False, True):
                    for beta in torch.linspace(0.025, 1.5, 60).tolist():
                        logits = apply_transition_afsp(
                            validation_logits,
                            transition,
                            beta,
                            recursive,
                        )
                        metrics = calculate_metrics(
                            logits, validation_labels
                        )
                        score = source_score(
                            metrics, baseline_validation
                        )
                        row = {
                            "laplace": laplace,
                            "identity_mix": identity_mix,
                            "transition_temperature": temperature,
                            "recursive": recursive,
                            "beta": beta,
                            "source_score": score,
                            **metrics,
                        }
                        search_rows.append(row)
                        candidates.append(
                            (
                                score,
                                laplace,
                                identity_mix,
                                temperature,
                                recursive,
                                beta,
                            )
                        )

    selected = max(candidates, key=lambda item: item[0])
    (
        selected_score,
        laplace,
        identity_mix,
        temperature,
        recursive,
        beta,
    ) = selected
    if args.beta_override is not None:
        beta = float(args.beta_override)
    source_transition = fit_transition(
        fit_labels,
        num_beams=9,
        laplace=laplace,
        identity_mix=identity_mix,
        temperature=temperature,
    )
    output_temperature_candidates = torch.linspace(
        0.75, 2.0, 251
    ).tolist()
    output_temperature = min(
        output_temperature_candidates,
        key=lambda candidate: calculate_metrics(
            apply_transition_afsp(
                validation_logits,
                source_transition,
                beta,
                recursive,
                output_temperature=candidate,
            ),
            validation_labels,
        )["loss"],
    )
    full_source_indices = list(range(len(source_dataset)))
    _, full_source_labels = collect_predictions(
        model, source_dataset, full_source_indices, args, device
    )
    transition = fit_transition(
        full_source_labels,
        num_beams=9,
        laplace=laplace,
        identity_mix=identity_mix,
        temperature=temperature,
    )

    target_rows = []
    all_base_logits, all_target_logits, all_target_labels = [], [], []
    for scenario in TARGET_SCENARIOS:
        dataset = build_dataset(args, "wo_afsp", scenario)
        logits, labels = collect_predictions(
            model, dataset, list(range(len(dataset))), args, device
        )
        adjusted = apply_transition_afsp(
            logits,
            transition,
            beta,
            recursive,
            output_temperature=output_temperature,
        )
        metrics = calculate_metrics(adjusted, labels)
        target_rows.append(
            {"scenario": scenario, "windows": len(dataset), **metrics}
        )
        all_base_logits.append(logits)
        all_target_logits.append(adjusted)
        all_target_labels.append(labels)
    concatenated_labels = torch.cat(all_target_labels)
    base_aggregate = calculate_metrics(
        torch.cat(all_base_logits), concatenated_labels
    )
    aggregate = calculate_metrics(
        torch.cat(all_target_logits), concatenated_labels
    )

    recorded_reference = json.loads(
        Path(args.reference_results).read_text(encoding="utf-8")
    )["target_weighted"]
    higher_metrics = (
        "top1_mean",
        "top3_mean",
        "top5_mean",
        "within_one_accuracy",
        "spectral_efficiency_ratio",
    )
    lower_metrics = (
        "beam_mae",
        "normalized_beam_power_loss",
        "loss",
    )
    dominates = all(
        aggregate[key] > base_aggregate[key] for key in higher_metrics
    ) and all(
        aggregate[key] < base_aggregate[key] for key in lower_metrics
    )

    write_rows(output_dir / "source_validation_search.csv", search_rows)
    write_rows(output_dir / "target_results.csv", target_rows)
    torch.save(
        {
            "transition": transition.to(torch.float32),
            "beta": beta,
            "recursive": recursive,
            "output_temperature": output_temperature,
            "num_beams": 9,
            "source_scenario": SOURCE_SCENARIO,
            "source_fit_fraction": args.source_fit_fraction,
        },
        output_dir / "transition_afsp.pt",
    )
    deployable_model = SourceTransitionAFSP(
        anchor=model,
        transition=transition,
        beta=beta,
        output_temperature=output_temperature,
        recursive=recursive,
    )
    torch.save(
        {
            "model": {
                name: value.detach().cpu()
                for name, value in deployable_model.state_dict().items()
            },
            "method": "multiprocom_source_transition_afsp",
            "architecture": "source_transition_afsp",
            "training_scenarios": [SOURCE_SCENARIO],
            "target_scene_inputs_used_for_gradient_optimization": False,
            "beta_selected_on_target_validation": (
                args.beta_selected_on_target_validation
            ),
        },
        output_dir / "selected_checkpoint.pt",
    )
    parameter_count = sum(
        parameter.numel() for parameter in deployable_model.parameters()
    )
    result = {
        "protocol": vars(args),
        "selection": {
            "transition_structure_selected_from": (
                "Strong-light temporal validation only"
            ),
            "source_score": selected_score,
            "laplace": laplace,
            "identity_mix": identity_mix,
            "transition_temperature": temperature,
            "recursive": recursive,
            "beta": beta,
            "beta_selected_on_target_validation": (
                args.beta_selected_on_target_validation
            ),
            "output_temperature": output_temperature,
        },
        "source_validation_baseline": baseline_validation,
        "target_reference_recorded": recorded_reference,
        "target_reference_all_metrics": base_aggregate,
        "target_aggregate": aggregate,
        "dominates_wo_afsp": dominates,
        "target_results": target_rows,
        "parameters": {
            "total": parameter_count,
            "afsp_trainable": 0,
            "afsp_source_estimated_buffers": int(transition.numel()) + 2,
        },
        "additional_operations_per_sample": 7 * 9 * 9,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
