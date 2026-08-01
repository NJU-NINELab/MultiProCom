from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plot_repeated_system_log_metrics import (
    METHODS,
    MISSING_LOG_COLUMNS,
    discover_runs,
    read_rx,
)


OUTPUT_DIR = Path("experiments/actual_control_lead")
POST_COMMAND_GUARD_S = 0.15
EVALUATION_WINDOW_S = 1.50
MINIMUM_RX_RECORDS = 5
OUTAGE_BER_PERCENT = 50.0

TIMESTAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")
CSI_COMMAND = re.compile(
    r"\[6\. 波束指向 \(Az=[-+0-9.]+°, El=([-+0-9.]+)°\)\]"
)
CSI_BEST = re.compile(r"最佳 RX 俯仰角:\s*([-+]?\d+(?:\.\d+)?)°")
CSI_SCAN_START = re.compile(r"开始第\s+\d+\s+次 RX 俯仰角扫描")
CSI_NO_VALID = re.compile(r"本次扫描无有效 CSI")
MPC_COMMAND = re.compile(
    r"PAC控制:\s*beam=(\d+),.*?俯仰角=([-+0-9.]+)°"
)


@dataclass(frozen=True)
class AvailabilityRule:
    name: str
    minimum_availability: float
    minimum_throughput_fraction: float
    maximum_penalized_ber_percent: float


RULES = (
    AvailabilityRule("relaxed", 0.50, 0.25, 25.0),
    AvailabilityRule("moderate", 0.60, 0.40, 20.0),
    AvailabilityRule("conservative", 0.70, 0.50, 15.0),
)


def parse_control_commands(path: Path, method: str) -> pd.DataFrame:
    commands: list[dict[str, Any]] = []
    in_scan = False
    pending_best_angle: float | None = None
    pending_best_timestamp: datetime | None = None
    committed_angle: float | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp_match = TIMESTAMP.match(line)
            if timestamp_match is None:
                continue
            timestamp = datetime.fromisoformat(timestamp_match.group(1))
            if method == "MultiProCom":
                command_match = MPC_COMMAND.search(line)
                if command_match:
                    commands.append(
                        {
                            "timestamp": timestamp,
                            "angle_deg": float(command_match.group(2)),
                            "beam_index": int(command_match.group(1)),
                            "command_type": "PAC control",
                            "committed_decision": True,
                            "decision_declared_timestamp": timestamp,
                        }
                    )
                continue

            if CSI_SCAN_START.search(line):
                in_scan = True
                pending_best_angle = None
                pending_best_timestamp = None
                continue

            best_match = CSI_BEST.search(line)
            if best_match:
                pending_best_angle = float(best_match.group(1))
                pending_best_timestamp = timestamp
                in_scan = False
                continue

            command_match = CSI_COMMAND.search(line)
            if command_match:
                angle = float(command_match.group(1))
                committed = False
                command_type = "scan command" if in_scan else "initialization command"
                declaration_time: Any = pd.NaT
                if (
                    pending_best_angle is not None
                    and pending_best_timestamp is not None
                    and 0
                    <= (timestamp - pending_best_timestamp).total_seconds()
                    <= 15.0
                    and abs(angle - pending_best_angle) < 1e-9
                ):
                    committed = True
                    command_type = "selected-best control"
                    declaration_time = pending_best_timestamp
                    committed_angle = angle
                    pending_best_angle = None
                    pending_best_timestamp = None
                elif (
                    not in_scan
                    and committed_angle is not None
                    and abs(angle - committed_angle) < 1e-9
                ):
                    committed = True
                    command_type = "committed hold control"
                commands.append(
                    {
                        "timestamp": timestamp,
                        "angle_deg": angle,
                        "beam_index": math.nan,
                        "command_type": command_type,
                        "committed_decision": committed,
                        "decision_declared_timestamp": declaration_time,
                    }
                )
                continue

            if CSI_NO_VALID.search(line):
                in_scan = False
                pending_best_angle = None
                pending_best_timestamp = None

    if not commands:
        raise ValueError(f"No actual beam-control commands found in {path}")
    return pd.DataFrame(commands).sort_values("timestamp", kind="mergesort").reset_index(
        drop=True
    )


def command_candidates(
    commands: pd.DataFrame, rx_start: datetime, committed_only: bool
) -> pd.DataFrame:
    selected = commands[commands["committed_decision"]] if committed_only else commands
    selected = selected.copy()
    if selected.empty:
        return selected

    # Only the last command issued before the first RX record can determine the
    # beam state at that record; earlier pre-RX commands have already been
    # superseded.
    before = selected[selected["timestamp"] < rx_start].tail(1)
    after = selected[selected["timestamp"] >= rx_start]
    return pd.concat([before, after], ignore_index=True)


def evaluate_after_command(
    rx_data: pd.DataFrame,
    command_time: datetime,
    run_positive_throughput_median: float,
    rule: AvailabilityRule,
) -> dict[str, Any]:
    rx_start = rx_data["timestamp"].iloc[0]
    window_start = max(
        command_time + timedelta(seconds=POST_COMMAND_GUARD_S), rx_start
    )
    window_end = window_start + timedelta(seconds=EVALUATION_WINDOW_S)
    selected = rx_data[
        (rx_data["timestamp"] >= window_start) & (rx_data["timestamp"] <= window_end)
    ]
    if len(selected) < MINIMUM_RX_RECORDS:
        return {
            "stable": False,
            "record_count": int(len(selected)),
            "availability_fraction": math.nan,
            "throughput_fraction": math.nan,
            "mean_penalized_ber_percent": math.nan,
            "window_start": window_start,
            "window_end": window_end,
        }

    available = (selected["throughput_mbps"] > 0) & selected["ber_percent"].notna()
    availability = float(available.mean())
    throughput_fraction = (
        float(selected["throughput_mbps"].mean()) / run_positive_throughput_median
    )
    penalized_ber = selected["ber_percent"].fillna(OUTAGE_BER_PERCENT)
    mean_ber = float(penalized_ber.mean())
    stable = (
        availability >= rule.minimum_availability
        and throughput_fraction >= rule.minimum_throughput_fraction
        and mean_ber <= rule.maximum_penalized_ber_percent
    )
    return {
        "stable": bool(stable),
        "record_count": int(len(selected)),
        "availability_fraction": availability,
        "throughput_fraction": throughput_fraction,
        "mean_penalized_ber_percent": mean_ber,
        "window_start": window_start,
        "window_end": window_end,
    }


def first_effective_control(
    rx_data: pd.DataFrame,
    commands: pd.DataFrame,
    rule: AvailabilityRule,
    committed_only: bool,
) -> dict[str, Any]:
    rx_start = rx_data["timestamp"].iloc[0]
    positive = rx_data.loc[rx_data["throughput_mbps"] > 0, "throughput_mbps"]
    if positive.empty:
        raise ValueError("Run contains no positive-throughput RX record")
    run_median = float(positive.median())
    candidates = command_candidates(commands, rx_start, committed_only)
    for command in candidates.to_dict("records"):
        assessment = evaluate_after_command(
            rx_data, command["timestamp"], run_median, rule
        )
        if assessment["stable"]:
            return {
                "status": "observed",
                "effective_command_timestamp": command["timestamp"],
                "latency_from_first_rx_s": (
                    command["timestamp"] - rx_start
                ).total_seconds(),
                "angle_deg": command["angle_deg"],
                "beam_index": command["beam_index"],
                "command_type": command["command_type"],
                **assessment,
            }

    censor_time = min(commands["timestamp"].max(), rx_data["timestamp"].iloc[-1])
    return {
        "status": "right-censored",
        "effective_command_timestamp": pd.NaT,
        "latency_from_first_rx_s": math.nan,
        "angle_deg": math.nan,
        "beam_index": math.nan,
        "command_type": "",
        "stable": False,
        "record_count": 0,
        "availability_fraction": math.nan,
        "throughput_fraction": math.nan,
        "mean_penalized_ber_percent": math.nan,
        "window_start": pd.NaT,
        "window_end": pd.NaT,
        "censor_latency_s": (censor_time - rx_start).total_seconds(),
    }


def mean_sd(values: pd.Series) -> tuple[float, float, int]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return math.nan, math.nan, 0
    return (
        float(finite.mean()),
        float(finite.std(ddof=1)) if len(finite) > 1 else math.nan,
        int(len(finite)),
    )


def summarize_leads(leads: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in leads.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        lead_mean, lead_sd, lead_n = mean_sd(group["lead_s"])
        bound_mean, bound_sd, bound_n = mean_sd(group["lead_lower_bound_s"])
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "mean_lead_s": lead_mean,
                "sd_lead_s": lead_sd,
                "observed_pair_count": lead_n,
                "mean_censored_lower_bound_s": bound_mean,
                "sd_censored_lower_bound_s": bound_sd,
                "censored_pair_count": bound_n,
                "total_pair_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    run_results: pd.DataFrame,
    scene_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Actual-command proactive control lead",
        "",
        "The analysis uses only timestamps of commands that were actually issued by the hardware controllers. Prediction-horizon indices (t+1 to t+8) are not used.",
        "",
        "Link availability is evaluated over the 1.5-s interval beginning 0.15 s after each command. A command is effective when the interval satisfies the selected availability, throughput and outage-penalized BER thresholds. The first RX metric timestamp is the run-specific time origin.",
        "",
        "Transient CSI scan angles are excluded. A CSI-based command enters the analysis only when the controller has logged a best elevation and subsequently issued the matching hardware command. Repeated commands that maintain this selected elevation remain committed controls. MultiProCom PAC commands are direct decisions. Runs without a qualifying committed command are right-censored.",
        "",
        "## Criteria",
        "",
    ]
    for rule in RULES:
        lines.append(
            f"- {rule.name}: availability >= {100 * rule.minimum_availability:.0f}%, "
            f"throughput >= {100 * rule.minimum_throughput_fraction:.0f}% of the "
            f"run positive-throughput median, and outage-penalized BER <= "
            f"{rule.maximum_penalized_ber_percent:g}%."
        )
    lines.extend(["", "## Overall results", ""])
    for row in overall_summary.to_dict("records"):
        label = (
            f"{row['criterion']}, MultiProCom versus "
            f"{row['baseline_method']}"
        )
        if int(row["observed_pair_count"]) > 0:
            lines.append(
                f"- {label}: {row['mean_lead_s']:.3f} +/- {row['sd_lead_s']:.3f} s "
                f"(mean +/- SD, n={int(row['observed_pair_count'])})."
            )
        if int(row["censored_pair_count"]) > 0:
            lines.append(
                f"- {label}: {int(row['censored_pair_count'])} additional runs were "
                f"right-censored; their mean lead is at least "
                f"{row['mean_censored_lower_bound_s']:.3f} s."
            )
    lines.extend(["", "Multi-target is excluded from this analysis.", ""])
    (output_dir / "actual_control_lead_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, missing_groups = discover_runs(Path(args.log_root))
    event_rows: list[dict[str, Any]] = []
    command_rows: list[pd.DataFrame] = []

    for spec in specs:
        rx_data = read_rx(spec["rx_path"])
        commands = parse_control_commands(spec["opt_path"], spec["method"])
        exported = commands.copy()
        exported.insert(0, "repeat", spec["repeat"])
        exported.insert(0, "method", spec["method"])
        exported.insert(0, "scene", spec["scene"])
        exported["opt_log"] = str(spec["opt_path"])
        command_rows.append(exported)

        for rule in RULES:
            result = first_effective_control(
                rx_data, commands, rule, committed_only=True
            )
            event_rows.append(
                {
                    "criterion": rule.name,
                    "control_scope": "committed decision",
                    "scene": spec["scene"],
                    "method": spec["method"],
                    "repeat": spec["repeat"],
                    "rx_log": str(spec["rx_path"]),
                    "opt_log": str(spec["opt_path"]),
                    "first_rx_timestamp": rx_data["timestamp"].iloc[0],
                    **result,
                }
            )

    events = pd.DataFrame(event_rows)
    commands = pd.concat(command_rows, ignore_index=True)
    lead_rows: list[dict[str, Any]] = []
    for criterion in [rule.name for rule in RULES]:
        selected = events[events["criterion"] == criterion]
        for scene in selected["scene"].drop_duplicates():
            for repeat in (1, 2, 3):
                mpc_rows = selected[
                    (selected["scene"] == scene)
                    & (selected["repeat"] == repeat)
                    & (selected["method"] == "MultiProCom")
                ]
                if mpc_rows.empty:
                    continue
                mpc = mpc_rows.iloc[0]
                for baseline_method in METHODS:
                    if baseline_method == "MultiProCom":
                        continue
                    baseline_rows = selected[
                        (selected["scene"] == scene)
                        & (selected["repeat"] == repeat)
                        & (selected["method"] == baseline_method)
                    ]
                    if baseline_rows.empty:
                        continue
                    csi = baseline_rows.iloc[0]
                    both_observed = (
                        mpc["status"] == "observed"
                        and csi["status"] == "observed"
                    )
                    lower_bound = math.nan
                    if (
                        mpc["status"] == "observed"
                        and csi["status"] == "right-censored"
                    ):
                        lower_bound = float(csi["censor_latency_s"]) - float(
                            mpc["latency_from_first_rx_s"]
                        )
                    lead_rows.append(
                        {
                            "criterion": criterion,
                            "control_scope": "committed decision",
                            "scene": scene,
                            "repeat": repeat,
                            "baseline_method": baseline_method,
                            "multiprocom_status": mpc["status"],
                            "baseline_status": csi["status"],
                            "multiprocom_latency_s": mpc[
                                "latency_from_first_rx_s"
                            ],
                            "baseline_latency_s": csi["latency_from_first_rx_s"],
                            "lead_s": (
                                float(csi["latency_from_first_rx_s"])
                                - float(mpc["latency_from_first_rx_s"])
                                if both_observed
                                else math.nan
                            ),
                            "lead_lower_bound_s": lower_bound,
                        }
                    )

    leads = pd.DataFrame(lead_rows)
    scene_summary = summarize_leads(
        leads, ["criterion", "control_scope", "scene", "baseline_method"]
    )
    overall_summary = summarize_leads(
        leads, ["criterion", "control_scope", "baseline_method"]
    )

    commands.to_csv(output_dir / "parsed_actual_commands.csv", index=False)
    pd.DataFrame(missing_groups, columns=MISSING_LOG_COLUMNS).to_csv(
        output_dir / "missing_log_groups.csv", index=False
    )
    events.to_csv(
        output_dir / "first_effective_control_events.csv",
        index=False,
        float_format="%.8f",
    )
    leads.to_csv(
        output_dir / "paired_actual_control_leads.csv",
        index=False,
        float_format="%.8f",
    )
    scene_summary.to_csv(
        output_dir / "actual_control_lead_by_scene.csv",
        index=False,
        float_format="%.8f",
    )
    overall_summary.to_csv(
        output_dir / "actual_control_lead_overall.csv",
        index=False,
        float_format="%.8f",
    )
    write_report(events, scene_summary, overall_summary, output_dir)

    print("Overall actual-command lead (s)")
    print(overall_summary.to_string(index=False))
    print(f"\nOutputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
