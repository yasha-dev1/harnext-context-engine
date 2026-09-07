"""Light, consistent PNG charts for docs/evaluation-spec.md §7 outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

_COLORS = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0f172a",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "font.size": 10,
        "legend.frameon": False,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    }
)


def _output_path(output: str | Path, filename: str) -> Path:
    path = Path(output)
    if path.suffix.casefold() != ".png":
        path = path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _require(data: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(data.columns))
    if missing:
        raise ValueError(f"missing chart columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("chart data must not be empty")


def _save(fig: Figure, output: str | Path, filename: str) -> Path:
    path = _output_path(output, filename)
    fig.savefig(path, format="png")
    plt.close(fig)
    return path


def calibration(deciles: pd.DataFrame, output: str | Path) -> Path:
    """Plot E1 calibration from ``decile``, ``observed_rate`` and optional ``policy``.

    ``predicted_rate`` is optional; when present it is drawn as a dashed reference.
    ``output`` may be a PNG path or a directory (creating ``calibration.png``).
    """

    _require(deciles, {"decile", "observed_rate"})
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    groups = deciles.groupby("policy", sort=False) if "policy" in deciles else [("Observed", deciles)]
    for index, (label, frame) in enumerate(groups):
        ordered = frame.sort_values("decile")
        ax.plot(
            ordered["decile"],
            ordered["observed_rate"],
            marker="o",
            linewidth=2,
            color=_COLORS[index % len(_COLORS)],
            label=str(label),
        )
    if "predicted_rate" in deciles:
        reference = deciles.sort_values("decile").drop_duplicates("decile")
        ax.plot(
            reference["decile"],
            reference["predicted_rate"],
            linestyle="--",
            color="#64748b",
            label="Predicted",
        )
    ax.set(title="Urgency calibration by score decile", xlabel="Score decile", ylabel="Observed urgency rate")
    ax.set_ylim(bottom=0)
    ax.legend()
    return _save(fig, output, "calibration.png")


def operating_curves(
    policy_curves: Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float]]],
    output: str | Path,
) -> Path:
    """Plot E1 operating curves from ``policy -> (budget, recall, precision)`` arrays."""

    if not policy_curves:
        raise ValueError("policy_curves must not be empty")
    fig, (recall_ax, precision_ax) = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True)
    for index, (policy, values) in enumerate(policy_curves.items()):
        if len(values) != 3:
            raise ValueError("each policy curve must contain budget, recall, and precision")
        budget, recall_values, precision_values = (np.asarray(value, dtype=float) for value in values)
        if not (len(budget) == len(recall_values) == len(precision_values)):
            raise ValueError("budget, recall, and precision arrays must have equal lengths")
        color = _COLORS[index % len(_COLORS)]
        recall_ax.plot(budget, recall_values, marker="o", label=policy, color=color)
        precision_ax.plot(budget, precision_values, marker="o", label=policy, color=color)
    recall_ax.set(title="Recall at admission budget", xlabel="Budget (%)", ylabel="Recall")
    precision_ax.set(title="Precision at admission budget", xlabel="Budget (%)", ylabel="Precision")
    recall_ax.set_ylim(0, 1.03)
    precision_ax.set_ylim(0, 1.03)
    recall_ax.legend()
    return _save(fig, output, "operating_curves.png")


def e2_family_bars(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot E2 grouped bars from long-form ``arm``, ``family``, ``accuracy`` rows."""

    _require(data, {"arm", "family", "accuracy"})
    pivot = data.pivot_table(index="family", columns="arm", values="accuracy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(7.2, 1.2 * len(pivot)), 4.8))
    pivot.plot(kind="bar", ax=ax, color=_COLORS[: len(pivot.columns)], width=0.82)
    ax.set(title="State fidelity by probe family", xlabel="Probe family", ylabel="Accuracy")
    ax.set_ylim(0, 1.03)
    ax.tick_params(axis="x", rotation=25)
    return _save(fig, output, "e2_family_bars.png")


def e3_curve(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot E3 accuracy curves from ``store``, ``budget``, ``acc`` and optional CI rows.

    If supplied, confidence bounds must be named ``ci_low`` and ``ci_high``.
    """

    _require(data, {"store", "budget", "acc"})
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for index, (store, frame) in enumerate(data.groupby("store", sort=False)):
        ordered = frame.sort_values("budget")
        color = _COLORS[index % len(_COLORS)]
        ax.plot(ordered["budget"], ordered["acc"], marker="o", linewidth=2, label=store, color=color)
        if {"ci_low", "ci_high"}.issubset(ordered.columns):
            ax.fill_between(
                ordered["budget"].to_numpy(float),
                ordered["ci_low"].to_numpy(float),
                ordered["ci_high"].to_numpy(float),
                color=color,
                alpha=0.15,
            )
    ax.set(title="Store accuracy vs read budget", xlabel="Read budget (tokens)", ylabel="Macro accuracy")
    ax.set_ylim(0, 1.03)
    ax.legend(title="Store")
    return _save(fig, output, "curve.png")


def erosion(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot longitudinal E3 accuracy from ``store``, ``checkpoint`` and ``acc`` rows."""

    _require(data, {"store", "checkpoint", "acc"})
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for index, (store, frame) in enumerate(data.groupby("store", sort=False)):
        ordered = frame.sort_values("checkpoint")
        ax.plot(
            ordered["checkpoint"],
            ordered["acc"],
            marker="o",
            linewidth=2,
            label=store,
            color=_COLORS[index % len(_COLORS)],
        )
    ax.set(title="Store fidelity over replay time", xlabel="Checkpoint", ylabel="Accuracy")
    ax.set_ylim(0, 1.03)
    ax.legend(title="Store")
    return _save(fig, output, "erosion.png")


def health_table(data: pd.DataFrame, output: str | Path) -> Path:
    """Render E3 health from long-form ``store``, ``metric``, ``value`` rows.

    Values should already use a comparable orientation (larger is better or
    smaller is better consistently) when color interpretation matters.
    """

    _require(data, {"store", "metric", "value"})
    pivot = data.pivot_table(index="store", columns="metric", values="value", aggfunc="mean")
    values = pivot.to_numpy(dtype=float)
    column_min = np.nanmin(values, axis=0)
    column_range = np.nanmax(values, axis=0) - column_min
    normalised = (values - column_min) / np.where(column_range == 0, 1, column_range)
    fig, ax = plt.subplots(figsize=(max(7.2, 1.25 * len(pivot.columns)), max(2.8, 0.6 * len(pivot))))
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.grid(False)
    ax.set_title("Store health metrics")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.3g}", ha="center", va="center")
    return _save(fig, output, "health_table.png")


def e4_envelopes(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot E4 envelope ``Q`` with CI and ``tokens`` from one row per envelope.

    Required columns are ``envelope``, ``Q``, and ``tokens``. Optional
    ``ci_low``/``ci_high`` bounds add asymmetric error bars.
    """

    _require(data, {"envelope", "Q", "tokens"})
    ordered = data.sort_values("envelope")
    positions = np.arange(len(ordered))
    error: Any = None
    if {"ci_low", "ci_high"}.issubset(ordered.columns):
        q = ordered["Q"].to_numpy(float)
        error = np.vstack((q - ordered["ci_low"].to_numpy(float), ordered["ci_high"].to_numpy(float) - q))
    fig, quality_ax = plt.subplots(figsize=(8.2, 4.8))
    quality_ax.bar(positions, ordered["Q"], yerr=error, color="#2563eb", alpha=0.85, capsize=4)
    quality_ax.set_xticks(positions, labels=ordered["envelope"])
    quality_ax.set(title="Context envelope quality and size", xlabel="Envelope", ylabel="Action quality Q")
    quality_ax.set_ylim(0, 1.03)
    token_ax = quality_ax.twinx()
    token_ax.plot(positions, ordered["tokens"], color="#ea580c", marker="o", linewidth=2)
    token_ax.set_ylabel("Median tokens", color="#ea580c")
    token_ax.grid(False)
    return _save(fig, output, "e4_envelopes.png")


def e5_pareto(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot E5 cost trade-offs from ``cadence``, ``cost``, ``acc``, ``freshness`` rows."""

    _require(data, {"cadence", "cost", "acc", "freshness"})
    fig, (accuracy_ax, freshness_ax) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for color_index, (_, row) in enumerate(data.reset_index(drop=True).iterrows()):
        color = _COLORS[color_index % len(_COLORS)]
        accuracy_ax.scatter(row["cost"], row["acc"], s=65, color=color)
        freshness_ax.scatter(row["cost"], row["freshness"], s=65, color=color)
        accuracy_ax.annotate(str(row["cadence"]), (row["cost"], row["acc"]), xytext=(5, 4), textcoords="offset points")
        freshness_ax.annotate(str(row["cadence"]), (row["cost"], row["freshness"]), xytext=(5, 4), textcoords="offset points")
    accuracy_ax.set(title="Cost vs accuracy", xlabel="Cost per 1,000 events ($)", ylabel="Macro accuracy")
    accuracy_ax.set_ylim(0, 1.03)
    freshness_ax.set(title="Cost vs freshness", xlabel="Cost per 1,000 events ($)", ylabel="Freshness delay")
    return _save(fig, output, "pareto.png")


def e6_burst_slo(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot E6 burst results from ``lane_design``, ``load``, ``slo_attainment`` rows."""

    _require(data, {"lane_design", "load", "slo_attainment"})
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for index, (design, frame) in enumerate(data.groupby("lane_design", sort=False)):
        ordered = frame.sort_values("load")
        ax.plot(
            ordered["load"],
            ordered["slo_attainment"],
            marker="o",
            linewidth=2,
            label=design,
            color=_COLORS[index % len(_COLORS)],
        )
    ax.axhline(0.99, color="#64748b", linestyle="--", linewidth=1, label="99% target")
    ax.set(title="Urgent-event SLO attainment under burst load", xlabel="Load (× knee)", ylabel="SLO attainment")
    ax.set_ylim(0, 1.03)
    ax.legend()
    return _save(fig, output, "burst_slo.png")


def self_amplification(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot E6 ``time``, ``admission_rate`` and ``slo_attainment`` on two axes."""

    _require(data, {"time", "admission_rate", "slo_attainment"})
    ordered = data.sort_values("time")
    fig, admission_ax = plt.subplots(figsize=(8.4, 4.6))
    admission_ax.plot(ordered["time"], ordered["admission_rate"], color="#2563eb", linewidth=2)
    admission_ax.set(title="Fast-lane self-amplification", xlabel="Time", ylabel="Admission rate",)
    compliance_ax = admission_ax.twinx()
    compliance_ax.plot(ordered["time"], ordered["slo_attainment"], color="#dc2626", linewidth=2)
    compliance_ax.set_ylabel("Urgent-event SLO attainment", color="#dc2626")
    compliance_ax.set_ylim(0, 1.03)
    compliance_ax.grid(False)
    return _save(fig, output, "self_amplification.png")


def demand_curve(data: pd.DataFrame, output: str | Path) -> Path:
    """Plot E6 load-to-resource demand from ``load``, ``partitions``, ``workers`` rows."""

    _require(data, {"load", "partitions", "workers"})
    ordered = data.sort_values("load")
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.step(ordered["load"], ordered["partitions"], where="mid", marker="o", label="Partitions", color="#2563eb")
    ax.step(ordered["load"], ordered["workers"], where="mid", marker="o", label="Workers", color="#059669")
    ax.set(title="Minimum resources meeting the SLO", xlabel="Load", ylabel="Resource count")
    ax.set_ylim(bottom=0)
    ax.legend()
    return _save(fig, output, "demand_curve.png")


def checks_table(
    checks: Mapping[str, bool | float | Mapping[str, object]], output: str | Path
) -> Path:
    """Render a pass/fail table from a check mapping.

    Values may be booleans, numeric values (display-only), or mappings containing
    ``passed`` and optionally ``value``. Numeric-only rows have neutral status.
    """

    if not checks:
        raise ValueError("checks must not be empty")
    rows: list[list[str]] = []
    colors: list[list[str]] = []
    for name, raw in checks.items():
        value: object = raw
        passed: bool | None = raw if isinstance(raw, bool) else None
        if isinstance(raw, Mapping):
            value = raw.get("value", raw.get("passed", ""))
            candidate = raw.get("passed")
            passed = candidate if isinstance(candidate, bool) else None
        status = "PASS" if passed is True else "FAIL" if passed is False else "INFO"
        rows.append([str(name), str(value), status])
        status_color = "#dcfce7" if passed is True else "#fee2e2" if passed is False else "#e2e8f0"
        colors.append(["white", "white", status_color])
    fig, ax = plt.subplots(figsize=(8.4, max(2.2, 0.45 * len(rows) + 1.2)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Check", "Value", "Status"],
        cellColours=colors,
        colColours=["#e2e8f0"] * 3,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.58, 0.24, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title("Validity checks", pad=12)
    return _save(fig, output, "checks_table.png")
