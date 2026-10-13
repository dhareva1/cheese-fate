"""Static figures for the experiment reports (matplotlib).

Colours follow a fixed categorical order (blue, orange, aqua) assigned to entities, never
to rank; text stays in neutral ink; grids and axes are recessive; lines are labelled
directly. Every figure uses a single y-scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cheese_fate.analysis import HORIZON_LABELS, SIXTY_DAYS_H  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"
MODE_COLOURS = {"measured": SERIES[0], "naive": SERIES[1], "coupled": SERIES[2]}
MODE_LABELS = {"measured": "Measured inputs", "naive": "Forecast: hold last",
               "coupled": "Forecast: coupled"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "legend.frameon": False, "legend.fontsize": 8,
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def model_accuracy(summaries: dict[str, pd.DataFrame], path: Path, order: list[str]) -> Path:
    """Pooled RMSE per model (measured inputs, origin 0) with the persistence reference."""
    _style()
    fig, axes = plt.subplots(1, len(summaries), figsize=(4.2 * len(summaries), 3.6),
                             sharey=True, squeeze=False)
    for ax, (organism, table) in zip(axes[0], summaries.items()):
        rows = table[(table["mode"] == "measured") & (table["origin_h"] == 0)].set_index("model")
        models = [m for m in order if m in rows.index and m != "persistence"]
        y = np.arange(len(models))
        rmse = rows.loc[models, "rmse"].to_numpy()
        reference = rows.loc["persistence", "rmse"]
        ax.hlines(y, 0, rmse, color=GRID, linewidth=2)
        ax.scatter(rmse, y, s=36, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for yi, value in zip(y, rmse):
            ax.text(value + 0.05, yi, f"{value:.2f}", va="center", fontsize=8, color=INK_2)
        ax.axvline(reference, color=INK_2, linestyle="--", linewidth=1)
        ax.set_yticks(y, models)
        ax.set_ylim(len(models) - 0.5, -1.1)
        ax.text(reference, -0.75, f" persistence {reference:.2f}", fontsize=8, color=INK_2,
                va="center")
        ax.set_xlim(0, max(rmse.max(), reference) * 1.25)
        ax.set_xlabel("RMSE, log10 CFU/g (lower is better)")
        ax.set_title(organism, style="italic")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Accuracy on held-out sources, measured inputs", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def forecast_gap(gaps: dict[str, pd.DataFrame], model: str, path: Path) -> Path:
    """RMSE by forecast horizon in each input mode (origin 0), for one model."""
    _style()
    fig, axes = plt.subplots(1, len(gaps), figsize=(4.4 * len(gaps), 3.4), sharey=True,
                             squeeze=False)
    for ax, (organism, table) in zip(axes[0], gaps.items()):
        rows = table[(table["model"] == model) & (table["origin_h"] == 0)]
        rows = rows.set_index("horizon").reindex(list(HORIZON_LABELS)).dropna(subset=["measured"])
        x = np.arange(len(rows))
        for mode in ("measured", "naive", "coupled"):
            if mode not in rows:
                continue
            ax.plot(x, rows[mode], color=MODE_COLOURS[mode], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE)
            ax.annotate(MODE_LABELS[mode], (x[-1], rows[mode].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        persistence = table[(table["model"] == "persistence") & (table["origin_h"] == 0)]
        persistence = persistence.set_index("horizon").reindex(rows.index)["measured"]
        ax.plot(x, persistence, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate("Persistence", (x[0], persistence.iloc[0]), xytext=(0, 6),
                    textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xticks(x, rows.index)
        ax.set_xlim(-0.3, len(x) + 0.9)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("RMSE, log10 CFU/g")
        ax.set_title(organism, style="italic")
    fig.suptitle(f"Forecast-mode gap by horizon ({model}, origin day 0)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def rebound_scatter(per_record: dict[str, pd.DataFrame], model: str, threshold: float,
                    path: Path) -> Path:
    """Post-minimum pH attribution against the size of the pH rebound, per curve."""
    _style()
    fig, axes = plt.subplots(1, len(per_record), figsize=(4.2 * len(per_record), 3.4),
                             squeeze=False)
    for ax, (organism, table) in zip(axes[0], per_record.items()):
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.axvline(threshold, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text(threshold, 1.0, f" rebound {threshold}", transform=ax.get_xaxis_transform(),
                fontsize=8, color=MUTED, va="top")
        ax.scatter(table["rebound"], table["ph_attribution"], s=22, color=SERIES[0], alpha=0.8,
                   edgecolor=SURFACE, linewidth=0.8)
        ax.set_xlabel("pH rise after the pH minimum")
        ax.set_ylabel("pH attribution after minimum (log10)")
        ax.set_title(organism, style="italic")
    fig.suptitle(f"Deacidification: SHAP attribution of pH ({model})", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def example_curves(dense: pd.DataFrame, targets: pd.DataFrame, bands: pd.DataFrame,
                   record_ids: list[str], model: str, path: Path) -> Path:
    """Observed counts, model trajectory and conformal band for a few held-out curves."""
    _style()
    n = len(record_ids)
    cols = min(n, 3)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.6 * cols, 2.8 * rows_n), squeeze=False,
                             sharey=True)
    q_of = bands.set_index("record_id")["q"]
    for ax, record_id in zip(axes.flat, record_ids):
        curve = dense[(dense["record_id"] == record_id) & (dense["model"] == model)]
        observed = targets[targets["record_id"] == record_id]
        days = curve["time_h"] / 24
        q = q_of.get(record_id, np.nan)
        ax.fill_between(days, curve["predicted"] - q, curve["predicted"] + q, color=SERIES[0],
                        alpha=0.15, linewidth=0)
        ax.plot(days, curve["predicted"], color=SERIES[0], linewidth=2)
        detected = observed[~observed["censored"]]
        censored = observed[observed["censored"]]
        ax.scatter(detected["time_h"] / 24, detected["log_count"], s=20, color=INK, zorder=3)
        ax.scatter(censored["time_h"] / 24, censored["log_count"], s=24, marker="v",
                   facecolor="none", edgecolor=INK, zorder=3)
        anchor = observed[["anchor_time", "anchor_log"]].iloc[0]
        ax.scatter([anchor["anchor_time"] / 24], [anchor["anchor_log"]], s=28, marker="s",
                   color=INK_2, zorder=3)
        ax.set_xscale("symlog", linthresh=1)
        food = observed["food"].iloc[0].removesuffix(" cheese").removesuffix(" Cheese")
        ax.set_title(f"{record_id} · {food if len(food) <= 24 else food[:23] + '…'}", fontsize=9)
        ax.set_xlabel("Days")
    for ax in axes[:, 0]:
        ax.set_ylabel("log10 CFU/g")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Held-out curves: {model} with 90% conformal band "
                 "(■ anchor, ● count, ▽ not detected)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def time_to_safe(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    """Observed share of curves safe from each ripening day on, with the conformal verdict.

    A curve counts as safe from day d when every observation from d on is at or below the
    target; curves never observed safe stay out of the share at every day.
    """
    _style()
    fig, axes = plt.subplots(1, len(tables), figsize=(4.6 * len(tables), 3.5), sharey=True,
                             squeeze=False)
    labels = {"detection_limit": "At or below detection limit", "reduction": "2-log reduction"}
    for ax, (organism, table) in zip(axes[0], tables.items()):
        for colour, (target, label) in zip(SERIES, labels.items()):
            rows = table[table["target"] == target]
            if rows.empty:
                continue
            days = rows["observed_safe_h"].to_numpy() / 24
            grid = np.logspace(-1, np.log10(max(rows["end_h"].max() / 24, 1)), 200)
            share = [(days <= g).mean() for g in grid]
            ax.plot(grid, share, color=colour, linewidth=2)
            ax.annotate(label, (grid[-1], share[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
        ax.axvline(SIXTY_DAYS_H / 24, color=INK_2, linestyle="--", linewidth=1)
        ax.text(SIXTY_DAYS_H / 24, 1.0, " 60 days", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK_2, va="top")
        certified = np.isfinite(table["predicted_safe_h"]).mean()
        ax.text(0.02, 0.97, f"90% upper band: +{table['q'].mean():.1f} log\n"
                f"certifies {certified:.0%} of curves", transform=ax.transAxes, fontsize=8,
                color=INK_2, va="top")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Ripening time, days")
        ax.set_ylabel("Share of curves observed safe from then on")
        ax.set_title(organism, style="italic")
    fig.suptitle("Time-to-safe: observed vs. conformally certified", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)
