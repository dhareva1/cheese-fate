"""Publication figures for the cheese manuscript.

Reads ``reports/experiments/<split>/`` (after ``run_inference.py``) and the processed
observations, and writes vector PDF and 600-dpi PNG files to ``reports/figures/``:

- Fig. 1: one production batch: process conditions, same-batch LAB and both pathogens;
- Fig. 2: skill against persistence with 95% source-bootstrap intervals (RQ1, RQ2, RQ6);
- Fig. 3: where measured future inputs help or hurt, per held-out source (RQ2);
- Fig. 4: the pre-registered deacidification test (RQ4);
- Fig. 5: observed time-to-safe and the 60-day rule (RQ5);
- Fig. S1: example held-out curves with 90% conformal bands (supplementary).

Example (from the repository root):
    python scripts/make_publication_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORGANISMS = {"escherichia_coli": "Escherichia coli",
             "staphylococcus_aureus": "Staphylococcus aureus"}
BEST = {"escherichia_coli": "gru", "staphylococcus_aureus": "gbt"}
NAMES = {
    "persistence": "Persistence", "mechanistic": "Mechanistic", "gbt": "GBT",
    "gbt_nolab": "GBT, no LAB", "gru": "GRU", "lstm": "LSTM",
    "residual_gbt": "Residual hybrid", "residual_gbt_nolab": "Residual hybrid, no LAB",
    "node_hybrid": "Neural-ODE hybrid",
}
ORDER = ["mechanistic", "gbt", "gbt_nolab", "gru", "lstm", "residual_gbt",
         "residual_gbt_nolab", "node_hybrid"]
# First three slots of the validated categorical palette (all-pairs safe), plus a marker
# per mode so that identity never depends on colour alone.
MODES = {"measured": ("#2a78d6", "o", "Measured inputs"),
         "naive": ("#eb6834", "s", "Held at origin (naive)"),
         "coupled": ("#1baf7a", "^", "Coupled forecast")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 174 / 25.4  # double-column width, inches


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
        "lines.linewidth": 1.4,
    })


def letter(ax, text: str) -> None:
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, fontweight="bold", fontsize=9,
            ha="right", va="bottom", color=INK)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}", flush=True)


def fig1_batch(processed: Path, out: Path, batch: str = "BitBo1") -> None:
    frames = {slug: pd.read_csv(processed / f"observations_{slug}.csv") for slug in ORGANISMS}
    ec = frames["escherichia_coli"].query("batch_id == @batch")
    sa = frames["staphylococcus_aureus"].query("batch_id == @batch")
    days = lambda t: np.maximum(t, 0.0) / 24.0 + 0.01  # noqa: E731
    fig, axes = plt.subplots(4, 1, figsize=(90 / 25.4, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.6], "hspace": 0.45})
    series = [("temperature_c", "Temperature (°C)", "step"), ("ph", "pH", "line"),
              ("lab_log_count", "LAB (log$_{10}$ CFU/g)", "line")]
    for ax, (column, label, kind), tag in zip(axes, series, "abc"):
        frame = ec.dropna(subset=[column])
        if kind == "step":
            ax.step(days(frame["time_h"]), frame[column], where="post", color=INK, lw=1.2)
        else:
            # Measured values only; aligned (interpolated) values are not data.
            measured = frame[~frame[f"{column}_interpolated"].astype(bool)]
            ax.plot(days(measured["time_h"]), measured[column], "-o", ms=3.5, lw=1.0, color=INK)
        ax.set_ylabel(label)
        ax.grid(axis="y", color=GRID, lw=0.5)
        letter(ax, tag)
    ax = axes[3]
    for frame, colour, marker, name in ((ec, "#2a78d6", "o", "E. coli"),
                                        (sa, "#eb6834", "s", "S. aureus")):
        detected = frame[~frame["censored"]]
        censored = frame[frame["censored"]]
        ax.plot(days(frame["time_h"]), frame["log_count"], color=colour, lw=1.0, alpha=0.6)
        ax.plot(days(detected["time_h"]), detected["log_count"], marker, ms=4.5, color=colour,
                mec="white", mew=0.6, label=f"$\\it{{{name.replace(' ', '~')}}}$")
        ax.plot(days(censored["time_h"]), censored["log_count"], "v", ms=5, mfc="white",
                mec=colour, mew=1.0)
    ax.plot([], [], "v", ms=5, mfc="white", mec=MUTED, label="Not detected (≤ limit)")
    ax.axvline(60, color=INK, lw=1.0, ls="--")
    ax.text(60, ax.get_ylim()[1], " 60 d", color=INK, va="top", fontsize=8)
    ax.set_ylabel("Pathogen (log$_{10}$ CFU/g)")
    ax.set_xscale("log")
    ax.set_xlabel("Time since start of manufacture (days, log scale)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.02), ncol=3,
              fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
    letter(ax, "d")
    fig.align_ylabels(axes)
    save(fig, out, "fig1_batch")


def fig2_skill(experiments: Path, out: Path) -> None:
    splits = [("source_id", "Held-out sources"), ("food", "Held-out cheese types")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    offsets = {"measured": 0.22, "naive": 0.0, "coupled": -0.22}
    for row, (slug, organism) in enumerate(ORGANISMS.items()):
        for col, (split, title) in enumerate(splits):
            ax = axes[row, col]
            table = pd.read_csv(experiments / split / f"{slug}_validation.csv").query("origin_h == 0")
            for mode, (colour, marker, label) in MODES.items():
                part = table[table["mode"] == mode].set_index("model").reindex(ORDER)
                yy = y + offsets[mode]
                ax.errorbar(part["skill"], yy,
                            xerr=[part["skill"] - part["skill_lo"], part["skill_hi"] - part["skill"]],
                            fmt=marker, ms=4.5, color=colour, mec="white", mew=0.5,
                            elinewidth=1.0, capsize=0, label=label)
            ax.axvline(0, color=INK, lw=0.8)
            ax.axvspan(-2, 0, color="#f3f2ef", zorder=0)
            ax.set_xlim(-1.6, 0.45)
            ax.set_yticks(y, [NAMES[m] for m in ORDER])
            ax.grid(axis="x", color=GRID, lw=0.5)
            ax.set_title(f"$\\it{{{organism.replace(' ', '~')}}}$ · {title}", loc="left")
            if row == 1:
                ax.set_xlabel("Skill vs persistence, 1 − RMSE/RMSE$_{pers}$ (95% CI)")
            letter(ax, "abcd"[2 * row + col])
            ax.set_ylim(y[-1] - 0.75, y[0] + 0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "fig2_skill")


def fig3_sources(experiments: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    for ax, (slug, organism), tag in zip(axes, ORGANISMS.items(), "ab"):
        model = BEST[slug]
        gap = pd.read_csv(experiments / "source_id" / f"probe_{slug}_gap_by_source_{model}.csv")
        gap = gap.sort_values("difference", ascending=False).reset_index(drop=True)
        cumulative = gap["difference"].cumsum()
        x = np.arange(len(gap))
        colours = np.where(gap["difference"] > 0, "#eb6834", "#2a78d6")
        ax.bar(x, gap["difference"], color=colours, width=0.8)
        ax.plot(x, cumulative, color=INK, lw=1.0)
        ax.axhline(0, color=INK, lw=0.6)
        for i in range(3):
            ax.annotate(gap.loc[i, "source_id"], (x[i], gap.loc[i, "difference"]),
                        xytext=(3, 0), textcoords="offset points", fontsize=7.5, color=MUTED,
                        va="center")
        ax.set_xticks([])
        ax.set_xlabel(f"Held-out sources, sorted ({len(gap)})")
        ax.set_ylabel("SSE(measured) − SSE(naive)")
        ax.set_title(f"$\\it{{{organism.replace(' ', '~')}}}$, {NAMES[model]}", loc="left")
        ax.text(0.5, 0.62, f"total {gap['difference'].sum():+.0f}\n"
                f"top 3 sources {gap['difference'].head(3).sum():+.0f}\n"
                f"other {len(gap) - 3} sources {gap['difference'].iloc[3:].sum():+.0f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.5)
        letter(ax, tag)
    handles = [Patch(color="#eb6834", label="Measured inputs worse than naive"),
               Patch(color="#2a78d6", label="Measured inputs better than naive"),
               Line2D([], [], color=INK, lw=1.0, label="Cumulative sum")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig3_sources")


def fig4_rebound(experiments: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, (slug, organism), tag in zip(axes, ORGANISMS.items(), "ab"):
        frame = pd.read_csv(experiments / "source_id" / f"{slug}_ph_attribution_gbt.csv")
        frame = frame.dropna(subset=["rebound", "ph_attribution"])
        rebound = frame[frame["rebound"] > 0.3]
        other = frame[frame["rebound"] <= 0.3]
        ax.plot(other["rebound"], other["ph_attribution"], "o", ms=3, color="#b9b8b2",
                mec="white", mew=0.3, label="Rebound ≤ 0.3")
        ax.plot(rebound["rebound"], rebound["ph_attribution"], "o", ms=3.5, color="#2a78d6",
                mec="white", mew=0.3, label="Rebound > 0.3 (tested)")
        rho, _ = stats.spearmanr(rebound["rebound"], rebound["ph_attribution"])
        fit = np.polyfit(rebound["rebound"], rebound["ph_attribution"], 1)
        xs = np.linspace(0.3, rebound["rebound"].max(), 50)
        ax.plot(xs, np.polyval(fit, xs), color=INK, lw=1.0)
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(0.3, color=MUTED, lw=0.6, ls="--")
        ax.text(0.98, 0.95, f"n = {len(rebound)}, Spearman ρ = {rho:.2f}\n"
                "hypothesis: ρ > 0 and attribution > 0", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=MUTED)
        ax.set_xlabel("pH rise after the pH minimum")
        ax.set_title(f"$\\it{{{organism.replace(' ', '~')}}}$ (GBT)", loc="left")
        ax.grid(color=GRID, lw=0.5)
        letter(ax, tag)
    axes[0].set_ylabel("Mean pH attribution after minimum\n(SHAP, log$_{10}$ CFU/g)")
    handles = [Line2D([], [], ls="", marker="o", ms=4, color="#b9b8b2", label="Rebound ≤ 0.3"),
               Line2D([], [], ls="", marker="o", ms=4, color="#2a78d6",
                      label="Rebound > 0.3 (tested)"),
               Line2D([], [], color=INK, lw=1.0, label="Linear trend, tested curves")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig4_rebound")


def fig5_time_to_safe(experiments: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    targets = {"detection_limit": ("#2a78d6", "At or below detection limit"),
               "reduction": ("#eb6834", "≥ 2-log reduction from start")}
    for ax, (slug, organism), tag in zip(axes, ORGANISMS.items(), "ab"):
        table = pd.read_csv(experiments / "source_id" / f"{slug}_time_to_safe.csv")
        table = table.query("mode == 'measured' and band == 'two-sided'")
        safe60 = {}
        grid = np.logspace(-1, np.log10(400), 400)
        for target, (colour, label) in targets.items():
            part = table[table["target"] == target]
            safe_days = part["observed_safe_h"].to_numpy(dtype=float) / 24.0
            share = [(safe_days <= t).mean() for t in grid]
            ax.plot(grid, share, color=colour, label=label)
            past = part[part["observed_past_60d"].astype(bool)]
            safe60[target] = (int(past["observed_safe_from_60d"].astype(bool).sum()), len(past))
        ripening = pd.read_csv(experiments / "source_id" / f"{slug}_conformal_upper_ripening.csv")
        width = ripening.query("mode == 'measured'")["q_mean"].iloc[0]
        ax.axvline(60, ymax=0.74, color=INK, lw=1.2, ls="--")
        ax.text(66, 0.02, "60 days", ha="left", va="bottom", fontsize=8, color=INK,
                fontweight="bold")
        (dl, n60), (red, _) = safe60["detection_limit"], safe60["reduction"]
        ax.text(0.03, 0.95, f"Curves followed past day 60 (n = {n60}), safe from day 60 on:\n"
                f"  at or below detection limit {dl} ({dl / n60:.0%}); "
                f"≥ 2-log reduction {red} ({red / n60:.0%})\n"
                f"90% upper conformal band +{width:.1f} log: 0% certified",
                transform=ax.transAxes, va="top", fontsize=7.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Ripening time (days, log scale)")
        ax.set_title(f"$\\it{{{organism.replace(' ', '~')}}}$ ({NAMES[BEST[slug]]})", loc="left")
        ax.grid(color=GRID, lw=0.5)
        letter(ax, tag)
    axes[0].set_ylabel("Share of curves observed\nsafe from that time on")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig5_time_to_safe")


def figS1_example_curves(experiments: Path, out: Path, per_organism: int = 3) -> None:
    from cheese_fate.analysis import cross_conformal

    fig, axes = plt.subplots(2, per_organism, figsize=(WIDTH, 4.4), sharey=True)
    for row, ((slug, organism), tag) in enumerate(zip(ORGANISMS.items(), "ab")):
        folder = experiments / "source_id"
        best = BEST[slug]
        predictions = pd.read_csv(folder / f"{slug}_predictions.csv.gz")
        rows = predictions.query("origin_h == 0 and model == @best and mode == 'measured'")
        dense = pd.read_csv(folder / f"{slug}_dense.csv.gz").query(
            "model == @best and mode == 'measured'")
        q_of = cross_conformal(rows, 0.1).set_index("record_id")["q"]
        # Same rule as analyze_experiments.py: long, well-sampled curves, spread by length.
        counts = rows.groupby("record_id").agg(n=("time_h", "size"), end=("time_h", "max"))
        chosen = counts[(counts["n"] >= 6) & (counts["end"] > 720)].sort_values("n").index
        chosen = list(chosen[np.linspace(0, len(chosen) - 1, per_organism).astype(int)])
        for col, (ax, record_id) in enumerate(zip(axes[row], chosen)):
            curve = dense[dense["record_id"] == record_id]
            observed = rows[rows["record_id"] == record_id]
            days, q = curve["time_h"] / 24, q_of[record_id]
            ax.fill_between(days, curve["predicted"] - q, curve["predicted"] + q,
                            color=MODES["measured"][0], alpha=0.18, lw=0)
            ax.plot(days, curve["predicted"], color=MODES["measured"][0])
            detected, censored = observed[~observed["censored"]], observed[observed["censored"]]
            ax.scatter(detected["time_h"] / 24, detected["log_count"], s=10, color=INK, zorder=3)
            ax.scatter(censored["time_h"] / 24, censored["log_count"], s=14, marker="v",
                       facecolor="white", edgecolor=INK, lw=0.7, zorder=3)
            anchor = observed[["anchor_time", "anchor_log"]].iloc[0]
            ax.scatter([anchor["anchor_time"] / 24], [anchor["anchor_log"]], s=16, marker="s",
                       color=MUTED, zorder=3)
            ax.set_xscale("symlog", linthresh=1)
            food = observed["food"].iloc[0].removesuffix(" cheese").removesuffix(" Cheese")
            ax.set_title(f"{record_id}: {food if len(food) <= 22 else food[:21] + '…'}",
                         loc="left", fontsize=7.5)
            ax.grid(color=GRID, lw=0.5)
            if row == 1:
                ax.set_xlabel("Time (days, symlog scale)")
            if col == 0:
                ax.set_ylabel("log$_{10}$ CFU/g")
                ax.text(0, 1.2, f"$\\it{{{organism.replace(' ', '~')}}}$ ({NAMES[best]})",
                        transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK)
                letter(ax, tag)
    handles = [Line2D([], [], color=MODES["measured"][0], label="Prediction"),
               Patch(color=MODES["measured"][0], alpha=0.18, label="90% conformal band"),
               Line2D([], [], ls="", marker="s", color=MUTED, ms=4, label="Anchor (origin)"),
               Line2D([], [], ls="", marker="o", color=INK, ms=3.5, label="Observed count"),
               Line2D([], [], ls="", marker="v", mfc="white", mec=INK, ms=4,
                      label="Not detected (detection limit)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.tight_layout(h_pad=2.2)
    save(fig, out, "figS1_example_curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", type=Path, default=ROOT / "reports/experiments")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/cheese")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    fig1_batch(args.processed, args.out)
    fig2_skill(args.experiments, args.out)
    fig3_sources(args.experiments, args.out)
    fig4_rebound(args.experiments, args.out)
    fig5_time_to_safe(args.experiments, args.out)
    figS1_example_curves(args.experiments, args.out)


if __name__ == "__main__":
    main()
