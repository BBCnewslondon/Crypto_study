"""
Scientific Visualisation Module
================================
Generates publication-quality figures from pipeline results.
Uses matplotlib with a tuned academic style.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Global style configuration — publication-ready
# ---------------------------------------------------------------------------
STYLE_PARAMS = {
    # --- Font ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # --- Figure ---
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,

    # --- Axes ---
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.spines.top": False,
    "axes.spines.right": False,

    # --- Grid ---
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,

    # --- Ticks ---
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # --- Lines ---
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

# Colour palette — muted, accessible, scientific
PALETTE = {
    "primary":      "#2c6fbb",   # Steel blue
    "secondary":    "#cc4c02",   # Burnt orange
    "tertiary":     "#238b45",   # Forest green
    "quaternary":   "#6a3d9a",   # Purple
    "accent":       "#e31a1c",   # Red (for highlights)
    "neutral":      "#636363",   # Grey
    "light_fill":   "#d0e1f9",   # Light blue fill
    "neg_fill":     "#fdd0a2",   # Light orange fill
    "bg":           "#fafafa",   # Near-white background
    "grid":         "#e0e0e0",
}

SYSTEM_COLOURS = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["tertiary"],
    PALETTE["quaternary"],
]


def _apply_style() -> None:
    """Apply the publication style globally."""
    plt.rcParams.update(STYLE_PARAMS)


# ---------------------------------------------------------------------------
# 1.  CCF Correlogram (one per system)
# ---------------------------------------------------------------------------
def plot_ccf_correlogram(
    system_label: str,
    ccf_data: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot the cross-correlation function for each epoch of a system.

    Parameters
    ----------
    system_label : e.g. "BTC -> STX"
    ccf_data : list of dicts with 'lags', 'correlations', 'optimal_lag', 'peak_correlation'
    output_dir : directory to save the figure

    Returns
    -------
    Path to the saved figure.
    """
    _apply_style()
    n_epochs = len(ccf_data)
    cols = min(n_epochs, 2)
    rows = (n_epochs + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.5 * cols, 4 * rows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, epoch in enumerate(ccf_data):
        ax = axes_flat[i]
        lags = np.array(epoch["lags"])
        corrs = np.array(epoch["correlations"])
        opt_lag = epoch["optimal_lag"]
        peak = epoch["peak_correlation"]

        # Stem-like bar plot
        colours = np.where(lags > 0, PALETTE["primary"], PALETTE["neutral"])
        # Highlight the optimal lag bar
        opt_idx = np.where(lags == opt_lag)[0]
        if len(opt_idx) > 0:
            colours[opt_idx[0]] = PALETTE["accent"]

        ax.bar(lags, corrs, width=0.85, color=colours, alpha=0.75, edgecolor="none")

        # Zero line
        ax.axhline(0, color="#333333", linewidth=0.6, zorder=1)
        ax.axvline(0, color="#999999", linewidth=0.5, linestyle="--", alpha=0.5)

        # 95% confidence band (approx 2/sqrt(N))
        n_obs = epoch.get("n_obs", 43200)  # exact from signal_processing if available, else approx for 30-day epoch
        ci = 1.96 / np.sqrt(max(n_obs, 1))
        ax.axhspan(-ci, ci, color=PALETTE["light_fill"], alpha=0.35, zorder=0,
                    label="95% CI")

        # Annotate optimal lag
        ax.annotate(
            f"$\\tau^*$ = {opt_lag} min\n$\\rho$ = {peak:.4f}",
            xy=(opt_lag, peak),
            xytext=(opt_lag + 8, peak + 0.04 if peak > 0 else peak - 0.06),
            fontsize=8.5,
            color=PALETTE["accent"],
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE["accent"],
                lw=1.0,
                connectionstyle="arc3,rad=0.2",
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["accent"],
                      alpha=0.9, lw=0.7),
        )

        ax.set_xlabel("Lag $\\tau$ (minutes)")
        if i % cols == 0:
            ax.set_ylabel("Cross-Correlation $\\hat{\\rho}(\\tau)$")
        ax.set_title(f"Epoch {i + 1}", fontweight="medium")
        ax.set_xlim(-65, 65)

    for i in range(n_epochs, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle(
        f"Cross-Correlation Function:  {system_label}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out = output_dir / f"ccf_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2.  Granger Causality F-statistic plot
# ---------------------------------------------------------------------------
def plot_granger_fstats(
    system_label: str,
    granger_data: List[Dict[str, Any]],
    alpha: float,
    output_dir: Path,
) -> Path:
    """Bar chart of Granger F-statistics at each lag order."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    lags = [g["lag_order"] for g in granger_data]
    f_stats = [g["f_statistic"] for g in granger_data]
    p_vals = [g["p_value"] for g in granger_data]
    significant = [g["is_significant"] for g in granger_data]

    bar_colours = [
        PALETTE["primary"] if sig else PALETTE["neutral"]
        for sig in significant
    ]

    bars = ax.bar(lags, f_stats, color=bar_colours, alpha=0.8, edgecolor="white",
                  linewidth=0.5, width=0.7)

    # Annotate p-values on each bar
    for lag_i, bar, p in zip(lags, bars, p_vals):
        if p < 1e-10:
            p_text = f"p < 10$^{{{int(np.floor(np.log10(p)))}}}$"
        elif p < 0.001:
            p_text = f"p = {p:.1e}"
        else:
            p_text = f"p = {p:.3f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(f_stats) * 0.02,
            p_text,
            ha="center", va="bottom",
            fontsize=6.5, color="#444444", rotation=45,
        )

    # Significance marker
    ax.axhline(0, color="#333333", linewidth=0.5)

    ax.set_xlabel("Lag Order $p$")
    ax.set_ylabel("F-statistic")
    ax.set_title(
        f"Granger Causality Test:  {system_label}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(lags)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["primary"], alpha=0.8, label=f"Significant ($\\alpha$ = {alpha})"),
        Patch(facecolor=PALETTE["neutral"], alpha=0.8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / f"granger_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3.  Lag Stability Comparison (all systems)
# ---------------------------------------------------------------------------
def plot_lag_stability(
    systems: List[str],
    in_sample_lags: List[int],
    oos_lags: List[int],
    stability: List[bool],
    output_dir: Path,
) -> Path:
    """Grouped bar chart comparing in-sample vs OOS lag for each system."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(systems))
    width = 0.32

    bars1 = ax.bar(x - width / 2, in_sample_lags, width,
                   color=PALETTE["primary"], alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label="In-Sample (Q1)")
    bars2 = ax.bar(x + width / 2, oos_lags, width,
                   color=PALETTE["secondary"], alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label="Out-of-Sample (Q2)")

    # Stability markers
    for i, (stable, isl, oosl) in enumerate(zip(stability, in_sample_lags, oos_lags)):
        marker = "STABLE" if stable else "UNSTABLE"
        colour = PALETTE["tertiary"] if stable else PALETTE["accent"]
        y_pos = max(isl, oosl) + 0.8
        ax.text(i, y_pos, marker, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=colour,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=colour, alpha=0.9, lw=0.8))

    # Tolerance band annotation
    ax.annotate(
        "Tolerance: $|\\tau_{train} - \\tau_{OOS}| \\leq 3$ min",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, style="italic",
        color=PALETTE["neutral"],
    )

    ax.set_xlabel("Gravitational System")
    ax.set_ylabel("Optimal Lag $\\tau^*$ (minutes)")
    ax.set_title("Lag Stability:  In-Sample vs. Out-of-Sample",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(max(in_sample_lags), max(oos_lags)) * 1.5 + 2)

    fig.tight_layout()
    out = output_dir / "lag_stability_comparison.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4.  Friction Analysis (all systems)
# ---------------------------------------------------------------------------
def plot_friction_analysis(
    systems: List[str],
    predicted_bps: List[float],
    friction_bps: float,
    output_dir: Path,
) -> Path:
    """Horizontal bar chart: predicted edge vs. mechanical friction."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    y = np.arange(len(systems))
    bar_height = 0.5

    # Predicted move bars
    bars = ax.barh(y, predicted_bps, bar_height,
                   color=PALETTE["primary"], alpha=0.8,
                   edgecolor="white", linewidth=0.5,
                   label="Mean |Return| at $\\tau^*$ (bps)")

    # Friction threshold line
    ax.axvline(friction_bps, color=PALETTE["accent"], linewidth=2,
               linestyle="--", label=f"Round-trip Friction ({friction_bps:.0f} bps)",
               zorder=5)

    # Fill the "unprofitable" zone
    ax.axvspan(0, friction_bps, color=PALETTE["neg_fill"], alpha=0.2, zorder=0)
    ax.axvspan(friction_bps, ax.get_xlim()[1] if ax.get_xlim()[1] > friction_bps
               else friction_bps * 1.5,
               color="#d5efdb", alpha=0.2, zorder=0)

    # Labels
    ax.text(friction_bps / 2, len(systems) - 0.1, "UNPROFITABLE",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color=PALETTE["accent"], alpha=0.6)
    ax.text(friction_bps + (max(predicted_bps + [friction_bps * 1.3]) - friction_bps) / 2,
            len(systems) - 0.1,
            "PROFITABLE", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=PALETTE["tertiary"], alpha=0.6)

    # Value annotations
    for i, (bp, sys) in enumerate(zip(predicted_bps, systems)):
        net = bp - friction_bps
        net_text = f"{bp:.1f} bps  (net: {net:+.1f})"
        colour = PALETTE["tertiary"] if net > 0 else PALETTE["accent"]
        ax.text(bp + 1, i, net_text, va="center", fontsize=8, color=colour,
                fontweight="medium")

    ax.set_yticks(y)
    ax.set_yticklabels(systems, fontsize=10)
    ax.set_xlabel("Basis Points (bps)")
    ax.set_title("Friction Accounting:  Predicted Signal vs. Exchange Costs",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax.set_xlim(0, max(max(predicted_bps), friction_bps) * 1.6)
    ax.invert_yaxis()

    fig.tight_layout()
    out = output_dir / "friction_analysis.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5.  Summary Dashboard (multi-panel)
# ---------------------------------------------------------------------------
def plot_summary_dashboard(
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Multi-panel summary figure combining key metrics."""
    _apply_style()

    # Extract data from results
    systems = []
    in_sample_corrs = []
    oos_corrs = []
    granger_max_f = []
    in_lags = []
    oos_lags_list = []

    for key, data in results.items():
        if "error" in data and "validation" not in data:
            continue
        v = data.get("validation", {})
        if not v:
            continue
        label = f"{data['major']} -> {data['minor']}"
        systems.append(label)
        in_sample_corrs.append(abs(v.get("in_sample_peak_corr", 0)))
        oos_corrs.append(abs(v.get("oos_peak_corr", 0)))
        in_lags.append(v.get("in_sample_lag", 0))
        oos_lags_list.append(v.get("oos_lag", 0))

        # Max Granger F
        g_data = data.get("granger", [])
        if g_data:
            granger_max_f.append(max(g["f_statistic"] for g in g_data))
        else:
            granger_max_f.append(0)

    if not systems:
        return output_dir / "summary_dashboard.png"

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colours = SYSTEM_COLOURS[:len(systems)]

    # --- Panel A: Peak CCF Correlation ---
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(systems))
    w = 0.32
    ax1.bar(x - w / 2, in_sample_corrs, w, color=PALETTE["primary"],
            alpha=0.8, label="In-Sample", edgecolor="white", linewidth=0.5)
    ax1.bar(x + w / 2, oos_corrs, w, color=PALETTE["secondary"],
            alpha=0.8, label="Out-of-Sample", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontsize=8)
    ax1.set_ylabel("$|\\hat{\\rho}(\\tau^*)|$")
    ax1.set_title("(a) Peak Cross-Correlation", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=7.5, framealpha=0.9)

    # --- Panel B: Granger F-statistic (max) ---
    ax2 = fig.add_subplot(gs[0, 1])
    bars_g = ax2.bar(x, granger_max_f, 0.55, color=colours, alpha=0.8,
                     edgecolor="white", linewidth=0.5)
    for i, (bar, fval) in enumerate(zip(bars_g, granger_max_f)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{fval:.1f}", ha="center", va="bottom", fontsize=8,
                 fontweight="medium", color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, fontsize=8)
    ax2.set_ylabel("F-statistic")
    ax2.set_title("(b) Max Granger F-Statistic (Lag 1)", fontweight="bold", fontsize=11)

    # --- Panel C: Lag Stability ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - w / 2, in_lags, w, color=PALETTE["primary"],
            alpha=0.8, label="In-Sample $\\tau$", edgecolor="white", linewidth=0.5)
    ax3.bar(x + w / 2, oos_lags_list, w, color=PALETTE["secondary"],
            alpha=0.8, label="OOS $\\tau$", edgecolor="white", linewidth=0.5)
    for i, (il, ol) in enumerate(zip(in_lags, oos_lags_list)):
        stable = abs(il - ol) <= 3
        marker = "STABLE" if stable else "DRIFT"
        col = PALETTE["tertiary"] if stable else PALETTE["accent"]
        ax3.text(i, max(il, ol) + 0.5, marker, ha="center", fontsize=7.5,
                 fontweight="bold", color=col)
    ax3.set_xticks(x)
    ax3.set_xticklabels(systems, fontsize=8)
    ax3.set_ylabel("$\\tau^*$ (minutes)")
    ax3.set_title("(c) Lag Stability Across Quarters", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=7.5, framealpha=0.9)

    # --- Panel D: Summary table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    table_data = []
    col_labels = ["System", "$\\tau_{IS}$", "$\\tau_{OOS}$", "Stable?",
                  "$\\rho_{IS}$", "Granger F"]
    for i, s in enumerate(systems):
        stable_str = "Yes" if abs(in_lags[i] - oos_lags_list[i]) <= 3 else "No"
        table_data.append([
            s,
            f"{in_lags[i]} min",
            f"{oos_lags_list[i]} min",
            stable_str,
            f"{in_sample_corrs[i]:.4f}",
            f"{granger_max_f[i]:.1f}",
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=[PALETTE["light_fill"]] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8.5)
        cell.set_edgecolor("#cccccc")

    # Style data rows
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#dddddd")
            # Highlight stability column
            if j == 3:
                if table_data[i][3] == "Yes":
                    cell.set_facecolor("#d5efdb")
                else:
                    cell.set_facecolor("#fdd0a2")

    ax4.set_title("(d) Results Summary", fontweight="bold", fontsize=11, pad=15)

    fig.suptitle(
        "Crypto Lead-Lag Analysis:  Gravitational System Diagnostics",
        fontsize=14, fontweight="bold", y=0.98,
    )

    out = output_dir / "summary_dashboard.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 6.  CCF Heatmap — all systems on one plot
# ---------------------------------------------------------------------------
def plot_ccf_heatmap(
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Heatmap of mean CCF across systems vs. lag."""
    _apply_style()

    systems = []
    mean_ccf_arrays = []

    for key, data in results.items():
        if "ccf" not in data or not data["ccf"]:
            continue
        label = f"{data['major']} -> {data['minor']}"
        systems.append(label)

        # Average correlations across epochs
        all_corrs = [np.array(epoch["correlations"]) for epoch in data["ccf"]]
        mean_corr = np.mean(all_corrs, axis=0)
        mean_ccf_arrays.append(mean_corr)

    if not systems:
        return output_dir / "ccf_heatmap.png"

    lags = np.array(data["ccf"][0]["lags"])  # same for all
    ccf_matrix = np.array(mean_ccf_arrays)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Use a diverging colourmap centered at zero
    vmax = np.max(np.abs(ccf_matrix[:, lags != 0]))  # exclude lag-0 for scale
    im = ax.imshow(
        ccf_matrix, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=[lags[0] - 0.5, lags[-1] + 0.5, len(systems) - 0.5, -0.5],
        interpolation="nearest",
    )

    # Vertical line at lag = 0
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=9)
    ax.set_xlabel("Lag $\\tau$ (minutes)")
    ax.set_title("Mean Cross-Correlation Heatmap Across Systems",
                 fontsize=12, fontweight="bold")

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("$\\hat{\\rho}(\\tau)$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Mark optimal lags with arrows
    for i, (key, data) in enumerate(
        [(k, v) for k, v in results.items() if "ccf" in v and v["ccf"]]
    ):
        opt_lags = [ep["optimal_lag"] for ep in data["ccf"]]
        from collections import Counter
        mode_lag = Counter(opt_lags).most_common(1)[0][0]
        ax.plot(mode_lag, i, marker="v", color=PALETTE["accent"],
                markersize=8, zorder=5)
        ax.annotate(f"$\\tau^*$={mode_lag}", xy=(mode_lag, i),
                    xytext=(mode_lag + 6, i - 0.15),
                    fontsize=7.5, color=PALETTE["accent"], fontweight="bold")

    fig.tight_layout()
    out = output_dir / "ccf_heatmap.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ===================================================================
# Main entry point
# ===================================================================
def generate_all_figures(
    results_path: Path,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Load results JSON and generate all figures.

    Parameters
    ----------
    results_path : Path to ``full_results.json``
    output_dir : directory for figures (defaults to ``pipeline_results/figures/``)

    Returns
    -------
    List of paths to generated figures.
    """
    with open(results_path) as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = results_path.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    figures: List[Path] = []

    # --- Per-system figures ---
    valid_systems = []
    in_sample_lags = []
    oos_lags_all = []
    stability_flags = []
    predicted_bps_all = []

    for key, data in results.items():
        if "ccf" not in data or not data["ccf"]:
            continue

        label = f"{data['major']} -> {data['minor']}"

        # CCF correlogram
        fig_path = plot_ccf_correlogram(label, data["ccf"], output_dir)
        figures.append(fig_path)
        print(f"  [+] {fig_path.name}")

        # Granger F-stats
        if "granger" in data and data["granger"]:
            fig_path = plot_granger_fstats(label, data["granger"], 0.05, output_dir)
            figures.append(fig_path)
            print(f"  [+] {fig_path.name}")

        # Collect validation data
        v = data.get("validation", {})
        if v:
            valid_systems.append(label)
            in_sample_lags.append(v.get("in_sample_lag", 0))
            oos_lags_all.append(v.get("oos_lag", 0))
            stability_flags.append(v.get("lag_stable", False))

            # Mean predicted bps across friction checks
            friction = v.get("friction_checks", [])
            if friction:
                mean_pred = np.mean([fc["predicted_move_bps"] for fc in friction])
            else:
                mean_pred = 0.0
            predicted_bps_all.append(mean_pred)

    # --- Cross-system figures ---
    if valid_systems:
        fig_path = plot_lag_stability(
            valid_systems, in_sample_lags, oos_lags_all, stability_flags, output_dir
        )
        figures.append(fig_path)
        print(f"  [+] {fig_path.name}")

        fig_path = plot_friction_analysis(
            valid_systems, predicted_bps_all, 57.0, output_dir
        )
        figures.append(fig_path)
        print(f"  [+] {fig_path.name}")

    # Heatmap
    fig_path = plot_ccf_heatmap(results, output_dir)
    figures.append(fig_path)
    print(f"  [+] {fig_path.name}")

    # Summary dashboard
    fig_path = plot_summary_dashboard(results, output_dir)
    figures.append(fig_path)
    print(f"  [+] {fig_path.name}")

    return figures
# ---------------------------------------------------------------------------
# 7.  Robustness Data Gaps plots
# ---------------------------------------------------------------------------
def plot_robustness_comparison(
    system_label: str,
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Plot the degradation of CCF metrics under random data gaps."""
    _apply_style()
    
    fractions = []
    opt_lags = []
    corrs = []
    lags_std = []
    corrs_std = []
    
    if results.get("baseline"):
        bl_lags = [ep.optimal_lag for ep in results["baseline"]]
        bl_corrs = [ep.peak_correlation for ep in results["baseline"]]
        bl_corr_mean = np.mean(bl_corrs)
        import statistics
        bl_lag_mode = statistics.mode(bl_lags) if bl_lags else 0
        fractions.append(0.0)
        opt_lags.append(bl_lag_mode)
        corrs.append(bl_corr_mean)
        lags_std.append(np.std(bl_lags))
        corrs_std.append(np.std(bl_corrs))
        
    random_res = results.get("random", {})
    for frac_str in sorted(random_res.keys(), key=float):
        frac = float(frac_str)
        eps = random_res[frac_str]
        if eps:
            c_vals = [ep.peak_correlation for ep in eps]
            l_vals = [ep.optimal_lag for ep in eps]
            import statistics
            l_mode = statistics.mode(l_vals) if l_vals else 0
            
            fractions.append(frac)
            opt_lags.append(l_mode)
            corrs.append(np.mean(c_vals))
            lags_std.append(np.std(l_vals))
            corrs_std.append(np.std(c_vals))
            
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    
    color = PALETTE["primary"]
    ax1.set_xlabel("Missing Data Fraction")
    ax1.set_ylabel("Peak Cross-Correlation $\\rho$", color=color)
    ax1.plot(fractions, corrs, marker="o", linestyle="-", color=color, linewidth=2)
    ax1.fill_between(fractions, np.array(corrs) - np.array(corrs_std), np.array(corrs) + np.array(corrs_std), color=color, alpha=0.2, label="$\pm 1 \sigma$ (Correlation)")
    ax1.tick_params(axis='y', labelcolor=color)
    if corrs:
        ax1.set_ylim(0, max(np.array(corrs) + np.array(corrs_std)) * 1.1)
    
    ax2 = ax1.twinx()  
    color2 = PALETTE["secondary"]
    ax2.set_ylabel("Optimal Lag $\\tau^*$ (minutes)", color=color2)  
    ax2.plot(fractions, opt_lags, marker="s", linestyle="--", color=color2, linewidth=2)
    ax2.fill_between(fractions, np.array(opt_lags) - np.array(lags_std), np.array(opt_lags) + np.array(lags_std), color=color2, alpha=0.2, label="$\pm 1 \sigma$ (Lag)")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color("#333333")
    
    plt.title(f"Random Drop-out Robustness: {system_label}", fontweight="bold", fontsize=12)
    fig.tight_layout()
    
    out = output_dir / f"robustness_random_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out

def plot_robustness_periodic(
    system_label: str,
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Plot the degradation of CCF metrics under periodic data gaps."""
    _apply_style()
    
    periodic_res = results.get("periodic", {})
    if not periodic_res:
         return output_dir / f"robustness_periodic_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
         
    labels = []
    opt_lags = []
    corrs = []
    
    if results.get("baseline"):
        bl_lags = [ep.optimal_lag for ep in results["baseline"]]
        bl_corrs = [ep.peak_correlation for ep in results["baseline"]]
        import statistics
        bl_lag_mode = statistics.mode(bl_lags) if bl_lags else 0
        labels.append("Baseline (0m)")
        opt_lags.append(bl_lag_mode)
        corrs.append(np.mean(bl_corrs))

    for key, eps in periodic_res.items():
        if eps:
            c_vals = [ep.peak_correlation for ep in eps]
            l_vals = [ep.optimal_lag for ep in eps]
            import statistics
            l_mode = statistics.mode(l_vals) if l_vals else 0
            
            labels.append(key)
            corrs.append(np.mean(c_vals))
            opt_lags.append(l_mode)
            
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = PALETTE["primary"]
    rects1 = ax1.bar(x - width/2, corrs, width, label='Peak Correlation $\\rho$', color=color, alpha=0.8)
    ax1.set_ylabel("Peak Cross-Correlation $\\rho$", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    ax2 = ax1.twinx()  
    color2 = PALETTE["secondary"]
    rects2 = ax2.bar(x + width/2, opt_lags, width, label='Optimal Lag $\\tau^*$', color=color2, alpha=0.8)
    ax2.set_ylabel("Optimal Lag $\\tau^*$ (minutes)", color=color2)  
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color("#333333")
    
    for i, v in enumerate(opt_lags):
         ax2.text(i + width/2, v + 0.5, str(v), color='#333333', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title(f"Periodic Outage Robustness: {system_label}\nPeriod_Gap (mins)", fontweight="bold", fontsize=12)
    fig.tight_layout()

    out = output_dir / f"robustness_periodic_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    from pipeline.config import PATHS

    results_file = PATHS.root / "pipeline_results" / "full_results.json"
    print(f"Loading results from {results_file}...")
    figs = generate_all_figures(results_file)
    print(f"\nGenerated {len(figs)} figures.")
