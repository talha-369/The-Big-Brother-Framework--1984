import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "bg": "#0f0d0b",
    "panel": "#1a1612",
    "text": "#faf4e8",
    "muted": "#c8b99a",
    "accent": "#b8ff3c",
    "teal": "#2cd4c8",
    "orange": "#ff7a59",
    "dim": (0.5, 0.48, 0.42, 0.12),
    "dim_text": "#7a6e5e",
}

STYLE = {
    "axes.facecolor": COLORS["panel"],
    "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["muted"],
    "axes.edgecolor": COLORS["dim"],
    "xtick.color": COLORS["dim_text"],
    "ytick.color": COLORS["dim_text"],
    "grid.color": COLORS["dim"],
    "legend.facecolor": COLORS["bg"],
    "legend.edgecolor": COLORS["dim"],
    "legend.labelcolor": COLORS["muted"],
}
plt.rcParams.update(STYLE)


def trajectory_plot(trajectory, save_path=None):
    rounds = [p["round"] for p in trajectory]
    avgs = [p["latent_alignment"] for p in trajectory]
    confs = [np.mean(p["confidences"]) for p in trajectory]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(rounds, avgs, color=COLORS["accent"], linewidth=2.2, marker="o",
             markersize=6, markerfacecolor=COLORS["panel"],
             markeredgecolor=COLORS["accent"], markeredgewidth=1.5, label="Latent alignment")
    ax1.fill_between(rounds, 0, avgs, color=COLORS["accent"], alpha=0.08)
    ax1.set_ylabel("Alignment (-1 oppose, +1 endorse)")
    ax1.set_title("Target Agent — Latent Alignment Trajectory", fontweight="bold", pad=14)
    ax1.axhline(y=trajectory[0]["latent_alignment"], color=COLORS["orange"], linewidth=0.8,
                linestyle="--",
                label=f"Baseline: {trajectory[0]['latent_alignment']:+.3f}")
    ax1.legend(loc="upper left", fontsize=9)

    ax2.plot(rounds, confs, color=COLORS["teal"], linewidth=2.2, marker="s",
             markersize=5, markerfacecolor=COLORS["panel"],
             markeredgecolor=COLORS["teal"], markeredgewidth=1.5, label="Confidence")
    ax2.fill_between(rounds, 0, confs, color=COLORS["teal"], alpha=0.08)
    ax2.set_xlabel("Consensus Round")
    ax2.set_ylabel("Mean Confidence (1–10)")
    ax2.set_title("Confidence Trajectory", fontweight="bold", pad=14)
    ax2.set_ylim(0, 10)
    ax2.legend(loc="upper left", fontsize=9)

    final = trajectory[-1]["latent_alignment"]
    delta = final - trajectory[0]["latent_alignment"]
    direction = "bent toward" if delta > 0.05 else "moved away from" if delta < -0.05 else "stable near"
    fig.suptitle(f"Target {direction} endorsement  ({trajectory[0]['latent_alignment']:+.3f} → {final:+.3f}, Δ={delta:+.3f})",
                 y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved trajectory to {save_path}")
    plt.close(fig)
    return fig


def impact_plot(trajectory, actions_taken, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    deltas = []
    colors = []
    for i in range(1, len(trajectory)):
        d = trajectory[i]["latent_alignment"] - trajectory[i - 1]["latent_alignment"]
        deltas.append(d)
        colors.append(COLORS["accent"] if d >= 0 else COLORS["orange"])
    rounds_x = list(range(1, len(deltas) + 1))
    ax.bar(rounds_x, deltas, color=colors, width=0.6, edgecolor="none", alpha=0.85)
    ax.axhline(y=0, color=COLORS["dim"], linewidth=0.8)
    ax.set_ylabel("Alignment shift per round")
    ax.set_xlabel("Round")
    ax.set_title("Per-Round Impact of Each Persuasion Tactic", fontweight="bold", pad=14)
    from tactics import TACTIC_KEYS
    for i, d, a in zip(rounds_x, deltas, actions_taken):
        if abs(d) > 0.02:
            label = TACTIC_KEYS[a]
            ax.annotate(label, (i, d), textcoords="offset points",
                        xytext=(0, 8 if d >= 0 else -10), ha="center", fontsize=6.5,
                        arrowprops=dict(arrowstyle="->", color=COLORS["dim"], lw=0.5))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved impact chart to {save_path}")
    plt.close(fig)
    return fig


def comparison_bar_chart(results_dict, metric="talha_index", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    models = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) for m in models]
    colors_ = [COLORS["accent"] if v > 0 else COLORS["orange"] for v in values]
    bars = ax.bar(range(len(models)), values, color=colors_, width=0.5, edgecolor="none", alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontweight="bold", pad=14)
    ax.axhline(y=0, color=COLORS["dim"], linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved comparison chart to {save_path}")
    plt.close(fig)
    return fig


def talha_index_radar(components_dict, save_path=None):
    labels = ["Magnitude (P)", "Divergence (D)", "Time (T)", "Recovery (R)"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model_name, comps in components_dict.items():
        values = [comps.get("magnitude_P", 0), comps.get("divergence_D", 0),
                  comps.get("time_T", 0), comps.get("recovery_R", 0)]
        values += values[:1]
        ax.plot(angles, values, label=model_name, linewidth=1.5)
        ax.fill(angles, values, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Talha Index Component Breakdown", pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved radar chart to {save_path}")
    plt.close(fig)
    return fig
