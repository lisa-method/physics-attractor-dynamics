"""Plot fixed summary metrics from docs/FINAL_REPORT.md.

This presentation script does not read the dataset, execute the notebook,
train models, or recompute evaluation metrics. Values correspond to the saved
research results reported on 2026-09-04, sections 5 and 6 of FINAL_REPORT.md.

Run from the project root:
    uv run --no-sync python scripts/plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "assets" / "results.png"

# Sequential one-step: 36,664 full transitions, observed history available.
MATERIAL_FAILURES = {"TODO 8": 868, "TODO 10": 58, "TODO 11": 49}

# Autonomous rollout: RMSE over four coordinates and the entire horizon prefix,
# followed by equal-weight averaging over segments. TODO 11 was not evaluated.
# Horizon 100: 11 segments. Horizon 500: 10 segments.
ROLLOUT_RMSE = {
    "TODO 8": (8.6045, 95.3492),
    "TODO 10": (4.0930, 61.6632),
    "Постоянная позиция": (60.1758, 137.5219),
}


def main() -> None:
    colors = {
        "TODO 8": "#8296ad",
        "TODO 10": "#147d83",
        "TODO 11": "#7c61a8",
        "Постоянная позиция": "#d7a458",
    }
    ink = "#203044"
    muted = "#607084"

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": ink,
        "axes.labelcolor": muted,
        "xtick.color": muted,
        "ytick.color": muted,
        "axes.edgecolor": "#dce3eb",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "figure.facecolor": "#fbfcfe",
        "axes.facecolor": "#fbfcfe",
        "savefig.facecolor": "#fbfcfe",
    }):
        fig, (one_step, rollout) = plt.subplots(
            1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1, 1.28]}
        )
        fig.subplots_adjust(left=0.075, right=0.975, bottom=0.24, top=0.69, wspace=0.25)

        fig.text(0.055, 0.925, "Physics Attractor Dynamics", fontsize=24, weight="bold")
        fig.text(
            0.055, 0.868,
            "Результаты на исследованном наборе · меньше — лучше · независимого test нет",
            fontsize=12, color=muted,
        )

        one_step.set_title("Один шаг", loc="left", fontsize=18, weight="bold", pad=44)
        one_step.text(
            0, 1.065, "Наблюдённое прошлое доступно · 36 664 перехода",
            transform=one_step.transAxes, fontsize=11, color=muted,
        )
        bars = one_step.bar(
            list(MATERIAL_FAILURES), list(MATERIAL_FAILURES.values()),
            color=[colors[label] for label in MATERIAL_FAILURES], width=0.55,
        )
        one_step.bar_label(bars, padding=7, fontsize=14, weight="bold", color=ink)
        one_step.set_ylabel("Число material failures", labelpad=10)
        one_step.set_ylim(0, 1000)
        one_step.set_yticks([0, 250, 500, 750, 1000])
        one_step.grid(axis="y", color="#e4eaf1", linewidth=0.8)
        one_step.tick_params(axis="both", length=0, pad=8)

        rollout.set_title("Автономный rollout", loc="left", fontsize=18, weight="bold", pad=44)
        rollout.text(
            0, 1.065, "Модель использует собственные прогнозы",
            transform=rollout.transAxes, fontsize=11, color=muted,
        )
        positions = (0, 1)
        bar_width = 0.23
        for offset, (label, values) in zip((-1, 0, 1), ROLLOUT_RMSE.items()):
            bars = rollout.bar(
                [position + offset * bar_width for position in positions], values,
                width=bar_width * 0.88, label=label, color=colors[label],
            )
            rollout.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                              padding=5, fontsize=11, color=ink)
        rollout.set_xticks(positions, ["100 updates\n11 сегментов", "500 updates\n10 сегментов"])
        rollout.set_ylabel("Macro coordinate RMSE", labelpad=9)
        rollout.set_ylim(0, 165)
        rollout.set_yticks([0, 40, 80, 120, 160])
        rollout.grid(axis="y", color="#e4eaf1", linewidth=0.8)
        rollout.tick_params(axis="both", length=0, pad=8)
        rollout.legend(
            loc="upper left", bbox_to_anchor=(-0.01, -0.21), ncols=3,
            frameon=False, fontsize=10, handlelength=1.1, columnspacing=1.0,
        )

        fig.text(
            0.075, 0.146,
            "TODO 11: автономный rollout не выполнялся.",
            fontsize=11, weight="bold", color=colors["TODO 11"],
        )
        fig.text(
            0.055, 0.063,
            "Material failure: ошибка позиции ≥ 0.1, угла ≥ 1° или distance ≥ 0.1. "
            "RMSE: весь префикс, равный вес сегментов.",
            fontsize=10, color=muted,
        )
        fig.text(
            0.055, 0.027,
            "Источник: docs/FINAL_REPORT.md, разделы 5–6 · "
            "зафиксированные метрики; график не запускает обучение",
            fontsize=10, color=muted,
        )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=180)
        plt.close(fig)
    print(OUTPUT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
