#!/usr/bin/env python3
"""Figures: calibration vs shift severity, risk-coverage, expert-budget curves."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from expert_deferral.common import FIGURES, read_json  # noqa: E402
from expert_deferral.summary import group  # noqa: E402

METHODS = ["softmax", "temp_scaling", "mc_dropout", "ensemble", "laplace"]
COL = {
    "softmax": "#888888",
    "temp_scaling": "#1b9e77",
    "mc_dropout": "#d95f02",
    "ensemble": "#7570b3",
    "laplace": "#e7298a",
}
LAB = {
    "softmax": "single softmax",
    "temp_scaling": "temp. scaling",
    "mc_dropout": "MC-dropout",
    "ensemble": "ensemble (5)",
    "laplace": "last-layer Laplace",
}


def band(ax, x, g, key_fn, m):
    mu = [g[key_fn(v)][0] for v in x]
    sd = [g[key_fn(v)][1] for v in x]
    ax.plot(x, mu, "-o", ms=3, color=COL[m], label=LAB[m])
    ax.fill_between(x, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)], color=COL[m], alpha=0.15)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    cal = read_json("calibration.json")["records"]
    # 1 calibration vs severity
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, metric, title in zip(
        axes, ["ece", "nll", "acc"], ["ECE (lower is better)", "NLL (lower is better)", "accuracy"]
    ):
        g = group(cal, ["method", "condition"], metric)
        for m in METHODS:
            conds = ["clean"] + [f"pooled_s{s}" for s in range(1, 6)]
            ax.plot(range(6), [g[(m, c)][0] for c in conds], "-o", ms=3, color=COL[m], label=LAB[m])
            ax.fill_between(
                range(6),
                [g[(m, c)][0] - g[(m, c)][1] for c in conds],
                [g[(m, c)][0] + g[(m, c)][1] for c in conds],
                color=COL[m],
                alpha=0.15,
            )
        ax.set_xlabel("shift severity (0 = clean)")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_vs_shift.png", dpi=150)
    plt.close(fig)
    # 2 risk-coverage
    rc = read_json("risk_coverage.json")["records"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    for ax, st in zip(axes, ["clean", "shift_s3", "shift_s5"]):
        for m in METHODS:
            rs = [r for r in rc if r["method"] == m and r["stream"] == st]
            cov = rs[0]["coverage"]
            mu = [sum(r["risk"][i] for r in rs) / len(rs) for i in range(len(cov))]
            ax.plot(cov, mu, "-", color=COL[m], label=LAB[m])
        ax.set_title(f"stream: {st}")
        ax.set_xlabel("coverage (fraction answered by the model)")
    axes[0].set_ylabel("selective risk (error rate of answered cases)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "risk_coverage.png", dpi=150)
    plt.close(fig)
    # 3 expert budget curves
    d = [r for r in read_json("deferral.json")["records"] if r["kind"] == "budget"]
    budgets = sorted({r["budget"] for r in d})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (st, r_) in zip(axes, [("shift_s5", 1.0), ("shift_s5", 0.9), ("shift_s5", 0.8)]):
        sub = [r for r in d if r["stream"] == st and r["reliability"] == r_]
        g = group(sub, ["method", "budget"], "system_acc")
        for m in METHODS:
            band(ax, budgets, g, lambda b, m=m: (m, b), m)
        go = group(sub, ["budget"], "oracle_acc")
        ax.plot(budgets, [go[(b,)][0] for b in budgets], "k--", lw=1, label="oracle")
        ax.set_title(f"{st}, expert reliability {r_}")
        ax.set_xlabel("expert budget (fraction of stream)")
    axes[0].set_ylabel("expected system accuracy")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "expert_budget.png", dpi=150)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
