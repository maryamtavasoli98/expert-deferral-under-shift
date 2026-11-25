#!/usr/bin/env python3
"""Fill the ``<!-- table:NAME -->`` blocks of README.md and report/report.md from results/metrics/*.json."""

import re

from expert_deferral.common import ROOT, config, read_json
from expert_deferral.summary import group

TARGETS = [ROOT / "README.md", ROOT / "report" / "report.md"]
METHODS = ["softmax", "temp_scaling", "mc_dropout", "ensemble", "laplace"]
NAMES = {
    "softmax": "single softmax",
    "temp_scaling": "temperature scaling",
    "mc_dropout": "MC-dropout (30)",
    "ensemble": "deep ensemble (5)",
    "laplace": "last-layer Laplace",
}


def pm(t, d=3):
    return f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def table_calibration():
    g = {
        m: group(read_json("calibration.json")["records"], ["method", "condition"], m)
        for m in ["acc", "ece", "nll", "aurc"]
    }
    rows = ["| method | condition | accuracy | ECE ↓ | NLL ↓ | AURC ↓ |", "|---|---|---|---|---|---|"]
    for cond, label in [
        ("clean", "clean test"),
        ("pooled_s3", "mixed shift, severity 3"),
        ("pooled_s5", "mixed shift, severity 5"),
    ]:
        for m in METHODS:
            k = (m, cond)
            rows.append(
                f"| {NAMES[m]} | {label} | {pm(g['acc'][k])} | {pm(g['ece'][k])} | {pm(g['nll'][k])} | {pm(g['aurc'][k])} |"
            )
    return "\n".join(rows)


def table_ood():
    recs = read_json("ood.json")["records"]
    g = group(recs, ["method", "ood", "score"], "auroc")
    oods = ["dermamnist", "organamnist", "noise"]
    rows = [
        "| method | score | near-OOD: DermaMNIST | near-OOD: OrganAMNIST | far-OOD: uniform noise |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        for sc in ["max_prob", "entropy", "mutual_info"]:
            if (m, oods[0], sc) in g:
                rows.append(f"| {NAMES[m]} | {sc} | " + " | ".join(pm(g[(m, o, sc)]) for o in oods) + " |")
    return "\n".join(rows)


def table_deferral():
    recs = [
        r
        for r in read_json("deferral.json")["records"]
        if r["kind"] == "budget" and r["reliability"] == 0.9 and r["budget"] in (0.1, 0.2)
    ]
    g = group(recs, ["method", "stream", "budget"], "system_acc")
    o = group(recs, ["stream", "budget"], "oracle_acc")
    rnd = group(recs, ["method", "stream", "budget"], "random_acc")
    rows = [
        "| stream | expert budget | no expert | random deferral | "
        + " | ".join(NAMES[m] for m in METHODS)
        + " | oracle |",
        "|---" * (5 + len(METHODS)) + "|",
    ]
    base = group([r for r in read_json("calibration.json")["records"]], ["method", "condition"], "acc")
    for st, cond in [("clean", "clean"), ("shift_s3", "pooled_s3"), ("shift_s5", "pooled_s5")]:
        for b in (0.1, 0.2):
            best = max(METHODS, key=lambda m: g[(m, st, b)][0])
            cells = [(f"**{pm(g[(m, st, b)])}**" if m == best else pm(g[(m, st, b)])) for m in METHODS]
            rows.append(
                f"| {st} | {int(b * 100)} % | {base[('softmax', cond)][0]:.3f} | {rnd[('softmax', st, b)][0]:.3f} | "
                + " | ".join(cells)
                + f" | {o[(st, b)][0]:.3f} |"
            )
    return "\n".join(rows)


def table_threshold():
    recs = [r for r in read_json("deferral.json")["records"] if r["kind"] == "threshold" and r["stream"] == "shift_s5"]
    acc = group(recs, ["method", "reliability"], "system_acc")
    frac = group(recs, ["method", "reliability"], "deferred")
    rels = config()["expert_reliability"]
    rows = [
        "| method | " + " | ".join(f"r = {r}: system acc / deferred" for r in rels) + " |",
        "|---" * (1 + len(rels)) + "|",
    ]
    for m in METHODS:
        rows.append(
            f"| {NAMES[m]} | "
            + " | ".join(f"{acc[(m, r)][0]:.3f} / {100 * frac[(m, r)][0]:.0f} %" for r in rels)
            + " |"
        )
    return "\n".join(rows)


def table_ensemble():
    g = {m: group(read_json("ensemble_size.json")["records"], ["size", "condition"], m) for m in ["ece", "nll", "aurc"]}
    rows = [
        "| members | ECE clean | ECE shift 5 | NLL shift 5 | AURC clean | AURC shift 5 |",
        "|---|---|---|---|---|---|",
    ]
    for s in range(1, 6):
        rows.append(
            f"| {s} | {pm(g['ece'][(s, 'clean')])} | {pm(g['ece'][(s, 'pooled_s5')])} | {pm(g['nll'][(s, 'pooled_s5')])} | {pm(g['aurc'][(s, 'clean')])} | {pm(g['aurc'][(s, 'pooled_s5')])} |"
        )
    return "\n".join(rows)


def table_setup():
    t = read_json("train.json")
    runs = t["runs"]
    import numpy as np

    acc = np.mean([r["test_acc"] for r in runs])
    sec = np.mean([r["seconds"] for r in runs])
    c = read_json("calibration.json")
    post = c["post_hoc"]
    temps = ", ".join(f"{p['temperature']:.2f}" for p in post)
    lams = ", ".join(f"{p['laplace_prior']:g}" for p in post)
    return (
        f"- {len(runs)} CNNs ({len(t['config']['seeds'])} seeds x {t['config']['ensemble_size']} members), mean clean test accuracy {acc:.3f}, {sec:.0f} s per model on `{t['environment']['device']}`.\n"
        f"- Fitted temperatures per seed: {temps}. Laplace prior precision chosen on clean validation NLL per seed: {lams}.\n"
        f"- Software: torch {c['environment']['torch']} on {c['environment']['machine']}."
    )


TABLES = {
    "calibration": table_calibration,
    "ood": table_ood,
    "deferral": table_deferral,
    "threshold": table_threshold,
    "ensemble": table_ensemble,
    "setup": table_setup,
}


def main():
    built = {k: f() for k, f in TABLES.items()}
    for path in TARGETS:
        if not path.exists():
            continue
        s = path.read_text()
        for k, v in built.items():
            s = re.sub(
                rf"(<!-- table:{k} -->).*?(<!-- /table:{k} -->)",
                lambda m: f"{m.group(1)}\n{v}\n{m.group(2)}",
                s,
                flags=re.S,
            )
        path.write_text(s)
        print("updated", path)


if __name__ == "__main__":
    main()
