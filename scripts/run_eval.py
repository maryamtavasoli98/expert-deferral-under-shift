#!/usr/bin/env python3
"""Evaluate every method on clean + shifted test data, OOD sets and simulated expert deferral.

Writes results/metrics/{calibration,ood,deferral,ensemble_size}.json.
"""

import time

import numpy as np
import torch

from expert_deferral import metrics as M
from expert_deferral.common import config, environment, write_json
from expert_deferral.data import corrupt, load_npz, noise_images, pooled_shift, to_tensor
from expert_deferral.pipeline import Methods, load_member

STREAM_COND = {"clean": "clean", "shift_s3": "pooled_s3", "shift_s5": "pooled_s5"}
METHODS = ["softmax", "temp_scaling", "mc_dropout", "ensemble", "laplace"]


def conditions(xte, cfg):
    out = {"clean": xte}
    for k in cfg["corruptions"]:
        for s in cfg["severities"]:
            out[f"{k}_s{s}"] = corrupt(xte, k, s, seed=s)
    for s in cfg["severities"]:
        out[f"pooled_s{s}"] = pooled_shift(xte, cfg["corruptions"], s, seed=100 + s)
    return out


def main():
    cfg = config()
    t0 = time.time()
    d = load_npz("bloodmnist")
    xtr, ytr = to_tensor(d["train_images"]), torch.from_numpy(d["train_labels"].ravel()).long()
    xva, yva = to_tensor(d["val_images"]), torch.from_numpy(d["val_labels"].ravel()).long()
    xte, yte = to_tensor(d["test_images"]), d["test_labels"].ravel().astype(int)
    conds = conditions(xte, cfg)
    n_te = len(xte)
    rng = np.random.default_rng(0)
    ood = {
        "dermamnist": to_tensor(load_npz("dermamnist")["test_images"]),
        "organamnist": to_tensor(load_npz("organamnist")["test_images"]),
    }
    ood["organamnist"] = ood["organamnist"][rng.choice(len(ood["organamnist"]), n_te, replace=False)]
    ood["noise"] = noise_images(n_te, seed=7)

    calib, oodres, defer, ens_size, rc = [], [], [], [], []
    grid = np.round(np.arange(0.3, 1.0001, 0.05), 2)
    for seed in cfg["seeds"]:
        members = [load_member(seed, k, 8) for k in range(cfg["ensemble_size"])]
        meth = Methods(members, xtr, ytr, xva, yva, cfg)
        print(f"seed {seed}: T={meth.temp:.3f} laplace prior={meth.lap_prior}", flush=True)
        cache = {}
        for cname, x in conds.items():
            for m in METHODS:
                p, mi = meth.predict(m, x)
                correct = (p.argmax(1) == yte).astype(float)
                ent = M.entropy(p)
                calib.append(
                    dict(
                        seed=seed,
                        method=m,
                        condition=cname,
                        acc=float(correct.mean()),
                        ece=M.ece(p, yte, cfg["ece_bins"]),
                        nll=M.nll(p, yte),
                        brier=M.brier(p, yte),
                        aurc=M.aurc(ent, correct),
                        mean_conf=float(p.max(1).mean()),
                    )
                )
                if cname in STREAM_COND.values():
                    cache[(m, cname)] = (p, correct, ent)
        for stream in cfg["streams"]:
            for m in METHODS:
                p, correct, ent = cache[(m, STREAM_COND[stream])]
                if True:
                    cov, risk = M.risk_coverage(ent, correct)
                    rc.append(
                        dict(
                            seed=seed,
                            method=m,
                            stream=stream,
                            coverage=grid.tolist(),
                            risk=[float(risk[int(round(g * len(risk))) - 1]) for g in grid],
                        )
                    )
                    for r in cfg["expert_reliability"]:
                        for b in cfg["budgets"]:
                            defer.append(
                                dict(
                                    seed=seed,
                                    method=m,
                                    stream=stream,
                                    kind="budget",
                                    reliability=r,
                                    budget=b,
                                    system_acc=M.system_accuracy(ent, correct, b, r),
                                    oracle_acc=M.oracle_system_accuracy(correct, b, r),
                                    random_acc=float(correct.mean() * (1 - b) + r * b),
                                )
                            )
                        acc, frac = M.threshold_policy(p, correct, r)
                        defer.append(
                            dict(
                                seed=seed,
                                method=m,
                                stream=stream,
                                kind="threshold",
                                reliability=r,
                                system_acc=acc,
                                deferred=frac,
                            )
                        )
        # OOD detection: in-distribution = clean test set
        p_id = {m: meth.predict(m, conds["clean"]) for m in METHODS}
        for oname, xo in ood.items():
            for m in METHODS:
                p_o = meth.predict(m, xo)
                oodres.append(
                    dict(
                        seed=seed,
                        method=m,
                        ood=oname,
                        score="entropy",
                        auroc=M.ood_auroc(M.entropy(p_id[m][0]), M.entropy(p_o[0])),
                    )
                )
                oodres.append(
                    dict(
                        seed=seed,
                        method=m,
                        ood=oname,
                        score="max_prob",
                        auroc=M.ood_auroc(1 - p_id[m][0].max(1), 1 - p_o[0].max(1)),
                    )
                )
                if p_o[1] is not None:
                    oodres.append(
                        dict(seed=seed, method=m, ood=oname, score="mutual_info", auroc=M.ood_auroc(p_id[m][1], p_o[1]))
                    )
        # ensemble-size ablation
        for size in range(1, cfg["ensemble_size"] + 1):
            for cname in ["clean", "pooled_s3", "pooled_s5"]:
                p, _ = meth.predict(f"ensemble_{size}", conds[cname])
                correct = (p.argmax(1) == yte).astype(float)
                ens_size.append(
                    dict(
                        seed=seed,
                        size=size,
                        condition=cname,
                        acc=float(correct.mean()),
                        ece=M.ece(p, yte),
                        nll=M.nll(p, yte),
                        aurc=M.aurc(M.entropy(p), correct),
                    )
                )
        meta = dict(seed=seed, temperature=meth.temp, laplace_prior=meth.lap_prior, laplace_val_nll=meth.lap_val_nll)
        calib.append(dict(meta=meta))
        print(f"seed {seed} done at {time.time() - t0:.0f}s", flush=True)

    env = {**environment(), "eval_seconds": time.time() - t0}
    metas = [c["meta"] for c in calib if "meta" in c]
    calib = [c for c in calib if "meta" not in c]
    write_json("calibration.json", {"config": cfg, "environment": env, "post_hoc": metas, "records": calib})
    write_json("ood.json", {"config": cfg, "records": oodres})
    write_json("deferral.json", {"config": cfg, "records": defer})
    write_json("risk_coverage.json", {"records": rc})
    write_json("ensemble_size.json", {"records": ens_size})


if __name__ == "__main__":
    main()
