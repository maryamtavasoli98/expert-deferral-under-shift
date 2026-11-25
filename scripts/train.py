#!/usr/bin/env python3
"""Train ensemble_size CNNs for every seed on BloodMNIST (member 0 is the 'single model')."""

import json

import torch

from expert_deferral.common import config, environment, write_json
from expert_deferral.data import load_npz, to_tensor
from expert_deferral.pipeline import forward, save_member, train_member


def main():
    cfg = config()
    d = load_npz("bloodmnist")
    xtr, ytr = to_tensor(d["train_images"]), torch.from_numpy(d["train_labels"].ravel()).long()
    xva, yva = to_tensor(d["val_images"]), torch.from_numpy(d["val_labels"].ravel()).long()
    xte, yte = to_tensor(d["test_images"]), torch.from_numpy(d["test_labels"].ravel()).long()
    runs = []
    for seed in cfg["seeds"]:
        for k in range(cfg["ensemble_size"]):
            model, info = train_member(xtr, ytr, xva, yva, cfg, 1000 * seed + k)
            info.update(seed=seed, member=k, test_acc=(forward(model, xte).argmax(1) == yte).float().mean().item())
            save_member(model, seed, k)
            runs.append(info)
            print(json.dumps(info), flush=True)
    write_json("train.json", {"config": cfg, "environment": environment(), "runs": runs})


if __name__ == "__main__":
    main()
