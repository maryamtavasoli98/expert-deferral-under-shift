#!/usr/bin/env python3
"""Download BloodMNIST (35 MB), DermaMNIST (20 MB) and OrganAMNIST (38 MB) from Zenodo; write dataset.json."""

import hashlib
import urllib.request

import numpy as np

from expert_deferral.common import RAW, config, environment, write_json


def main():
    cfg = config()
    RAW.mkdir(parents=True, exist_ok=True)
    info = {"environment": environment(), "files": {}}
    for name, url in cfg["data_urls"].items():
        path = RAW / f"{name}.npz"
        if not path.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        z = np.load(path)
        info["files"][name] = {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k: list(z[k].shape) for k in z.files if k.endswith("images")},
            "classes": {k: int(len(np.unique(z[k]))) for k in z.files if k == "train_labels"},
        }
    b = np.load(RAW / "bloodmnist.npz")
    info["bloodmnist_train_class_counts"] = np.bincount(b["train_labels"].ravel()).tolist()
    write_json("dataset.json", info)
    print("ok")


if __name__ == "__main__":
    main()
