"""Training and the five predictive methods compared in the study."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from .common import RUNS, device
from .laplace import LastLayerLaplace, ggn_hessian
from .models import SmallCNN, set_mc_dropout


@torch.no_grad()
def forward(model: SmallCNN, x: torch.Tensor, bs: int = 1024, what: str = "logits") -> torch.Tensor:
    dev = device()
    out = []
    for s in range(0, len(x), bs):
        xb = x[s : s + bs].to(dev)
        out.append((model.features(xb) if what == "features" else model(xb)).cpu())
    return torch.cat(out)


def train_member(xtr, ytr, xva, yva, cfg: dict, seed: int) -> tuple[SmallCNN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = SmallCNN(n_classes=int(ytr.max()) + 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=-(-len(xtr) // cfg["batch_size"])
    )
    best, best_state, t0 = 1e9, None, time.time()
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(xtr))
        for s in range(0, len(xtr), cfg["batch_size"]):
            idx = perm[s : s + cfg["batch_size"]]
            loss = F.cross_entropy(model(xtr[idx].to(dev)), ytr[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        val_nll = F.cross_entropy(forward(model, xva), yva).item()
        if val_nll < best:
            best, best_state = val_nll, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    acc = (forward(model, xva).argmax(1) == yva).float().mean().item()
    return model, {"val_nll": best, "val_acc": acc, "seconds": time.time() - t0}


def save_member(model: SmallCNN, seed: int, k: int) -> None:
    RUNS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"s{seed}_m{k}.pt")


def load_member(seed: int, k: int, n_classes: int) -> SmallCNN:
    m = SmallCNN(n_classes=n_classes)
    m.load_state_dict(torch.load(RUNS / f"s{seed}_m{k}.pt", map_location="cpu"))
    return m.to(device()).eval()


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    f = lambda t: F.cross_entropy(logits / np.exp(t), y).item()  # noqa: E731
    return float(np.exp(minimize_scalar(f, bounds=(-2, 3), method="bounded").x))


class Methods:
    """All predictors of one seed. ``predict(method, x)`` returns (probs, mutual_information_or_None) as numpy."""

    def __init__(self, members: list[SmallCNN], xtr, ytr, xva, yva, cfg: dict):
        self.members, self.cfg = members, cfg
        base = members[0]
        self.temp = fit_temperature(forward(base, xva), yva)
        # last-layer Laplace: GGN on the training set, prior precision picked by clean validation NLL
        ftr = forward(base, xtr, what="features")
        with torch.no_grad():
            ptr = torch.softmax(base.head(ftr.to(device())).cpu(), -1)
        self.lap = LastLayerLaplace(
            base.head.weight.detach().cpu(), base.head.bias.detach().cpu(), ggn_hessian(ftr, ptr)
        )
        fva = forward(base, xva, what="features")
        best = None
        for lam in cfg["laplace_prior_grid"]:
            self.lap.set_prior(lam)
            n = F.nll_loss(torch.log(self.lap.predict(fva, 100).clamp_min(1e-12)), yva).item()
            if best is None or n < best[0]:
                best = (n, lam)
        self.lap.set_prior(best[1])
        self.lap_prior, self.lap_val_nll = best[1], best[0]

    def predict(self, method: str, x: torch.Tensor):
        c = self.cfg
        base = self.members[0]
        if method == "softmax":
            return torch.softmax(forward(base, x), -1).numpy(), None
        if method == "temp_scaling":
            return torch.softmax(forward(base, x) / self.temp, -1).numpy(), None
        if method == "mc_dropout":
            ps = []
            for t in range(c["mc_dropout_samples"]):
                torch.manual_seed(t)
                set_mc_dropout(base, True)
                ps.append(torch.softmax(forward(base, x), -1))
            base.eval()
            return _mean_mi(torch.stack(ps))
        if method.startswith("ensemble"):
            m = int(method.split("_")[1]) if "_" in method else len(self.members)
            return _mean_mi(torch.stack([torch.softmax(forward(mm, x), -1) for mm in self.members[:m]]))
        if method == "laplace":
            p, mi = self.lap.predict(forward(base, x, what="features"), c["laplace_samples"], return_mi=True)
            return p.numpy(), mi.numpy()
        raise ValueError(method)


def _mean_mi(stack: torch.Tensor):
    mean = stack.mean(0)
    h = lambda q: -(q * torch.log(q.clamp_min(1e-12))).sum(-1)  # noqa: E731
    return mean.numpy(), (h(mean) - h(stack).mean(0)).clamp_min(0).numpy()
