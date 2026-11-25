"""Last-layer Laplace approximation with the full generalised Gauss-Newton Hessian (from scratch).

Weights W (C x (D+1), bias folded in) get a Gaussian posterior N(W_map, H^-1) with
H = sum_n (diag(p_n) - p_n p_n^T) (x) phi_n phi_n^T + lambda I. Logit means are W_map phi, logit covariances
phi^T V phi (V = H^-1 blocks); the predictive is a Monte-Carlo average of softmax over sampled logits.
"""

from __future__ import annotations

import torch


def augment(phi: torch.Tensor) -> torch.Tensor:
    return torch.cat([phi, torch.ones(len(phi), 1, dtype=phi.dtype)], dim=1)


def ggn_hessian(phi: torch.Tensor, probs: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Return H_data with shape (C, D1, C, D1) (float64)."""
    phi = augment(phi.double())
    p = probs.double()
    n, c = p.shape
    d1 = phi.shape[1]
    h = torch.zeros(c, d1, c, d1, dtype=torch.float64)
    for s in range(0, n, chunk):
        ph, pp = phi[s : s + chunk], p[s : s + chunk]
        a = torch.diag_embed(pp) - pp[:, :, None] * pp[:, None, :]  # (n, C, C)
        h += torch.einsum("nab,ni,nj->aibj", a, ph, ph)
    return h


class LastLayerLaplace:
    def __init__(self, w_map: torch.Tensor, b_map: torch.Tensor, h_data: torch.Tensor):
        self.w = torch.cat([w_map.double(), b_map.double()[:, None]], dim=1)  # (C, D1)
        self.h_data = h_data
        self.c, self.d1 = self.w.shape
        self.set_prior(1.0)

    def set_prior(self, lam: float) -> None:
        self.lam = lam
        h = self.h_data.reshape(self.c * self.d1, self.c * self.d1) + lam * torch.eye(
            self.c * self.d1, dtype=torch.float64
        )
        self.v = torch.linalg.inv(h).reshape(self.c, self.d1, self.c, self.d1)

    def logit_stats(self, phi: torch.Tensor, chunk: int = 512):
        phi = augment(phi.double())
        mu = phi @ self.w.T
        cov = torch.cat(
            [
                torch.einsum("ni,aibj,nj->nab", phi[s : s + chunk], self.v, phi[s : s + chunk])
                for s in range(0, len(phi), chunk)
            ]
        )
        return mu, cov

    def predict(self, phi: torch.Tensor, n_samples: int = 200, seed: int = 0, return_mi: bool = False):
        mu, cov = self.logit_stats(phi)
        cov = 0.5 * (cov + cov.transpose(1, 2)) + 1e-9 * torch.eye(self.c, dtype=torch.float64)
        chol = torch.linalg.cholesky(cov)
        g = torch.Generator().manual_seed(seed)
        eps = torch.randn(n_samples, *mu.shape, generator=g, dtype=torch.float64)
        logits = mu[None] + torch.einsum("nab,snb->sna", chol, eps)
        probs = torch.softmax(logits, dim=-1)
        mean = probs.mean(0).float()
        if not return_mi:
            return mean
        h = lambda q: -(q * torch.log(q.clamp_min(1e-12))).sum(-1)  # noqa: E731
        return mean, (h(mean) - h(probs.float()).mean(0)).clamp_min(0)
