"""Calibration, OOD and selective-prediction metrics, all written from scratch (numpy)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error on the top-label confidence, equal-width bins."""
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def ood_auroc(score_id: np.ndarray, score_ood: np.ndarray) -> float:
    """AUROC for separating OOD (positive) from in-distribution by an uncertainty score."""
    y = np.r_[np.zeros(len(score_id)), np.ones(len(score_ood))]
    return float(roc_auc_score(y, np.r_[score_id, score_ood]))


def risk_coverage(score: np.ndarray, correct: np.ndarray):
    """Accept the most certain first. Returns coverage (k/N) and selective risk (error among accepted)."""
    order = np.argsort(score, kind="stable")
    err = 1.0 - correct[order]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_coverage(score, correct)[1].mean())


def system_accuracy(score: np.ndarray, correct: np.ndarray, budget: float, reliability: float) -> float:
    """Expected accuracy when the ``budget`` fraction with the highest uncertainty goes to an expert who is right
    with probability ``reliability`` (wrong answers are not modelled further: they count as errors)."""
    n = len(score)
    k = int(round(budget * n))
    order = np.argsort(-score, kind="stable")
    deferred = np.zeros(n, bool)
    deferred[order[:k]] = True
    return float((correct[~deferred].sum() + reliability * deferred.sum()) / n)


def oracle_system_accuracy(correct: np.ndarray, budget: float, reliability: float) -> float:
    """Upper bound: the expert is asked about the model's errors first."""
    return system_accuracy(1.0 - correct + 1e-9 * np.arange(len(correct)) / len(correct), correct, budget, reliability)


def threshold_policy(p: np.ndarray, correct: np.ndarray, reliability: float):
    """Defer iff the model's top probability is below the expert's reliability (Bayes-optimal if calibrated).
    Returns (system accuracy, fraction deferred)."""
    defer = p.max(1) < reliability
    acc = (correct[~defer].sum() + reliability * defer.sum()) / len(correct)
    return float(acc), float(defer.mean())
