import numpy as np

from expert_deferral import metrics as M


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    y = np.zeros(n, int)
    correct = rng.uniform(size=n) < conf
    p = np.zeros((n, 2))
    p[:, 0] = np.where(correct, conf, 1 - conf)
    p[:, 1] = 1 - p[:, 0]
    # predicted class 0 has probability conf only when it is "correct"; rebuild properly
    pred = np.where(correct, 0, 1)
    p = np.zeros((n, 2))
    p[np.arange(n), pred] = conf
    p[np.arange(n), 1 - pred] = 1 - conf
    assert M.ece(p, y) < 0.02
    over = np.tile([0.99, 0.01], (n, 1))
    assert M.ece(over, rng.integers(0, 2, n)) > 0.4


def test_nll_brier_known_values():
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 1])
    assert abs(M.nll(p, y) - (-(np.log(0.8) + np.log(0.7)) / 2)) < 1e-9
    assert abs(M.brier(p, y) - ((0.04 + 0.04) + (0.09 + 0.09)) / 2) < 1e-9


def test_ood_auroc_perfect_and_chance():
    assert M.ood_auroc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
    rng = np.random.default_rng(1)
    assert abs(M.ood_auroc(rng.normal(size=5000), rng.normal(size=5000)) - 0.5) < 0.03


def test_risk_coverage_and_aurc():
    score = np.array([0.1, 0.2, 0.3, 0.9])
    correct = np.array([1.0, 1.0, 1.0, 0.0])
    cov, risk = M.risk_coverage(score, correct)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0, 0, 0, 0.25])
    assert M.aurc(score, correct) < M.aurc(-score, correct)


def test_system_accuracy_budget_and_oracle():
    score = np.array([0.1, 0.2, 0.3, 0.9])
    correct = np.array([1.0, 1.0, 1.0, 0.0])
    assert M.system_accuracy(score, correct, 0.0, 0.9) == 0.75
    assert abs(M.system_accuracy(score, correct, 0.25, 0.9) - (3 + 0.9) / 4) < 1e-12
    assert M.system_accuracy(-score, correct, 0.25, 0.9) < M.system_accuracy(score, correct, 0.25, 0.9)
    assert M.oracle_system_accuracy(correct, 0.25, 0.9) >= M.system_accuracy(score, correct, 0.25, 0.9)


def test_threshold_policy_defers_low_confidence_only():
    p = np.array([[0.99, 0.01], [0.6, 0.4]])
    correct = np.array([1.0, 0.0])
    acc, frac = M.threshold_policy(p, correct, 0.9)
    assert frac == 0.5 and abs(acc - (1 + 0.9) / 2) < 1e-12
