import torch

from expert_deferral.data import PARAMS, corrupt, pooled_shift
from expert_deferral.laplace import LastLayerLaplace, augment, ggn_hessian


def test_ggn_equals_autograd_hessian_of_softmax_ce():
    torch.manual_seed(0)
    n, d, c = 12, 4, 3
    phi = torch.randn(n, d, dtype=torch.float64)
    w = torch.randn(c, d + 1, dtype=torch.float64)
    logits = augment(phi) @ w.T
    h = ggn_hessian(phi, torch.softmax(logits, -1)).reshape(c * (d + 1), c * (d + 1))

    def loss(wf):
        return torch.nn.functional.cross_entropy(
            augment(phi) @ wf.reshape(c, d + 1).T, torch.zeros(n, dtype=torch.long), reduction="sum"
        )

    ref = torch.autograd.functional.hessian(loss, w.reshape(-1))
    assert torch.allclose(h, ref, atol=1e-8)  # label-independent for softmax CE


def test_laplace_widens_with_weaker_prior_and_far_inputs():
    torch.manual_seed(0)
    phi = torch.randn(200, 5)
    w, b = torch.randn(3, 5), torch.zeros(3)
    probs = torch.softmax(phi @ w.T + b, -1)
    lap = LastLayerLaplace(w, b, ggn_hessian(phi, probs))
    lap.set_prior(100.0)
    near = lap.logit_stats(torch.zeros(1, 5))[1].diagonal(dim1=1, dim2=2).sum()
    far = lap.logit_stats(10 * torch.ones(1, 5))[1].diagonal(dim1=1, dim2=2).sum()
    assert far > near
    lap.set_prior(1.0)
    weak = lap.logit_stats(10 * torch.ones(1, 5))[1].diagonal(dim1=1, dim2=2).sum()
    assert weak > far  # smaller prior precision = weaker prior = wider posterior
    p = lap.predict(phi[:4], 50)
    assert p.shape == (4, 3) and torch.allclose(p.sum(1), torch.ones(4), atol=1e-5)


def test_corruptions_valid_and_monotone():
    torch.manual_seed(0)
    x = torch.rand(16, 3, 28, 28)
    for kind in PARAMS:
        d = [(corrupt(x, kind, s) - x).abs().mean().item() for s in range(1, 6)]
        assert all(0 <= v <= 1 for v in d)
        assert d[-1] > d[0], kind
        assert corrupt(x, kind, 3).min() >= 0 and corrupt(x, kind, 3).max() <= 1


def test_pooled_shift_deterministic_and_shape():
    x = torch.rand(30, 3, 28, 28)
    a = pooled_shift(x, list(PARAMS), 3, seed=1)
    b = pooled_shift(x, list(PARAMS), 3, seed=1)
    assert a.shape == x.shape and torch.equal(a, b)
