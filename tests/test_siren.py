import torch

from stmr.models.siren import GroupedSiren, Siren


def test_last_init_std_gives_random_but_tiny_velocity():
    # A normal last-layer init keeps the weights random (not all zero) while the initial
    # velocity is far below the U(+-1/600) default's (expected ratio ~ 1e-6 / (1/600/sqrt(3)) ~ 1e-3).
    torch.manual_seed(1)
    x = torch.rand(3, 500, 2) * 2 - 1
    torch.manual_seed(0)
    small = GroupedSiren(groups=3, layers=[2, 64, 64, 2], last_init_zero=True, last_init_std=1e-6)
    torch.manual_seed(0)
    default = GroupedSiren(groups=3, layers=[2, 64, 64, 2], last_init_zero=True)
    w = small.layers[-1].weight
    assert (w != 0).all()
    assert abs(w.std().item() - 1e-6) < 2e-7
    with torch.no_grad():
        assert small(0., x).abs().max() < 1e-2 * default(0., x).abs().max()


def test_siren_last_init_std_matches_grouped():
    torch.manual_seed(0)
    net = Siren(layers=[2, 64, 64, 2], last_init_zero=True, last_init_std=1e-6)
    assert abs(net.layers[-1].weight.std().item() - 1e-6) < 3e-7
    assert (net.layers[-1].bias == 0).all()


def test_grouped_siren_matches_per_group_reference():
    # GroupedLinear chunks the points and fuses omega/bias into one baddbmm for speed; output and
    # gradients must equal independent per-group SIRENs, sin(omega * (x @ W^T + b)).
    torch.manual_seed(0)
    net = GroupedSiren(groups=3, layers=[2, 16, 16, 2], omega=30).double()
    for p in net.parameters():
        p.data.normal_(0, 0.1)
    x = (torch.rand(3, 512, 2, dtype=torch.float64) * 2 - 1).requires_grad_()

    def reference(x):
        out = []
        for g in range(3):
            h = x[g]
            for i, layer in enumerate(net.layers):
                h = h @ layer.weight[g].T + layer.bias[g]
                if i < len(net.layers) - 1:
                    h = torch.sin(net.omega * h)
            out.append(h)
        return torch.stack(out)

    out, ref = net(0., x), reference(x)
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    grads = torch.autograd.grad(out.square().sum(), [x, *net.parameters()])
    ref_grads = torch.autograd.grad(ref.square().sum(), [x, *net.parameters()])
    for g, r in zip(grads, ref_grads):
        assert torch.allclose(g, r, rtol=1e-10, atol=1e-12)
