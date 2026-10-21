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
