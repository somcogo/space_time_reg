import pytest

from stmr.config import Config
from stmr.registration import velocity_lr_factor


def test_const_schedule_is_flat_after_warmup():
    config = Config(lr=1e-4, epochs=10, lr_warmup=4)
    assert [velocity_lr_factor(config, s) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert all(velocity_lr_factor(config, s) == 1.0 for s in range(4, 10))


def test_cosine_decays_from_lr_to_lr_min_by_last_epoch():
    config = Config(lr=1e-4, lr_min=1e-6, lr_schedule="cosine", epochs=101, lr_warmup=0)
    lrs = [config.lr * velocity_lr_factor(config, s) for s in range(config.epochs)]
    assert lrs[0] == pytest.approx(1e-4)
    assert lrs[50] == pytest.approx(0.5 * (1e-4 + 1e-6))
    assert lrs[-1] == pytest.approx(1e-6)
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_cosine_starts_after_warmup():
    config = Config(lr=2e-4, lr_min=1e-5, lr_schedule="cosine", epochs=1000, lr_warmup=200)
    assert velocity_lr_factor(config, 0) == pytest.approx(1 / 200)
    assert velocity_lr_factor(config, 199) == pytest.approx(1.0)
    assert velocity_lr_factor(config, 200) == pytest.approx(1.0)
    assert config.lr * velocity_lr_factor(config, 999) == pytest.approx(1e-5)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        velocity_lr_factor(Config(lr_schedule="step"), 0)
