
import pytest

from stmr.config import Config


def test_unsigned_exponent_is_coerced_to_float(tmp_path):
    # PyYAML loads "1.0e4" (no exponent sign) as a str; from_dict must coerce it.
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("lambda_rl2: 1.0e4\nrecon_scale: 6\n")
    config = Config.from_yaml(cfg_file)
    assert isinstance(config.lambda_rl2, float)
    assert config.lambda_rl2 == 10000.0
    assert isinstance(config.recon_scale, float)


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"not_a_real_field": 1})


def test_finalize_normalises_coupled_fields():
    config = Config(recon_scale=0, step_size=0, interval=0, epochs=50).finalize()
    assert config.recon_scale is None
    assert config.step_size is None
    assert config.interval == 50


def test_soft_con_yaml_parses():
    config = Config.from_yaml("configs/cmr_soft_con.yaml").finalize()
    assert config.func_name == "groupsiren"
    assert config.loss == "mse"
    assert config.lambda_rl2 == pytest.approx(1e4)
