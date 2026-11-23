from pytest import approx

from readout_service.stats import ab_test


def test_lift_is_relative_change():
    result = ab_test(control=(100, 1000), treatment=(120, 1000))
    assert result["lift"] == approx(0.20)


def test_big_difference_is_significant():
    result = ab_test(control=(100, 1000), treatment=(150, 1000))
    assert result["p_value"] < 0.05
    low, high = result["ci_95"]
    assert low > 0


def test_zero_conversions_does_not_crash():
    result = ab_test(control=(0, 1000), treatment=(0, 1000))
    assert result["lift"] is None
    assert result["p_value"] == 1.0