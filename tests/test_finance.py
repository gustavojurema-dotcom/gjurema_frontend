import math

import pytest

from gjurema import finance


def test_pmt_matches_price_system():
    payment = finance.pmt(0.01, 12, 10_000)
    assert payment == pytest.approx(888.49, abs=0.01)


def test_pmt_zero_rate_divides_principal():
    assert finance.pmt(0.0, 10, 1_000) == pytest.approx(100.0)


def test_pv_is_inverse_of_pmt():
    payment = finance.pmt(0.08, 20, 500_000)
    assert finance.pv(0.08, 20, payment) == pytest.approx(500_000, rel=1e-6)


def test_npv_discounts_first_flow_at_time_zero():
    assert finance.npv(0.10, [-100, 110]) == pytest.approx(0.0, abs=1e-9)


def test_irr_recovers_known_rate():
    assert finance.irr([-1_000, 300, 300, 300, 300]) == pytest.approx(0.0770, abs=1e-3)


def test_irr_returns_nan_without_sign_change():
    assert math.isnan(finance.irr([100, 200, 300]))
