import pytest

from gjurema.models import returns


def test_gross_yield():
    assert returns.gross_yield(3_000, 600_000) == pytest.approx(0.06)


def test_net_yield_below_gross_yield():
    gross = returns.gross_yield(3_000, 600_000)
    net = returns.net_yield(3_000, 600_000)
    assert 0 < net < gross


def test_cap_rate_rejects_non_positive_value():
    with pytest.raises(ValueError):
        returns.cap_rate(10_000, 0)


def test_cash_flows_length_and_sign():
    flows = returns.cash_flows(600_000, 3_000, years=5)
    assert len(flows) == 6
    assert flows[0] < 0
    assert flows[-1] > 0  # inclui a revenda do imóvel


def test_financing_reduces_upfront_capital():
    cash = returns.cash_flows(600_000, 3_000, years=5, down_payment_rate=1.0)
    levered = returns.cash_flows(600_000, 3_000, years=5, down_payment_rate=0.3)
    assert -levered[0] < -cash[0]


def test_payback_shrinks_with_higher_rent():
    assert returns.payback_years(600_000, 5_000) < returns.payback_years(600_000, 2_000)


def test_simulate_returns_all_horizons():
    simulation = returns.simulate(600_000, 3_000)
    assert set(simulation["horizontes"]) == {5, 10, 15}
    assert simulation["horizontes"][10]["tir"] > 0


def test_scenarios_are_ordered_by_appreciation():
    result = returns.scenarios(600_000, 3_000)
    assert result["conservador"]["tir"] < result["base"]["tir"] < result["otimista"]["tir"]


def test_scenarios_respond_to_leverage():
    cash = returns.scenarios(600_000, 3_000)
    levered = returns.scenarios(600_000, 3_000, down_payment_rate=0.5)
    assert levered["base"]["tir"] != cash["base"]["tir"]


def test_simulate_and_scenarios_agree_on_base_case():
    simulation = returns.simulate(600_000, 3_000, down_payment_rate=0.5)
    scenario = returns.scenarios(600_000, 3_000, down_payment_rate=0.5)
    assert simulation["horizontes"][10]["tir"] == pytest.approx(scenario["base"]["tir"], abs=1e-6)
