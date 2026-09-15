"""Camada 3 — motor financeiro: cap rate, yield, TIR, payback e ROI."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gjurema import finance
from gjurema.config import DEFAULT_OPERATING_ASSUMPTIONS


@dataclass
class Assumptions:
    """Premissas operacionais e macroeconômicas do investimento."""

    vacancy_rate: float = DEFAULT_OPERATING_ASSUMPTIONS["vacancy_rate"]
    iptu_rate_on_value: float = DEFAULT_OPERATING_ASSUMPTIONS["iptu_rate_on_value"]
    condo_monthly_rate_on_rent: float = DEFAULT_OPERATING_ASSUMPTIONS["condo_monthly_rate_on_rent"]
    maintenance_rate_on_rent: float = DEFAULT_OPERATING_ASSUMPTIONS["maintenance_rate_on_rent"]
    management_rate_on_rent: float = DEFAULT_OPERATING_ASSUMPTIONS["management_rate_on_rent"]
    income_tax_rate_on_rent: float = DEFAULT_OPERATING_ASSUMPTIONS["income_tax_rate_on_rent"]
    transaction_cost_rate: float = DEFAULT_OPERATING_ASSUMPTIONS["transaction_cost_rate"]
    rent_adjustment_rate: float = 0.045  # reajuste anual do aluguel (IGP-M projetado)
    appreciation_rate: float = 0.05  # valorização anual do imóvel
    selic_rate: float = 0.105  # custo de oportunidade da renda fixa

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def cap_rate(annual_net_income: float, market_value: float) -> float:
    """Cap rate = renda operacional líquida anual ÷ valor de mercado."""
    if market_value <= 0:
        raise ValueError("market_value deve ser positivo")
    return annual_net_income / market_value


def gross_yield(monthly_rent: float, purchase_price: float) -> float:
    """Yield bruto anual sobre o preço de compra."""
    if purchase_price <= 0:
        raise ValueError("purchase_price deve ser positivo")
    return (monthly_rent * 12) / purchase_price


def net_operating_income(
    monthly_rent: float,
    market_value: float,
    assumptions: Assumptions | None = None,
) -> float:
    """Renda anual líquida após vacância, IPTU, condomínio, gestão e IR."""
    assumptions = assumptions or Assumptions()
    effective_rent = monthly_rent * 12 * (1 - assumptions.vacancy_rate)
    condo = monthly_rent * 12 * assumptions.condo_monthly_rate_on_rent * assumptions.vacancy_rate
    maintenance = effective_rent * assumptions.maintenance_rate_on_rent
    management = effective_rent * assumptions.management_rate_on_rent
    iptu = market_value * assumptions.iptu_rate_on_value
    taxable = max(effective_rent - management - iptu, 0.0)
    income_tax = taxable * assumptions.income_tax_rate_on_rent
    return effective_rent - condo - maintenance - management - iptu - income_tax


def net_yield(
    monthly_rent: float,
    purchase_price: float,
    market_value: float | None = None,
    assumptions: Assumptions | None = None,
) -> float:
    """Yield líquido anual sobre o preço de compra."""
    market_value = market_value if market_value is not None else purchase_price
    return net_operating_income(monthly_rent, market_value, assumptions) / purchase_price


def cash_flows(
    purchase_price: float,
    monthly_rent: float,
    years: int,
    assumptions: Assumptions | None = None,
    down_payment_rate: float = 1.0,
    financing_rate: float = 0.11,
    financing_years: int = 30,
) -> np.ndarray:
    """Fluxo de caixa anual do investimento, incluindo a venda no ano final.

    Ano 0 concentra entrada e custos de transação; os anos seguintes trazem
    a renda líquida de aluguel menos o serviço da dívida; o último ano soma
    o valor de revenda líquido do saldo devedor.
    """
    assumptions = assumptions or Assumptions()
    down_payment = purchase_price * down_payment_rate
    financed = purchase_price - down_payment
    transaction_costs = purchase_price * assumptions.transaction_cost_rate

    annual_debt_service = 0.0
    if financed > 0:
        annual_debt_service = finance.pmt(financing_rate, financing_years, financed)

    flows = [-(down_payment + transaction_costs)]
    for year in range(1, years + 1):
        rent = monthly_rent * (1 + assumptions.rent_adjustment_rate) ** (year - 1)
        market_value = purchase_price * (1 + assumptions.appreciation_rate) ** year
        income = net_operating_income(rent, market_value, assumptions)
        debt_service = annual_debt_service if year <= financing_years else 0.0
        flows.append(income - debt_service)

    exit_value = purchase_price * (1 + assumptions.appreciation_rate) ** years
    outstanding = 0.0
    if financed > 0:
        remaining = max(financing_years - years, 0)
        if remaining > 0:
            outstanding = finance.pv(financing_rate, remaining, annual_debt_service)
    flows[-1] += exit_value * (1 - assumptions.transaction_cost_rate) - outstanding
    return np.array(flows, dtype=float)


def irr(flows: np.ndarray) -> float:
    """Taxa interna de retorno anual do fluxo de caixa."""
    return finance.irr(flows)


def npv(flows: np.ndarray, discount_rate: float) -> float:
    """Valor presente líquido do fluxo de caixa."""
    return finance.npv(discount_rate, flows)


def payback_years(purchase_price: float, monthly_rent: float, assumptions: Assumptions | None = None) -> float:
    """Anos necessários para recuperar o capital via renda líquida de aluguel."""
    assumptions = assumptions or Assumptions()
    invested = purchase_price * (1 + assumptions.transaction_cost_rate)
    accumulated = 0.0
    year = 0
    while accumulated < invested and year < 100:
        year += 1
        rent = monthly_rent * (1 + assumptions.rent_adjustment_rate) ** (year - 1)
        market_value = purchase_price * (1 + assumptions.appreciation_rate) ** year
        accumulated += net_operating_income(rent, market_value, assumptions)
    return float("inf") if year >= 100 else year


def simulate(
    purchase_price: float,
    monthly_rent: float,
    horizons: tuple[int, ...] = (5, 10, 15),
    assumptions: Assumptions | None = None,
    down_payment_rate: float = 1.0,
    financing_rate: float = 0.11,
) -> dict:
    """Simulador de ROI: indicadores e retorno total por horizonte."""
    assumptions = assumptions or Assumptions()
    noi = net_operating_income(monthly_rent, purchase_price, assumptions)

    results = {
        "cap_rate": cap_rate(noi, purchase_price),
        "yield_bruto": gross_yield(monthly_rent, purchase_price),
        "yield_liquido": net_yield(monthly_rent, purchase_price, assumptions=assumptions),
        "payback_anos": payback_years(purchase_price, monthly_rent, assumptions),
        "horizontes": {},
    }

    for years in horizons:
        flows = cash_flows(
            purchase_price,
            monthly_rent,
            years,
            assumptions,
            down_payment_rate=down_payment_rate,
            financing_rate=financing_rate,
        )
        invested = -flows[0]
        total_return = float(flows[1:].sum())
        results["horizontes"][years] = {
            "tir": irr(flows),
            "vpl_selic": npv(flows, assumptions.selic_rate),
            "capital_investido": float(invested),
            "retorno_total": total_return,
            "multiplo": total_return / invested if invested else float("nan"),
            "renda_fixa_equivalente": float(invested * ((1 + assumptions.selic_rate) ** years - 1)),
        }
    return results


def scenarios(
    purchase_price: float,
    monthly_rent: float,
    years: int = 10,
    base: Assumptions | None = None,
    down_payment_rate: float = 1.0,
    financing_rate: float = 0.11,
) -> dict[str, dict]:
    """Cenários conservador, base e otimista de valorização e juros."""
    base = base or Assumptions()
    variants = {
        "conservador": {
            "appreciation_rate": max(base.appreciation_rate - 0.03, 0.0),
            "vacancy_rate": base.vacancy_rate + 0.04,
        },
        "base": {},
        "otimista": {
            "appreciation_rate": base.appreciation_rate + 0.03,
            "vacancy_rate": max(base.vacancy_rate - 0.03, 0.0),
        },
    }

    output = {}
    for name, overrides in variants.items():
        assumptions = Assumptions(**{**base.as_dict(), **overrides})
        flows = cash_flows(
            purchase_price,
            monthly_rent,
            years,
            assumptions,
            down_payment_rate=down_payment_rate,
            financing_rate=financing_rate,
        )
        output[name] = {
            "tir": irr(flows),
            "vpl_selic": npv(flows, assumptions.selic_rate),
            "valorizacao_aa": assumptions.appreciation_rate,
            "vacancia": assumptions.vacancy_rate,
        }
    return output
