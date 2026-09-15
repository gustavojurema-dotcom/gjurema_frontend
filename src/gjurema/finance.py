"""Primitivas financeiras (PMT, PV, VPL e TIR) sem dependências externas."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pmt(rate: float, periods: int, present_value: float) -> float:
    """Prestação constante (sistema Price) de um financiamento."""
    if periods <= 0:
        raise ValueError("periods deve ser positivo")
    if rate == 0:
        return present_value / periods
    factor = (1 + rate) ** periods
    return present_value * rate * factor / (factor - 1)


def pv(rate: float, periods: int, payment: float) -> float:
    """Valor presente de uma série de pagamentos constantes."""
    if periods <= 0:
        return 0.0
    if rate == 0:
        return payment * periods
    return payment * (1 - (1 + rate) ** -periods) / rate


def npv(rate: float, flows: Sequence[float]) -> float:
    """Valor presente líquido, com o primeiro fluxo no instante zero."""
    values = np.asarray(flows, dtype=float)
    discounts = (1 + rate) ** np.arange(len(values))
    return float(np.sum(values / discounts))


def irr(flows: Sequence[float], tolerance: float = 1e-7, max_iterations: int = 200) -> float:
    """TIR por bisseção; devolve NaN quando o fluxo não tem raiz no intervalo."""
    values = np.asarray(flows, dtype=float)
    if values.size < 2 or np.all(values >= 0) or np.all(values <= 0):
        return float("nan")

    low, high = -0.9999, 10.0
    npv_low, npv_high = npv(low, values), npv(high, values)
    if npv_low * npv_high > 0:
        return float("nan")

    for _ in range(max_iterations):
        middle = (low + high) / 2
        npv_middle = npv(middle, values)
        if abs(npv_middle) < tolerance:
            return float(middle)
        if npv_low * npv_middle < 0:
            high, npv_high = middle, npv_middle
        else:
            low, npv_low = middle, npv_middle
    return float((low + high) / 2)
