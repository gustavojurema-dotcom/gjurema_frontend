"""Dados socioeconômicos municipais do IBGE (população e PIB)."""

from __future__ import annotations

import logging
import unicodedata

import pandas as pd
import requests

from gjurema.config import (
    HTTP_TIMEOUT,
    IBGE_AGREGADOS_URL,
    IBGE_GDP_AGGREGATE,
    IBGE_GDP_VARIABLE,
    IBGE_LOCALIDADES_URL,
    IBGE_POPULATION_AGGREGATE,
    IBGE_POPULATION_VARIABLE,
)

logger = logging.getLogger(__name__)

# UF de cada cidade coberta pelo FipeZAP — necessário para desambiguar
# homônimos (São José, Santa Maria, Campo Grande, entre outros).
CITY_UF = {
    "São Paulo": "SP",
    "Barueri": "SP",
    "Campinas": "SP",
    "Diadema": "SP",
    "Guarujá": "SP",
    "Guarulhos": "SP",
    "Osasco": "SP",
    "Praia Grande": "SP",
    "Ribeirão Preto": "SP",
    "Santo André": "SP",
    "Santos": "SP",
    "São Bernardo do Campo": "SP",
    "São Caetano do Sul": "SP",
    "São José do Rio Preto": "SP",
    "São José dos Campos": "SP",
    "São Vicente": "SP",
    "Rio de Janeiro": "RJ",
    "Niterói": "RJ",
    "Belo Horizonte": "MG",
    "Betim": "MG",
    "Contagem": "MG",
    "Porto Alegre": "RS",
    "Canoas": "RS",
    "Caxias do Sul": "RS",
    "Novo Hamburgo": "RS",
    "Pelotas": "RS",
    "Santa Maria": "RS",
    "São Leopoldo": "RS",
    "Curitiba": "PR",
    "Londrina": "PR",
    "São José dos Pinhais": "PR",
    "Florianópolis": "SC",
    "Balneário Camboriú": "SC",
    "Blumenau": "SC",
    "Itajaí": "SC",
    "Itapema": "SC",
    "Joinville": "SC",
    "São José": "SC",
    "Vitória": "ES",
    "Vila Velha": "ES",
    "Brasília": "DF",
    "Goiânia": "GO",
    "Campo Grande": "MS",
    "Cuiabá": "MT",
    "Aracaju": "SE",
    "Fortaleza": "CE",
    "João Pessoa": "PB",
    "Maceió": "AL",
    "Natal": "RN",
    "Recife": "PE",
    "Jaboatão dos Guararapes": "PE",
    "Salvador": "BA",
    "São Luís": "MA",
    "Teresina": "PI",
    "Belém": "PA",
    "Manaus": "AM",
}


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.casefold().strip()


def municipalities() -> pd.DataFrame:
    """Lista os municípios brasileiros com código IBGE, nome e UF."""
    response = requests.get(f"{IBGE_LOCALIDADES_URL}/municipios", timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    records = [
        {
            "ibge_code": item["id"],
            "city": item["nome"],
            "uf": item["microrregiao"]["mesorregiao"]["UF"]["sigla"],
        }
        for item in response.json()
        if item.get("microrregiao")
    ]
    frame = pd.DataFrame(records)
    frame["city_key"] = frame["city"].map(_normalize)
    return frame


def resolve_codes(cities: list[str]) -> pd.DataFrame:
    """Mapeia nomes de cidades do FipeZAP para códigos municipais do IBGE."""
    catalog = municipalities()
    rows = []
    for city in cities:
        uf = CITY_UF.get(city)
        candidates = catalog[catalog["city_key"] == _normalize(city)]
        if uf:
            candidates = candidates[candidates["uf"] == uf]
        if candidates.empty:
            logger.warning("Município não resolvido no IBGE: %s", city)
            continue
        rows.append({"city": city, "uf": candidates.iloc[0]["uf"], "ibge_code": int(candidates.iloc[0]["ibge_code"])})
    return pd.DataFrame(rows)


def _fetch_aggregate(aggregate: int, variable: int, period: str, codes: list[int]) -> dict[int, float]:
    localities = ",".join(str(code) for code in codes)
    url = f"{IBGE_AGREGADOS_URL}/{aggregate}/periodos/{period}/variaveis/{variable}"
    response = requests.get(url, params={"localidades": f"N6[{localities}]"}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    values: dict[int, float] = {}
    for variable_block in payload:
        for result in variable_block.get("resultados", []):
            for series in result.get("series", []):
                code = int(series["localidade"]["id"])
                raw = next(iter(series["serie"].values()), None)
                try:
                    values[code] = float(raw)
                except (TypeError, ValueError):
                    continue
    return values


def load_socioeconomic(
    cities: list[str],
    population_period: str = "2021",
    gdp_period: str = "2021",
) -> pd.DataFrame:
    """Devolve população, PIB e PIB per capita por cidade."""
    codes = resolve_codes(cities)
    if codes.empty:
        return pd.DataFrame(columns=["city", "uf", "ibge_code", "population", "gdp_brl_thousand", "gdp_per_capita"])

    code_list = codes["ibge_code"].tolist()
    population = _fetch_aggregate(IBGE_POPULATION_AGGREGATE, IBGE_POPULATION_VARIABLE, population_period, code_list)
    gdp = _fetch_aggregate(IBGE_GDP_AGGREGATE, IBGE_GDP_VARIABLE, gdp_period, code_list)

    codes["population"] = codes["ibge_code"].map(population)
    codes["gdp_brl_thousand"] = codes["ibge_code"].map(gdp)
    codes["gdp_per_capita"] = (codes["gdp_brl_thousand"] * 1_000) / codes["population"]
    return codes
