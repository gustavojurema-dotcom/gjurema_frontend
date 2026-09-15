"""Resolução de bairro a partir do CEP (ViaCEP), com cache em disco.

A guia de ITBI não traz bairro confiável, então o recorte por bairro do
produto é reconstruído pelo CEP do imóvel. O cache evita repetir consultas
entre execuções do pipeline — são dezenas de milhares de CEPs distintos.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from gjurema import artifacts_io
from gjurema.config import RAW_DIR

logger = logging.getLogger(__name__)

VIACEP_URL = "https://viacep.com.br/ws/{cep}/json/"
CACHE_PATH = RAW_DIR / "ceps.json"
REQUEST_TIMEOUT = 15
MAX_WORKERS = 8


def _normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.upper().strip())
    return text.encode("ascii", "ignore").decode()


def load_cache(path: Path = CACHE_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _fetch(cep: str) -> tuple[str, str | None]:
    """Bairro do CEP, `""` para resposta negativa definitiva e `None` para falha de rede."""
    try:
        response = requests.get(VIACEP_URL.format(cep=cep), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - rede
        logger.debug("CEP %s indisponível: %s", cep, exc)
        return cep, None
    if payload.get("erro") or payload.get("localidade") != "São Paulo":
        return cep, ""
    return cep, _normalize(payload.get("bairro") or "")


def resolve(ceps: list[str], path: Path = CACHE_PATH) -> dict[str, str]:
    """Devolve {cep: bairro} consultando apenas os CEPs ainda não cacheados."""
    cache = load_cache(path)
    pending = sorted({cep for cep in ceps if cep and cep not in cache})
    if not pending:
        return cache

    logger.info("Resolvendo %s CEPs no ViaCEP", len(pending))
    falhas = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for cep, bairro in pool.map(_fetch, pending):
            # Falha de rede fica fora do cache para a próxima execução tentar de novo;
            # cachear o erro excluiria o CEP da base para sempre.
            if bairro is None:
                falhas += 1
                continue
            cache[cep] = bairro
    if falhas:
        logger.warning("%s CEPs não resolvidos por falha de rede — serão reconsultados", falhas)

    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_io.write_json(cache, path)
    return cache
