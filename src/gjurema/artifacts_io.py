"""Escrita atômica dos artefatos do pipeline.

O dashboard lê os mesmos arquivos enquanto o pipeline roda; gravar em um
temporário no mesmo diretório e só então renomear evita leitura parcial.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def _temp_path(path: Path) -> Path:
    # Nome único por escrita (escritores simultâneos não compartilham temporário)
    # e extensão preservada: o XGBoost escolhe o formato pelo sufixo do arquivo.
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def write_with(path: Path, writer: Callable[[Path], Any]) -> Path:
    """Aplica `writer` a um caminho temporário e publica o resultado.

    Se a escrita falhar, o arquivo anterior continua válido e o temporário some.
    """
    temp = _temp_path(path)
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    return write_with(path, lambda temp: frame.to_parquet(temp, index=False))


def write_json(payload: Any, path: Path) -> Path:
    return write_with(path, lambda temp: temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)))
