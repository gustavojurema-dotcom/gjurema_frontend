"""Carteira do cliente: esquema de campos, validação e persistência.

A carteira é dado privado (contrato, matrícula, aluguel) e vive fora do
ITBI. O esquema é servido para o painel montar o formulário de cadastro,
então o que é obrigatório para precificar está declarado num único lugar.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid

from gjurema import artifacts_io
from gjurema.config import CARTEIRA_PATH

SEGMENTOS = ["Apartamento", "Casa", "Sala comercial", "Loja", "Terreno", "Galpão", "Outro"]
QUARTOS = ["Studio", "1 dorm.", "2 dorm.", "3 dorm.", "4+ dorm.", "Não se aplica"]

CAMPOS: tuple[dict, ...] = (
    {"nome": "nome", "rotulo": "Empreendimento", "tipo": "texto", "obrigatorio": True,
     "ajuda": "Nome do edifício ou do imóvel"},
    {"nome": "unidade", "rotulo": "Unidade", "tipo": "texto", "obrigatorio": True,
     "ajuda": "Ex.: Unidade 702"},
    {"nome": "endereco", "rotulo": "Endereço (logradouro e número)", "tipo": "texto", "obrigatorio": True,
     "ajuda": "Usado para localizar o prédio no ITBI: 'Rua Capitão Prudente, 209'"},
    {"nome": "bairro", "rotulo": "Bairro", "tipo": "texto", "obrigatorio": True,
     "ajuda": "Como aparece no ITBI (maiúsculas, sem acento)"},
    {"nome": "cidade", "rotulo": "Cidade", "tipo": "texto", "obrigatorio": False, "padrao": "São Paulo"},
    {"nome": "cep", "rotulo": "CEP", "tipo": "texto", "obrigatorio": False},
    {"nome": "segmento", "rotulo": "Segmento", "tipo": "opcao", "obrigatorio": True,
     "opcoes": SEGMENTOS, "padrao": "Apartamento", "ajuda": "Segmento usado na comparação com o ITBI"},
    {"nome": "tipo", "rotulo": "Tipo", "tipo": "texto", "obrigatorio": False, "padrao": "Apartamento"},
    {"nome": "quartos", "rotulo": "Dormitórios", "tipo": "opcao", "obrigatorio": False, "opcoes": QUARTOS},
    {"nome": "vagas", "rotulo": "Vagas de garagem", "tipo": "inteiro", "obrigatorio": False},
    {"nome": "padrao", "rotulo": "Padrão construtivo (1 a 5)", "tipo": "numero", "obrigatorio": False,
     "padrao": 3, "ajuda": "Equivalente ao padrão do IPTU; 3 é o padrão médio"},
    {"nome": "priv", "rotulo": "Área privativa (m²)", "tipo": "numero", "obrigatorio": True},
    {"nome": "comum", "rotulo": "Área comum (m²)", "tipo": "numero", "obrigatorio": False},
    {"nome": "total", "rotulo": "Área total / construída (m²)", "tipo": "numero", "obrigatorio": True,
     "ajuda": "Base da comparação com o ITBI; se vazio, soma privativa + comum"},
    {"nome": "fracao", "rotulo": "Fração ideal", "tipo": "numero", "obrigatorio": False},
    {"nome": "matricula", "rotulo": "Matrícula e cartório", "tipo": "texto", "obrigatorio": False},
    {"nome": "preco", "rotulo": "Valor pago no imóvel (R$)", "tipo": "numero", "obrigatorio": True},
    {"nome": "dataCompra", "rotulo": "Data da compra", "tipo": "data", "obrigatorio": True},
    {"nome": "aluguelMensal", "rotulo": "Valor do aluguel atual (R$/mês)", "tipo": "numero",
     "obrigatorio": False, "ajuda": "Não existe no ITBI: a rentabilidade do aluguel só aparece se informado"},
    {"nome": "aluguelDesde", "rotulo": "Aluguel vigente desde", "tipo": "data", "obrigatorio": False},
    {"nome": "condominioMensal", "rotulo": "Condomínio (R$/mês)", "tipo": "numero", "obrigatorio": False},
    {"nome": "iptuAnual", "rotulo": "IPTU (R$/ano)", "tipo": "numero", "obrigatorio": False},
    {"nome": "construtora", "rotulo": "Construtora / incorporadora", "tipo": "texto", "obrigatorio": False},
    {"nome": "corretora", "rotulo": "Corretora", "tipo": "texto", "obrigatorio": False},
    {"nome": "corretagem", "rotulo": "Corretagem paga (R$)", "tipo": "numero", "obrigatorio": False},
    {"nome": "habitese", "rotulo": "Habite-se", "tipo": "texto", "obrigatorio": False},
    {"nome": "unidadesEmp", "rotulo": "Unidades do empreendimento", "tipo": "inteiro", "obrigatorio": False},
    {"nome": "fluxo", "rotulo": "Fluxo de pagamento", "tipo": "fluxo", "obrigatorio": False,
     "ajuda": "Séries do contrato: descrição e valor"},
)

OBRIGATORIOS = tuple(campo["nome"] for campo in CAMPOS if campo["obrigatorio"])
NUMERICOS = {campo["nome"] for campo in CAMPOS if campo["tipo"] in {"numero", "inteiro"}}
INTEIROS = {campo["nome"] for campo in CAMPOS if campo["tipo"] == "inteiro"}
TEXTOS = {campo["nome"] for campo in CAMPOS if campo["tipo"] in {"texto", "opcao", "data"}}


class ValidationError(ValueError):
    """Cadastro recusado: falta campo obrigatório ou o valor é inválido."""


def load() -> list[dict]:
    if not CARTEIRA_PATH.exists():
        return []
    return json.loads(CARTEIRA_PATH.read_text())


def save(itens: list[dict]) -> None:
    CARTEIRA_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifacts_io.write_json(itens, CARTEIRA_PATH)


def _number(valor: object, campo: str) -> float | int | None:
    if valor is None or valor == "":
        return None
    if isinstance(valor, (int, float)):
        numero = float(valor)
    else:
        texto = re.sub(r"[^\d,.-]", "", str(valor))
        # Formato brasileiro: ponto é milhar e vírgula é decimal.
        if "," in texto:
            texto = texto.replace(".", "").replace(",", ".")
        try:
            numero = float(texto)
        except ValueError as exc:
            raise ValidationError(f"campo {campo} não é numérico: {valor}") from exc
    if numero < 0:
        raise ValidationError(f"campo {campo} não pode ser negativo")
    return int(round(numero)) if campo in INTEIROS else numero


def _bairro(valor: str) -> str:
    """O ITBI grava bairro em maiúsculas e sem acento."""
    texto = unicodedata.normalize("NFKD", str(valor).upper().strip())
    return re.sub(r"\s+", " ", texto.encode("ascii", "ignore").decode())


def _fluxo(valor: object) -> list[list]:
    if not valor:
        return []
    series = []
    for linha in valor:
        if isinstance(linha, dict):
            descricao, montante = linha.get("descricao"), linha.get("valor")
        else:
            descricao, montante = (list(linha) + [None, None])[:2]
        if not descricao:
            continue
        series.append([str(descricao), _number(montante, "fluxo") or 0.0])
    return series


def validate(payload: dict, *, parcial: bool = False) -> dict:
    """Normaliza o cadastro vindo do painel e recusa o que impede precificar."""
    item: dict = {}
    for campo in CAMPOS:
        nome = campo["nome"]
        if nome not in payload:
            if not parcial and "padrao" in campo:
                item[nome] = campo["padrao"]
            continue
        valor = payload[nome]
        if nome == "fluxo":
            item[nome] = _fluxo(valor)
        elif nome in NUMERICOS:
            item[nome] = _number(valor, nome)
        elif nome == "bairro":
            item[nome] = _bairro(valor)
        elif nome in TEXTOS:
            item[nome] = str(valor).strip()

    if not parcial:
        item.setdefault("total", None)
        if not item.get("total") and item.get("priv"):
            item["total"] = float(item["priv"]) + float(item.get("comum") or 0)
        faltando = [c for c in OBRIGATORIOS if item.get(c) in (None, "", 0)]
        if faltando:
            rotulos = ", ".join(campo["rotulo"] for campo in CAMPOS if campo["nome"] in faltando)
            raise ValidationError(f"campos obrigatórios faltando: {rotulos}")
        if item.get("dataCompra") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item["dataCompra"]):
            raise ValidationError("dataCompra deve estar no formato AAAA-MM-DD")
    return item


def create(payload: dict) -> dict:
    item = validate(payload)
    item["id"] = payload.get("id") or f"u{uuid.uuid4().hex[:8]}"
    itens = load()
    if any(existente["id"] == item["id"] for existente in itens):
        raise ValidationError(f"já existe um imóvel {item['id']} na carteira")
    item.setdefault("fluxo", [])
    itens.append(item)
    save(itens)
    return item


def update(item_id: str, payload: dict) -> dict:
    alteracoes = validate(payload, parcial=True)
    itens = load()
    for indice, existente in enumerate(itens):
        if existente["id"] == item_id:
            itens[indice] = {**existente, **alteracoes}
            save(itens)
            return itens[indice]
    raise KeyError(item_id)


def delete(item_id: str) -> None:
    itens = load()
    restantes = [item for item in itens if item["id"] != item_id]
    if len(restantes) == len(itens):
        raise KeyError(item_id)
    save(restantes)
