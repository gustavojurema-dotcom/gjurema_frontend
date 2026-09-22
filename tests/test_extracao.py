import pytest

from gjurema.api import extracao

FICHA = """
Empreendimento Nova Higienópolis
Unidade nº 701 — Rua das Palmeiras, 410 — Bairro: SANTA CECILIA
CEP: 01226-010
Construtora: Vitacon Participações
Área privativa: 33,01 m2
Área comum: 23,74 m2
Área total: 56,75 m2
Matrícula nº 130.099
Valor da compra: R$ 318.920,00
Data da compra: 13 de dezembro de 2019
Corretagem: R$ 21.080,00
Aluguel: R$ 2.400,00
Condomínio: R$ 690,00
IPTU: R$ 1.850,00
1 vaga, 1 dormitório
"""


@pytest.fixture(scope="module")
def campos() -> dict:
    return extracao.parse_fields(FICHA)


def test_extrai_valores_monetarios_em_formato_brasileiro(campos):
    assert campos["preco"] == 318_920.0
    assert campos["aluguelMensal"] == 2_400.0
    assert campos["condominioMensal"] == 690.0
    assert campos["iptuAnual"] == 1_850.0
    assert campos["corretagem"] == 21_080.0


def test_extrai_endereco_areas_e_matricula(campos):
    assert campos["endereco"] == "Rua das Palmeiras, 410"
    assert campos["cep"] == "01226-010"
    assert campos["bairro"] == "SANTA CECILIA"
    assert (campos["priv"], campos["comum"], campos["total"]) == (33.01, 23.74, 56.75)
    assert campos["matricula"] == "130.099"
    assert campos["unidade"] == "Unidade 701"


def test_extrai_data_por_extenso(campos):
    assert campos["dataCompra"] == "2019-12-13"


def test_documento_sem_dados_nao_inventa_campos():
    assert extracao.parse_fields("apenas um texto qualquer sem números") == {}


def test_formato_sem_leitor_e_reportado_por_arquivo():
    resultado = extracao.parse_documents([("planta.dwg", b"binario")])
    assert resultado["campos"] == {}
    assert "não suportado" in resultado["arquivos"][0]["erro"]


def test_documentos_multiplos_se_completam():
    resultado = extracao.parse_documents(
        [
            ("contrato.txt", b"Valor da compra: R$ 500.000,00"),
            ("locacao.txt", b"Aluguel: R$ 3.100,00"),
        ]
    )
    assert resultado["campos"]["preco"] == 500_000.0
    assert resultado["campos"]["aluguelMensal"] == 3_100.0
    assert [a["arquivo"] for a in resultado["arquivos"]] == ["contrato.txt", "locacao.txt"]
