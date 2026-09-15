import pandas as pd
import pytest

from gjurema.sources import itbi_sp


@pytest.fixture
def guias() -> pd.DataFrame:
    base = {
        "sql": "0010100001",
        "logradouro": "CAPITÃO PRUDENTE",
        "numero": 209,
        "complemento": "AP 702",
        "cep": 5421030,
        "natureza": "Compra e venda",
        "valor_transacao": 1_000_000.0,
        "data": pd.Timestamp("2025-03-10"),
        "valor_venal": 800_000.0,
        "proporcao_pct": 100.0,
        "valor_venal_proporcional": 800_000.0,
        "base_calculo": "Valor de transação",
        "tipo_financiamento": "SFH",
        "valor_financiado": 400_000.0,
        "area_construida": 100.0,
        "uso": "APARTAMENTO EM CONDOMÍNIO (EXIGE FRAÇÃO IDEAL)",
        "padrao": 4,
        "competencia": pd.Timestamp("2025-03-01"),
    }
    variantes = [
        base,
        # dação em pagamento: preço pactuado fora de mercado
        {**base, "sql": "2", "natureza": "Dação em pagamento"},
        # transmissão parcial
        {**base, "sql": "3", "proporcao_pct": 50.0},
        # uso fora dos segmentos cobertos
        {**base, "sql": "4", "uso": "TERRENO"},
        # área irrisória
        {**base, "sql": "5", "area_construida": 4.0},
        # R$/m² fora da faixa plausível
        {**base, "sql": "6", "valor_transacao": 90_000_000.0},
        # sem financiamento e com deságio sobre o venal
        {
            **base,
            "sql": "7",
            "valor_financiado": 0.0,
            "valor_transacao": 700_000.0,
            "numero": 211,
        },
        # guia retificada republicada em outra planilha
        base,
    ]
    return pd.DataFrame(variantes)


def test_clean_mantem_apenas_compra_e_venda_de_imovel_inteiro(guias):
    limpo = itbi_sp.clean(guias)
    assert sorted(limpo["sql"]) == ["0010100001", "7"]
    assert limpo["segmento"].unique().tolist() == ["Apartamento"]


def test_clean_deriva_preco_predio_e_agio(guias):
    limpo = itbi_sp.clean(guias).set_index("sql")

    principal = limpo.loc["0010100001"]
    assert principal["preco_m2"] == pytest.approx(10_000.0)
    assert principal["predio_id"] == "CAPITAO PRUDENTE|209|05421030"
    assert principal["financiado"]
    assert principal["agio_venal"] == pytest.approx(0.25)
    assert principal["ano_mes"] == pd.Timestamp("2025-03-01")

    sem_financiamento = limpo.loc["7"]
    assert not sem_financiamento["financiado"]
    assert sem_financiamento["agio_venal"] == pytest.approx(-0.125)
    assert sem_financiamento["predio_id"] != principal["predio_id"]


def test_clean_ignora_coluna_bairro_da_guia(guias):
    # O campo é livre e recebe "TORRE 1"/"BLOCO B"; o produto usa CEP.
    limpo = itbi_sp.clean(guias.assign(bairro="TORRE 1"))
    assert "bairro" not in limpo.columns


def test_parse_workbook_le_somente_abas_mensais(tmp_path, guias):
    caminho = tmp_path / "itbi.xlsx"
    fonte = {alvo: origem for origem, alvo in itbi_sp.COLUMNS.items()}
    planilha = guias.drop(columns=["competencia"]).rename(columns=fonte)
    with pd.ExcelWriter(caminho) as writer:
        planilha.to_excel(writer, sheet_name="MAR-2025", index=False)
        planilha.head(1).to_excel(writer, sheet_name="ABR-2025", index=False)
        planilha.head(1).to_excel(writer, sheet_name="LEGENDA", index=False)

    bruto = itbi_sp.parse_workbook(caminho)
    assert len(bruto) == len(guias) + 1
    assert set(bruto["competencia"]) == {pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")}
    assert "sql" in bruto.columns


def test_parse_workbook_tolera_colunas_ausentes(tmp_path, guias):
    caminho = tmp_path / "itbi_antigo.xlsx"
    fonte = {alvo: origem for origem, alvo in itbi_sp.COLUMNS.items()}
    planilha = guias.drop(columns=["competencia", "valor_financiado"]).rename(columns=fonte)
    with pd.ExcelWriter(caminho) as writer:
        planilha.to_excel(writer, sheet_name="MAR-2025", index=False)

    bruto = itbi_sp.parse_workbook(caminho)
    # coluna ausente vira nula para o filtro não confundir "não informado" com zero
    assert bruto["valor_financiado"].isna().all()

    limpo = itbi_sp.clean(bruto)
    assert len(limpo) == 2
    assert limpo["financiado"].isna().all()


def test_parse_workbook_rejeita_planilha_sem_coluna_obrigatoria(tmp_path, guias):
    caminho = tmp_path / "itbi_incompleto.xlsx"
    fonte = {alvo: origem for origem, alvo in itbi_sp.COLUMNS.items()}
    planilha = guias.drop(columns=["competencia", "area_construida"]).rename(columns=fonte)
    with pd.ExcelWriter(caminho) as writer:
        planilha.to_excel(writer, sheet_name="MAR-2025", index=False)

    with pytest.raises(ValueError, match="area_construida"):
        itbi_sp.parse_workbook(caminho)


def test_parse_workbook_sem_aba_mensal(tmp_path):
    caminho = tmp_path / "vazio.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(caminho, sheet_name="LEGENDA", index=False)
    with pytest.raises(ValueError):
        itbi_sp.parse_workbook(caminho)


def test_workbook_urls_filtra_planilhas_de_itbi(monkeypatch):
    html = """
      <a href="/upload/fazenda/arquivos/itbi/GUIAS-DE-ITBI-PAGAS-2024.xlsx">2024</a>
      <a href="/upload/fazenda/arquivos/itbi/GUIAS-DE-ITBI-PAGAS-2024.ods">2024 ODS</a>
      <a href="/upload/fazenda/arquivos/itbi/GUIAS DE ITBI PAGAS (2019).xlsx">2019</a>
      <a href="/upload/fazenda/arquivos/iptu/valores-venais.xlsx">IPTU</a>
      <a href="/web/fazenda/w/acesso_a_informacao/31501">página</a>
    """

    class Resposta:
        text = html

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(itbi_sp.requests, "get", lambda *a, **k: Resposta())
    urls = itbi_sp.workbook_urls("https://prefeitura.sp.gov.br/pagina")

    assert len(urls) == 2
    assert urls[0].endswith("GUIAS-DE-ITBI-PAGAS-2024.xlsx")
    assert itbi_sp._file_name(urls[1]) == "guias-de-itbi-pagas-2019-xlsx.xlsx"
