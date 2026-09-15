import json

from gjurema.sources import cep as cep_source


class Resposta:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_resolve_consulta_apenas_ceps_novos(tmp_path, monkeypatch):
    cache_path = tmp_path / "ceps.json"
    cache_path.write_text(json.dumps({"05421030": "PINHEIROS"}))
    consultados: list[str] = []

    def fake_get(url: str, timeout: int) -> Resposta:
        cep = url.rsplit("/ws/", 1)[1].split("/")[0]
        consultados.append(cep)
        return Resposta({"bairro": "Água Branca", "localidade": "São Paulo"})

    monkeypatch.setattr(cep_source.requests, "get", fake_get)
    cache = cep_source.resolve(["05421030", "05036000", "05036000", ""], path=cache_path)

    assert consultados == ["05036000"]
    assert cache["05036000"] == "AGUA BRANCA"
    assert cache["05421030"] == "PINHEIROS"
    assert json.loads(cache_path.read_text()) == cache


def test_resolve_nao_grava_bairro_de_outro_municipio(tmp_path, monkeypatch):
    cache_path = tmp_path / "ceps.json"
    monkeypatch.setattr(
        cep_source.requests,
        "get",
        lambda url, timeout: Resposta({"bairro": "Centro", "localidade": "Guarulhos"}),
    )
    cache = cep_source.resolve(["07011000"], path=cache_path)
    assert cache == {"07011000": ""}


def test_resolve_tolera_cep_inexistente(tmp_path, monkeypatch):
    cache_path = tmp_path / "ceps.json"
    monkeypatch.setattr(cep_source.requests, "get", lambda url, timeout: Resposta({"erro": True}))
    assert cep_source.resolve(["00000000"], path=cache_path) == {"00000000": ""}


def test_resolve_sem_pendencia_nao_toca_a_rede(tmp_path, monkeypatch):
    cache_path = tmp_path / "ceps.json"
    cache_path.write_text(json.dumps({"05421030": "PINHEIROS"}))

    def explode(*args, **kwargs):  # pragma: no cover - não deve ser chamado
        raise AssertionError("ViaCEP não deveria ser consultado")

    monkeypatch.setattr(cep_source.requests, "get", explode)
    assert cep_source.resolve(["05421030"], path=cache_path) == {"05421030": "PINHEIROS"}
