# GJurema — MVP do Modelo de Precificação e Análise de Investimento Imobiliário

Implementação da **Fase MVP** do roadmap (0–3 meses): modelo de precificação
básico + dashboard com cap rate e yield por cidade, alimentado por
**FipeZAP + IBGE + Banco Central**.

As quatro camadas do documento estão mapeadas diretamente no código:

| Camada | Descrição | Código |
| --- | --- | --- |
| 1 — Inputs | FipeZAP (preço de venda e locação por m², por cidade e tipologia), BCB/SGS (SELIC, IPCA, IGP-M, INCC, taxa de crédito imobiliário), IBGE (população e PIB municipal) | `src/gjurema/sources/` |
| 2 — Feature engineering | Momentum de preço (3/6/12m), yield implícito, volatilidade, preço relativo à mediana, índice de mercado por cidade | `src/gjurema/features.py` |
| 3 — Modelos | Preço justo (XGBoost com validação temporal e intervalo de incerteza), valorização (Holt com tendência amortecida), rentabilidade (cap rate, yield, TIR, VPL, payback) | `src/gjurema/models/` |
| 4 — Outputs | Score de oportunidade, ranking de rentabilidade, simulador de ROI com cenários | `src/gjurema/dashboard/app.py` |

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Uso

```bash
# 1. Ingestão + features + índices (baixa ~5 MB do FipeZAP e consulta BCB/IBGE)
python -m gjurema.pipeline build

# 2. Treino do modelo de preço justo (grava artifacts/)
python -m gjurema.pipeline train

# ou tudo de uma vez
python -m gjurema.pipeline all

# 3. Dashboard
streamlit run src/gjurema/dashboard/app.py
```

Artefatos gerados:

- `data/raw/fipezap-serieshistoricas.xlsx` — planilha original em cache
- `data/processed/features.parquet` — painel cidade × tipologia × mês (base de treino)
- `data/processed/indice_mercado.parquet` — índice de aquecimento por cidade
- `data/processed/valorizacao_projetada.parquet` — projeção de 12 meses
- `artifacts/fair_price.json` + `fair_price_meta.json` — modelo treinado
- `artifacts/metrics.json` — métricas de validação e importância das variáveis

## Dashboard

Cinco telas:

1. **Preço justo e oportunidade** — informe cidade, tipologia, área e preço pedido;
   devolve preço justo, faixa de incerteza, desvio percentual e score de oportunidade (0–100).
2. **Rentabilidade por cidade** — ranking de cap rate, yield bruto/líquido e payback
   para uma área de referência.
3. **Séries históricas** — evolução de preço de venda, locação e yield por cidade,
   com projeção de valorização em 12 meses.
4. **Simulador de ROI** — TIR, VPL contra a SELIC, retorno total em 5/10/15 anos,
   com e sem alavancagem, e cenários conservador/base/otimista.
5. **Qualidade do modelo** — métricas de validação temporal e importância das variáveis.

## Modelo de preço justo

- **Alvo:** log do preço de venda por m² (estabiliza variância entre cidades).
- **Features:** cidade, UF, tipologia, sazonalidade, nível de preço 12 meses antes,
  macro do BCB e socioeconômicos do IBGE.
- **Anti-vazamento:** features derivadas do preço corrente (lag de 1 mês, variações
  recentes, yield, preço relativo) ficam fora do treino — ver `price.LEAKY_FEATURES`.
- **Validação:** temporal, com os últimos 12 meses como holdout.
- **Incerteza:** quantis empíricos dos resíduos da validação (p10/p50/p90); o p50
  também corrige o viés da previsão pontual.

Métricas da última execução (validação a partir de 09/2025): R² em log ≈ 0,985 e
erro percentual médio ≈ 2,7%. O backtest da projeção de 12 meses fica em ≈ 4,8% de MAPE.

## Limitações conhecidas (e como a Fase 2 as resolve)

- O FipeZAP agrega por **cidade e tipologia**, não por imóvel: o preço justo é um
  preço de referência por m², sem atributos como andar, vaga, padrão de acabamento
  ou distância a POIs. O score de localização descrito no documento exige o scraping
  de anúncios (Fase 2) e dados de cartório (Fase 3).
- Os preços do FipeZAP são de **anúncio**, não de fechamento; o desconto médio de
  negociação só entra com dados de cartório.
- A projeção de valorização usa tendência amortecida; Prophet/LSTM com variáveis
  exógenas entram quando houver histórico por bairro.
- O IBGE só é consultado para população e PIB municipal — renda e perfil
  socioeconômico por setor censitário ficam para a camada de demanda.

## Testes e lint

```bash
python -m pytest -q
python -m ruff check src tests
```
