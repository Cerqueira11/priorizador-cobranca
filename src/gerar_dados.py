"""Geração de base sintética de clientes inadimplentes para simulação de cobrança."""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clientes_inadimplentes.csv"


def gerar_base_clientes(n_clientes: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """Gera base fictícia de clientes em atraso com variável alvo probabilística."""
    rng = np.random.default_rng(seed)

    id_cliente = np.arange(1, n_clientes + 1)

    valor_devido = np.round(rng.lognormal(mean=7.2, sigma=0.8, size=n_clientes), 2)
    valor_devido = np.clip(valor_devido, 100.0, 25000.0)

    dias_atraso = rng.integers(low=5, high=181, size=n_clientes)

    qtd_atrasos_12m = rng.poisson(lam=1.8, size=n_clientes)
    qtd_atrasos_12m = np.clip(qtd_atrasos_12m, 0, 6)

    tempo_relacionamento_meses = rng.integers(low=1, high=121, size=n_clientes)

    # 1 = até 2 SM, 2 = 2 a 5 SM, 3 = acima de 5 SM
    renda_faixa = rng.choice([1, 2, 3], size=n_clientes, p=[0.50, 0.35, 0.15])

    canal_preferido = rng.choice(
        ["WhatsApp", "App_Push", "Telefone", "Email"],
        size=n_clientes,
        p=[0.45, 0.25, 0.20, 0.10],
    )

    # Renda de referência por faixa (R$ 2.500, R$ 5.000, R$ 10.000)
    renda_ref = np.array([2500, 5000, 10000])[renda_faixa - 1]
    divida_relativa = np.clip(valor_devido / renda_ref, 0.0, 5.0)

    efeito_canal = (
        pd.Series(canal_preferido)
        .map({"WhatsApp": 0.15, "App_Push": 0.10, "Telefone": 0.0, "Email": -0.20})
        .to_numpy()
    )

    score_latente = (
        1.5
        - 0.020 * dias_atraso
        - 0.60 * (dias_atraso > 90)
        - 0.45 * qtd_atrasos_12m
        + 0.012 * tempo_relacionamento_meses
        + 0.35 * (renda_faixa - 1)
        - 0.50 * divida_relativa
        + efeito_canal
        + rng.normal(0, 1.1, size=n_clientes)
    )

    prob_real_pagamento = 1 / (1 + np.exp(-score_latente))
    pagou = rng.binomial(n=1, p=prob_real_pagamento)

    df = pd.DataFrame(
        {
            "id_cliente": id_cliente,
            "valor_devido": valor_devido,
            "dias_atraso": dias_atraso,
            "qtd_atrasos_12m": qtd_atrasos_12m,
            "renda_faixa": renda_faixa,
            "tempo_relacionamento_meses": tempo_relacionamento_meses,
            "canal_preferido": canal_preferido,
            "pagou": pagou,
        }
    )

    return df


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = gerar_base_clientes()
    df.to_csv(DATA_PATH, index=False)

    taxa_pagamento = df["pagou"].mean() * 100
    print(f"[OK] Base gerada em: {DATA_PATH}")
    print(f"Total de clientes: {len(df)} | Taxa média de pagamento: {taxa_pagamento:.1f}%")


if __name__ == "__main__":
    main()
