"""Cálculo de score de prioridade, régua de acionamento e gravação em SQLite."""

from pathlib import Path
import sqlite3
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.treinar_modelo import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        adicionar_features_negocio,
    )
except ModuleNotFoundError:
    from treinar_modelo import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        adicionar_features_negocio,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clientes_inadimplentes.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_cobranca.joblib"
DB_PATH = BASE_DIR / "data" / "cobranca.db"
OUTPUT_CSV_PATH = BASE_DIR / "data" / "clientes_priorizados.csv"
SQL_PATH = BASE_DIR / "src" / "indicadores.sql"
ATRASO_PLOT_PATH = BASE_DIR / "assets" / "recuperacao_por_atraso.png"


def definir_acao_por_probabilidade(prob: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Classifica clientes em faixas de propensão e define a ação recomendada."""
    condicoes = [
        prob >= 0.60,
        (prob >= 0.30) & (prob < 0.60),
        prob < 0.30,
    ]
    faixas = ["Alta", "Media", "Baixa"]
    acoes = ["Mensagem Automatica", "Ligacao", "Proposta de Acordo"]

    faixa_propensao = pd.Series(
        np.select(condicoes, faixas, default="Baixa"), index=prob.index
    )
    acao_sugerida = pd.Series(
        np.select(condicoes, acoes, default="Proposta de Acordo"), index=prob.index
    )
    return faixa_propensao, acao_sugerida


def priorizar_carteira(df: pd.DataFrame, modelo) -> pd.DataFrame:
    """Aplica o modelo na carteira, calcula o valor esperado recuperável e ordena por prioridade."""
    df_feat = adicionar_features_negocio(df)
    X = df_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prob = pd.Series(modelo.predict_proba(X)[:, 1], index=df.index)
    faixa_propensao, acao_sugerida = definir_acao_por_probabilidade(prob)

    resultado = df.copy()
    resultado["prob_pagamento"] = np.round(prob, 4)
    # Score de prioridade = Valor Esperado Recuperável (R$) = P(pagar) * valor_devido
    resultado["score_prioridade"] = np.round(
        resultado["prob_pagamento"] * resultado["valor_devido"], 2
    )
    resultado["faixa_propensao"] = faixa_propensao
    resultado["acao_sugerida"] = acao_sugerida

    resultado = resultado.sort_values(by="score_prioridade", ascending=False).reset_index(
        drop=True
    )
    return resultado


def salvar_no_sqlite_e_consultar(df_priorizado: pd.DataFrame) -> None:
    """Grava a carteira priorizada no banco SQLite e executa as consultas de indicadores.sql."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        df_priorizado.to_sql("clientes_priorizados", conn, if_exists="replace", index=False)

        sql_content = SQL_PATH.read_text(encoding="utf-8")
        blocos = [b.strip() for b in sql_content.split(";") if b.strip()]

        print("=" * 72)
        print("INDICADORES DE NEGÓCIO (SQLITE)")
        print("=" * 72)
        for bloco in blocos:
            linhas = bloco.splitlines()
            titulo = linhas[0].replace("--", "").strip() if linhas[0].startswith("--") else "Consulta SQL"
            df_query = pd.read_sql_query(bloco, conn)
            print(f"\n[{titulo}]")
            print(df_query.to_string(index=False))
        print("=" * 72)


def salvar_grafico_atraso(df_priorizado: pd.DataFrame) -> None:
    """Gera gráfico da taxa de pagamento por faixa de atraso para o README."""
    ATRASO_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bins = [0, 30, 60, 90, 200]
    labels = ["Até 30d", "31 a 60d", "61 a 90d", "Acima de 90d"]
    faixas = pd.cut(df_priorizado["dias_atraso"], bins=bins, labels=labels)

    resumo = (
        df_priorizado.groupby(faixas, observed=False)
        .agg(
            taxa_real=("pagou", lambda s: s.mean() * 100),
            prob_prevista=("prob_pagamento", lambda s: s.mean() * 100),
        )
        .reset_index()
    )

    x = np.arange(len(resumo))
    largura = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - largura / 2, resumo["taxa_real"], largura, label="Taxa Real (%)", color="#cc092f")
    plt.bar(
        x + largura / 2,
        resumo["prob_prevista"],
        largura,
        label="Probabilidade Média Prevista (%)",
        color="#1f4e78",
    )
    plt.xticks(x, resumo["dias_atraso"])
    plt.ylabel("Taxa de Pagamento (%)")
    plt.title("Recuperação por Faixa de Atraso (Real vs. Previsto pelo Modelo)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ATRASO_PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    modelo = joblib.load(MODEL_PATH)

    df_priorizado = priorizar_carteira(df, modelo)
    df_priorizado.to_csv(OUTPUT_CSV_PATH, index=False)

    salvar_no_sqlite_e_consultar(df_priorizado)
    salvar_grafico_atraso(df_priorizado)

    print(f"[OK] Lista priorizada salva em: {OUTPUT_CSV_PATH}")
    print(f"[OK] Banco SQLite atualizado em: {DB_PATH}")
    print(f"[OK] Gráfico de recuperação salvo em: {ATRASO_PLOT_PATH}")


if __name__ == "__main__":
    main()
