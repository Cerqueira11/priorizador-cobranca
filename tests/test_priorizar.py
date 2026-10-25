"""Testes unitários para geração de dados, engenharia de atributos e priorização."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.gerar_dados import gerar_base_clientes
from src.priorizar import definir_acao_por_probabilidade, priorizar_carteira
from src.treinar_modelo import adicionar_features_negocio


class ModeloMock:
    """Modelo simulado para testar a lógica de priorização de forma isolada."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prob_pagar = np.linspace(0.10, 0.90, len(X))
        return np.column_stack([1 - prob_pagar, prob_pagar])


def test_gerar_base_sem_nulos_e_alvo_binario() -> None:
    df = gerar_base_clientes(n_clientes=200, seed=42)
    assert len(df) == 200
    assert df.isna().sum().sum() == 0
    assert set(df["pagou"].unique()).issubset({0, 1})


def test_adicionar_features_negocio() -> None:
    df = pd.DataFrame(
        {
            "valor_devido": [1000.0, 20000.0],
            "dias_atraso": [45, 120],
            "renda_faixa": [1, 1],  # renda_ref = 2500
        }
    )
    df_feat = adicionar_features_negocio(df)

    assert df_feat.loc[0, "divida_relativa"] == 0.4
    assert df_feat.loc[1, "divida_relativa"] == 5.0  # limitado pelo clip em 5.0
    assert df_feat.loc[0, "atraso_90"] == 0
    assert df_feat.loc[1, "atraso_90"] == 1


def test_definir_acao_por_probabilidade() -> None:
    probs = pd.Series([0.75, 0.45, 0.15])
    faixas, acoes = definir_acao_por_probabilidade(probs)

    assert list(faixas) == ["Alta", "Media", "Baixa"]
    assert list(acoes) == ["Mensagem Automatica", "Ligacao", "Proposta de Acordo"]


def test_priorizar_carteira_intervalos_e_ordenacao() -> None:
    df = gerar_base_clientes(n_clientes=50, seed=42)
    resultado = priorizar_carteira(df, ModeloMock())

    assert resultado.isna().sum().sum() == 0
    assert resultado["prob_pagamento"].between(0.0, 1.0).all()
    assert (resultado["score_prioridade"] >= 0.0).all()
    assert resultado["score_prioridade"].is_monotonic_decreasing
