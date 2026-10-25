"""
Módulo 2: Treinamento e Avaliação do Modelo de Propensão ao Pagamento.

Por que Regressão Logística?
Em bancos (crédito e cobrança), modelos lineares como Regressão Logística são o
padrão de mercado para baseline e produção por três motivos:
1. Interpretabilidade (governança e exigências regulatórias do Bacen).
2. Probabilidades bem calibradas nativamente (essencial para calcular Valor Esperado).
3. Baixo custo computacional e menor risco de overfitting.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
     classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clientes_inadimplentes.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_cobranca.joblib"
ROC_PLOT_PATH = BASE_DIR / "assets" / "curva_roc.png"

# Variáveis preditoras (features) usadas pelo modelo
FEATURES = [
    "valor_devido",
    "dias_atraso",
    "qtd_atrasos_12m",
    "renda_faixa",
    "tempo_relacionamento_meses",
]
TARGET = "pagou"


def treinar_e_avaliar() -> dict:
    """Lê a base, treina o pipeline de Regressão Logística, avalia métricas e salva artefatos."""
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    # 1. Divisão Treino (80%) e Teste (20%) com estratificação (mantém a proporção de 32% de pagadores)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # 2. Pipeline: Padronização (evita Data Leakage) + Regressão Logística
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=SEED)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # 3. Predições no conjunto de teste (dados que o modelo nunca viu)
    prob_pagamento = pipeline.predict_proba(X_test)[:, 1]
    pred_classe = (prob_pagamento >= 0.50).astype(int)

    # 4. Métricas de avaliação
    auc = roc_auc_score(y_test, prob_pagamento)
    matriz = confusion_matrix(y_test, pred_classe)
    relatorio = classification_report(y_test, pred_classe, digits=3)

    # 5. Interpretabilidade: pesos aprendidos pelo modelo para cada variável
    modelo_lr = pipeline.named_steps["logreg"]
    pesos = pd.Series(modelo_lr.coef_[0], index=FEATURES).sort_values(ascending=False)

    # 6. Salvar o modelo treinado e o gráfico da Curva ROC
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    salvar_grafico_roc(y_test, prob_pagamento, auc)

    print("=" * 55)
    print("RESULTADOS DO MODELO (BASE DE TESTE - 20%)")
    print("=" * 55)
    print(f"AUC-ROC: {auc:.4f}")
    print("\nMatriz de Confusão:")
    print(matriz)
    print("\nRelatório de Classificação:")
    print(relatorio)
    print("Pesos das Variáveis (Coeficientes Padronizados):")
    for feat, peso in pesos.items():
        sinal = "+" if peso > 0 else ""
        print(f"  {feat:<28}: {sinal}{peso:.4f}")
    print("=" * 55)
    print(f"[OK] Modelo salvo em: {MODEL_PATH}")
    print(f"[OK] Gráfico ROC salvo em: {ROC_PLOT_PATH}")

    return {"auc": auc, "pesos": pesos.to_dict()}


def salvar_grafico_roc(y_true: pd.Series, y_prob: pd.Series, auc: float) -> None:
    """Gera e salva o gráfico da Curva ROC para documentação no README."""
    ROC_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#cc092f", lw=2.5, label=f"Regressão Logística (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Aleatório (AUC = 0.500)")
    plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    plt.ylabel("Taxa de Verdadeiros Positivos (Recall)")
    plt.title("Curva ROC - Propensão ao Pagamento em Cobrança")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=150)
    plt.close()


if __name__ == "__main__":
    treinar_e_avaliar()
