"""Treinamento e avaliação do modelo de propensão ao pagamento."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clientes_inadimplentes.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_cobranca.joblib"
ROC_PLOT_PATH = BASE_DIR / "assets" / "curva_roc.png"

NUMERIC_FEATURES = [
    "valor_devido",
    "dias_atraso",
    "qtd_atrasos_12m",
    "renda_faixa",
    "tempo_relacionamento_meses",
]
CATEGORICAL_FEATURES = ["canal_preferido"]
TARGET = "pagou"


def criar_preprocessador() -> ColumnTransformer:
    """Cria transformador de colunas numéricas e categóricas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def treinar_e_avaliar() -> dict:
    """Treina Regressão Logística, compara com Gradient Boosting e salva o modelo final."""
    df = pd.read_csv(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # 1. Modelo principal: Regressão Logística
    pipe_lr = Pipeline(
        steps=[
            ("prep", criar_preprocessador()),
            ("model", LogisticRegression(random_state=SEED)),
        ]
    )
    pipe_lr.fit(X_train, y_train)
    prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
    pred_lr = (prob_lr >= 0.50).astype(int)

    # 2. Modelo de comparação: Gradient Boosting
    pipe_gb = Pipeline(
        steps=[
            ("prep", criar_preprocessador()),
            ("model", GradientBoostingClassifier(random_state=SEED)),
        ]
    )
    pipe_gb.fit(X_train, y_train)
    prob_gb = pipe_gb.predict_proba(X_test)[:, 1]
    pred_gb = (prob_gb >= 0.50).astype(int)

    # 3. Tabela comparativa
    comparacao = pd.DataFrame(
        [
            {
                "Modelo": "Regressão Logística",
                "AUC-ROC": roc_auc_score(y_test, prob_lr),
                "Precision (1)": precision_score(y_test, pred_lr),
                "Recall (1)": recall_score(y_test, pred_lr),
            },
            {
                "Modelo": "Gradient Boosting",
                "AUC-ROC": roc_auc_score(y_test, prob_gb),
                "Precision (1)": precision_score(y_test, pred_gb),
                "Recall (1)": recall_score(y_test, pred_gb),
            },
        ]
    )

    # 4. Métricas detalhadas e coeficientes da Regressão Logística
    auc_lr = roc_auc_score(y_test, prob_lr)
    auc_gb = roc_auc_score(y_test, prob_gb)
    matriz = confusion_matrix(y_test, pred_lr)
    relatorio = classification_report(y_test, pred_lr, digits=3)

    nomes_features = pipe_lr.named_steps["prep"].get_feature_names_out()
    pesos = pd.Series(
        pipe_lr.named_steps["model"].coef_[0], index=nomes_features
    ).sort_values(ascending=False)

    # 5. Persistência do modelo escolhido (Regressão Logística) e gráfico ROC
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_lr, MODEL_PATH)
    salvar_grafico_roc(y_test, prob_lr, auc_lr, prob_gb, auc_gb)

    print("=" * 60)
    print("COMPARAÇÃO DE MODELOS (BASE DE TESTE - 20%)")
    print("=" * 60)
    print(comparacao.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + "=" * 60)
    print("DETALHAMENTO: REGRESSÃO LOGÍSTICA (MODELO ESCOLHIDO)")
    print("=" * 60)
    print("Matriz de Confusão:")
    print(matriz)
    print("\nRelatório de Classificação:")
    print(relatorio)
    print("Coeficientes do Modelo:")
    for feat, peso in pesos.items():
        sinal = "+" if peso > 0 else ""
        print(f"  {feat:<34}: {sinal}{peso:.4f}")
    print("=" * 60)
    print(f"[OK] Modelo salvo em: {MODEL_PATH}")
    print(f"[OK] Gráfico ROC salvo em: {ROC_PLOT_PATH}")

    return {"auc_lr": auc_lr, "auc_gb": auc_gb, "pesos": pesos.to_dict()}


def salvar_grafico_roc(
    y_true: pd.Series,
    prob_lr: pd.Series,
    auc_lr: float,
    prob_gb: pd.Series,
    auc_gb: float,
) -> None:
    """Gera e salva o gráfico comparativo da Curva ROC."""
    ROC_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, prob_lr)
    fpr_gb, tpr_gb, _ = roc_curve(y_true, prob_gb)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_lr, tpr_lr, color="#cc092f", lw=2.5, label=f"Regressão Logística (AUC = {auc_lr:.3f})")
    plt.plot(fpr_gb, tpr_gb, color="#1f4e78", lw=1.8, linestyle="-.", label=f"Gradient Boosting (AUC = {auc_gb:.3f})")
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
