"""Treinamento e avaliação do modelo de propensão ao pagamento."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "pagou"


def criar_preprocessador() -> ColumnTransformer:
    """Padroniza variáveis numéricas e aplica One-Hot Encoding nas categóricas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def avaliar_modelo(nome: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de discriminação e classificação na base de teste."""
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)
    return {
        "modelo": nome,
        "auc_roc": roc_auc_score(y_test, prob),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "prob": prob,
        "pred": pred,
    }


def treinar_e_avaliar() -> dict:
    """Treina Regressão Logística (baseline) e Gradient Boosting, compara métricas e salva o modelo final."""
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # 1. Modelo principal: Regressão Logística
    pipe_lr = Pipeline(
        steps=[
            ("prep", criar_preprocessador()),
            ("clf", LogisticRegression(random_state=SEED)),
        ]
    )
    pipe_lr.fit(X_train, y_train)
    res_lr = avaliar_modelo("Regressão Logística", pipe_lr, X_test, y_test)

    # 2. Modelo de comparação: Gradient Boosting
    pipe_gb = Pipeline(
        steps=[
            ("prep", criar_preprocessador()),
            ("clf", GradientBoostingClassifier(random_state=SEED)),
        ]
    )
    pipe_gb.fit(X_train, y_train)
    res_gb = avaliar_modelo("Gradient Boosting", pipe_gb, X_test, y_test)

    # Tabela comparativa
    comparacao = pd.DataFrame([res_lr, res_gb])[["modelo", "auc_roc", "precision", "recall"]]

    # Matriz de confusão e coeficientes da Regressão Logística
    matriz = confusion_matrix(y_test, res_lr["pred"])
    nomes_features = pipe_lr.named_steps["prep"].get_feature_names_out()
    pesos = pd.Series(
        pipe_lr.named_steps["clf"].coef_[0], index=nomes_features
    ).sort_values(ascending=False)

    # Salva a Regressão Logística como modelo de produção (melhor balanço entre AUC e interpretabilidade)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_lr, MODEL_PATH)

    salvar_grafico_roc(y_test, res_lr, res_gb)

    print("=" * 60)
    print("COMPARAÇÃO DE MODELOS (BASE DE TESTE - 20%)")
    print("=" * 60)
    print(comparacao.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMatriz de Confusão (Regressão Logística):")
    print(matriz)
    print("\nCoeficientes da Regressão Logística:")
    for feat, peso in pesos.items():
        sinal = "+" if peso > 0 else ""
        print(f"  {feat:<35}: {sinal}{peso:.4f}")
    print("=" * 60)
    print(f"[OK] Modelo salvo em: {MODEL_PATH}")
    print(f"[OK] Gráfico ROC salvo em: {ROC_PLOT_PATH}")

    return {"auc_lr": res_lr["auc_roc"], "auc_gb": res_gb["auc_roc"], "pesos": pesos.to_dict()}


def salvar_grafico_roc(y_true: pd.Series, res_lr: dict, res_gb: dict) -> None:
    """Gera e salva o gráfico comparativo da Curva ROC."""
    ROC_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, res_lr["prob"])
    fpr_gb, tpr_gb, _ = roc_curve(y_true, res_gb["prob"])

    plt.figure(figsize=(7, 5))
    plt.plot(
        fpr_lr,
        tpr_lr,
        color="#cc092f",
        lw=2.5,
        label=f"Regressão Logística (AUC = {res_lr['auc_roc']:.3f})",
    )
    plt.plot(
        fpr_gb,
        tpr_gb,
        color="#1f4e78",
        lw=2.0,
        linestyle="-.",
        label=f"Gradient Boosting (AUC = {res_gb['auc_roc']:.3f})",
    )
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
