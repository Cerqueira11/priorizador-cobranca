"""Treinamento e avaliação do modelo de propensão ao pagamento."""

from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clientes_inadimplentes.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_cobranca.joblib"
ROC_PLOT_PATH = BASE_DIR / "assets" / "curva_roc.png"

RAW_NUMERIC_FEATURES = [
    "valor_devido",
    "dias_atraso",
    "qtd_atrasos_12m",
    "renda_faixa",
    "tempo_relacionamento_meses",
]
ENGINEERED_FEATURES = [
    "divida_relativa",
    "atraso_90",
]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = ["canal_preferido"]
TARGET = "pagou"


def adicionar_features_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variáveis derivadas de negócio (comprometimento de renda e marco de 90 dias)."""
    df = df.copy()
    renda_ref = df["renda_faixa"].map({1: 2500, 2: 5000, 3: 10000})
    df["divida_relativa"] = (df["valor_devido"] / renda_ref).clip(0.0, 5.0)
    df["atraso_90"] = (df["dias_atraso"] > 90).astype(int)
    return df


def criar_preprocessador(num_cols: list[str]) -> ColumnTransformer:
    """Cria transformador de colunas numéricas e categóricas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def treinar_e_avaliar() -> dict:
    """Compara modelos via validação cruzada, avalia impacto de feature engineering e salva modelo final."""
    df = pd.read_csv(DATA_PATH)
    df = adicionar_features_negocio(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # 1. Pipelines: Baseline (sem features de negócio), Gradient Boosting e LogReg com features de negócio
    pipe_lr_raw = Pipeline(
        steps=[
            ("prep", criar_preprocessador(RAW_NUMERIC_FEATURES)),
            ("model", LogisticRegression(random_state=SEED)),
        ]
    )
    pipe_gb_raw = Pipeline(
        steps=[
            ("prep", criar_preprocessador(RAW_NUMERIC_FEATURES)),
            ("model", GradientBoostingClassifier(random_state=SEED)),
        ]
    )
    pipe_lr_fe = Pipeline(
        steps=[
            ("prep", criar_preprocessador(NUMERIC_FEATURES)),
            ("model", LogisticRegression(random_state=SEED)),
        ]
    )

    # 2. Validação cruzada (5 folds) no conjunto de treino para comparação justa
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_lr_raw = cross_val_score(pipe_lr_raw, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_gb_raw = cross_val_score(pipe_gb_raw, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_lr_fe = cross_val_score(pipe_lr_fe, X_train, y_train, cv=cv, scoring="roc_auc")

    # 3. Ajuste final e avaliação no conjunto de teste (holdout 20%)
    pipe_lr_raw.fit(X_train, y_train)
    pipe_gb_raw.fit(X_train, y_train)
    pipe_lr_fe.fit(X_train, y_train)

    prob_lr_raw = pipe_lr_raw.predict_proba(X_test)[:, 1]
    prob_gb_raw = pipe_gb_raw.predict_proba(X_test)[:, 1]
    prob_lr_fe = pipe_lr_fe.predict_proba(X_test)[:, 1]
    pred_lr_fe = (prob_lr_fe >= 0.50).astype(int)

    comparacao = pd.DataFrame(
        [
            {
                "Abordagem": "Regressão Logística (features brutas)",
                "CV AUC (média ± dp)": f"{cv_lr_raw.mean():.4f} ± {cv_lr_raw.std():.4f}",
                "Teste AUC": roc_auc_score(y_test, prob_lr_raw),
            },
            {
                "Abordagem": "Gradient Boosting (features brutas)",
                "CV AUC (média ± dp)": f"{cv_gb_raw.mean():.4f} ± {cv_gb_raw.std():.4f}",
                "Teste AUC": roc_auc_score(y_test, prob_gb_raw),
            },
            {
                "Abordagem": "Regressão Logística (+ features negócio)",
                "CV AUC (média ± dp)": f"{cv_lr_fe.mean():.4f} ± {cv_lr_fe.std():.4f}",
                "Teste AUC": roc_auc_score(y_test, prob_lr_fe),
            },
        ]
    )

    auc_final = roc_auc_score(y_test, prob_lr_fe)
    auc_gb = roc_auc_score(y_test, prob_gb_raw)
    matriz = confusion_matrix(y_test, pred_lr_fe)
    relatorio = classification_report(y_test, pred_lr_fe, digits=3)

    nomes_features = pipe_lr_fe.named_steps["prep"].get_feature_names_out()
    pesos = pd.Series(
        pipe_lr_fe.named_steps["model"].coef_[0], index=nomes_features
    ).sort_values(ascending=False)

    # 4. Salvar modelo final (Regressão Logística + Features de Negócio) e gráfico ROC
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_lr_fe, MODEL_PATH)
    salvar_grafico_roc(y_test, prob_lr_fe, auc_final, prob_gb_raw, auc_gb)

    print("=" * 72)
    print("COMPARAÇÃO DE MODELOS (VALIDAÇÃO CRUZADA 5-FOLD E TESTE)")
    print("=" * 72)
    print(comparacao.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + "=" * 72)
    print("MODELO ESCOLHIDO: REGRESSÃO LOGÍSTICA (+ FEATURES DE NEGÓCIO)")
    print("=" * 72)
    print("Matriz de Confusão (limiar de referência = 0.50):")
    print(matriz)
    print("\nRelatório de Classificação:")
    print(relatorio)
    print("Coeficientes Padronizados:")
    for feat, peso in pesos.items():
        sinal = "+" if peso > 0 else ""
        print(f"  {feat:<34}: {sinal}{peso:.4f}")
    print("=" * 72)
    print(f"[OK] Modelo salvo em: {MODEL_PATH}")
    print(f"[OK] Gráfico ROC salvo em: {ROC_PLOT_PATH}")

    return {
        "auc_lr_raw": float(roc_auc_score(y_test, prob_lr_raw)),
        "auc_gb_raw": float(auc_gb),
        "auc_final": float(auc_final),
        "precision": float(precision_score(y_test, pred_lr_fe)),
        "recall": float(recall_score(y_test, pred_lr_fe)),
        "pesos": pesos.to_dict(),
    }


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
    plt.plot(
        fpr_lr,
        tpr_lr,
        color="#cc092f",
        lw=2.5,
        label=f"Regressão Logística + Features Negócio (AUC = {auc_lr:.3f})",
    )
    plt.plot(
        fpr_gb,
        tpr_gb,
        color="#1f4e78",
        lw=1.8,
        linestyle="-.",
        label=f"Gradient Boosting Padrão (AUC = {auc_gb:.3f})",
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
