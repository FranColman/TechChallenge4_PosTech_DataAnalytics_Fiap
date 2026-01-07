"""
Tech Challenge Fase 4 — EDA + Pipeline ML (Obesity)

O que este script faz:
1) Carrega o dataset
2) EDA (gráficos essenciais)
3) Monta pipeline (scaler + onehot + modelo)
4) Compara modelos e seleciona o melhor
5) Avalia no conjunto de teste
6) Salva pipeline treinada em /models/model_pipeline.joblib
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import joblib


# =========================
# Configs
# =========================
DATA_PATH = "Obesity.csv"  # ajuste se necessário
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def save_fig(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"[OK] Gráfico salvo em: {path}")


def detect_target_column(df: pd.DataFrame) -> str:
    # O PDF menciona Obesity_level como alvo, mas alguns datasets usam Obesity.
    # Este método tenta achar automaticamente.
    candidates = ["Obesity_level", "Obesity"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Não encontrei coluna alvo. Procurei por: {candidates}. Colunas disponíveis: {list(df.columns)}")


def basic_checks(df: pd.DataFrame, target_col: str) -> None:
    print("\n======================")
    print("VISÃO GERAL DO DATASET")
    print("======================")
    print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
    print("\nTipos de dados:")
    print(df.dtypes)

    missing = df.isna().sum().sort_values(ascending=False)
    print("\nValores ausentes (top 10):")
    print(missing.head(10))

    print("\nAmostra:")
    print(df.head())

    print("\nDistribuição do alvo:")
    print(df[target_col].value_counts())


def eda_plots(df: pd.DataFrame, target_col: str) -> None:
    """
    EDA com gráficos úteis para:
    - mostrar distribuição das classes
    - evidenciar relação peso/altura com obesidade
    - avaliar variáveis comportamentais
    - mostrar correlação numérica
    """
    # 1) Distribuição do alvo
    plt.figure(figsize=(10, 5))
    df[target_col].value_counts().plot(kind="bar")
    plt.title("Distribuição da variável alvo (Nível de Obesidade)")
    plt.xlabel("Classe")
    plt.ylabel("Contagem")
    plt.xticks(rotation=45, ha="right")
    save_fig("01_distribuicao_alvo.png")
    plt.show()

    # 2) Scatter: Height x Weight por classe (amostra para ficar leve se necessário)
    sample_df = df.copy()
    if len(sample_df) > 2000:
        sample_df = sample_df.sample(2000, random_state=RANDOM_STATE)

    plt.figure(figsize=(8, 6))
    classes = sample_df[target_col].unique()
    for cls in classes:
        sub = sample_df[sample_df[target_col] == cls]
        plt.scatter(sub["Height"], sub["Weight"], alpha=0.6, label=str(cls))
    plt.title("Altura x Peso por Nível de Obesidade")
    plt.xlabel("Height (m)")
    plt.ylabel("Weight (kg)")
    plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    save_fig("02_scatter_altura_peso.png")
    plt.show()

    # 3) Boxplot simples sem seaborn: FAF por classe (atividade física)
    if "FAF" in df.columns:
        plt.figure(figsize=(10, 5))
        df.boxplot(column="FAF", by=target_col, grid=False)
        plt.title("Atividade Física (FAF) por Nível de Obesidade")
        plt.suptitle("")
        plt.xlabel("Classe")
        plt.ylabel("FAF")
        plt.xticks(rotation=45, ha="right")
        save_fig("03_boxplot_faf.png")
        plt.show()

    # 4) Boxplot: FCVC (vegetais)
    if "FCVC" in df.columns:
        plt.figure(figsize=(10, 5))
        df.boxplot(column="FCVC", by=target_col, grid=False)
        plt.title("Consumo de Vegetais (FCVC) por Nível de Obesidade")
        plt.suptitle("")
        plt.xlabel("Classe")
        plt.ylabel("FCVC")
        plt.xticks(rotation=45, ha="right")
        save_fig("04_boxplot_fcvc.png")
        plt.show()

    # 5) Correlação numérica
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Mapa de Correlação (variáveis numéricas)")
        save_fig("05_correlacao_numericas.png")
        plt.show()


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Treina alguns modelos (baseline e fortes) e escolhe o melhor por acurácia.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results.append((name, acc))
        trained_pipelines[name] = pipe

        print(f"\n--- {name} ---")
        print(f"Acurácia: {acc:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_acc = results[0]
    best_pipe = trained_pipelines[best_name]

    print("\n======================")
    print("RESULTADO FINAL")
    print("======================")
    print("Ranking (maior acurácia primeiro):")
    for n, a in results:
        print(f"- {n}: {a:.4f}")

    print(f"\n✅ Melhor modelo: {best_name} | Acurácia: {best_acc:.4f}")

    # Relatório completo do melhor
    best_preds = best_pipe.predict(X_test)
    print("\nClassification Report (melhor modelo):")
    print(classification_report(y_test, best_preds))

    # Matriz de confusão
    plt.figure(figsize=(10, 7))
    ConfusionMatrixDisplay.from_predictions(y_test, best_preds, xticks_rotation=45)
    plt.title(f"Matriz de Confusão — {best_name}")
    save_fig("06_matriz_confusao_melhor_modelo.png")
    plt.show()

    return best_name, best_pipe, best_acc


def main():
    # 1) Carregar
    df = pd.read_csv(DATA_PATH)

    # 2) Detectar alvo
    target_col = detect_target_column(df)

    # 3) Checks básicos
    basic_checks(df, target_col)

    # 4) EDA + gráficos
    eda_plots(df, target_col)

    # 5) Preparar dados
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    print("\n======================")
    print("FEATURES")
    print("======================")
    print(f"Numéricas ({len(numeric_features)}): {numeric_features}")
    print(f"Categóricas ({len(categorical_features)}): {categorical_features}")

    # 6) Treinar e avaliar modelos
    best_name, best_pipe, best_acc = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # 7) Checagem do requisito do desafio (>= 0.75)
    # O desafio pede assertividade acima de 75%. :contentReference[oaicite:2]{index=2}
    if best_acc < 0.75:
        print("\n⚠️ Atenção: A acurácia ficou abaixo de 75%.")
        print("Sugestões rápidas:")
        print("- Tentar XGBoost/LightGBM (se permitido e instalado)")
        print("- Ajustar hiperparâmetros do RandomForest")
        print("- Testar CatBoost (lida bem com categóricas)")
    else:
        print("\n✅ Requisito atendido: acurácia acima de 75%.")

    # 8) Salvar pipeline
    model_path = os.path.join(MODELS_DIR, "model_pipeline.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\n[OK] Pipeline salva em: {model_path}")
    print("\nPróximo passo: usar esse arquivo no Streamlit para prever com inputs do usuário.")


if __name__ == "__main__":
    main()
