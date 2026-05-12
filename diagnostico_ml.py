"""Diagnóstico de retomada em Aprendizado de Máquina.

Complete os TODOs, rode o arquivo e copie os resultados principais para o
README.md.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def avaliar_modelo(nome, modelo, x_train, x_test, y_train, y_test):
    """Treina, prediz e imprime métricas simples."""
    # TODO 5: treine o modelo com os dados de treino.
    modelo.fit(x_train, y_train)

    # TODO 6: gere as previsões para treino e teste.
    y_train_pred = modelo.predict(x_train)
    y_test_pred = modelo.predict(x_test)

    # TODO 7: imprima acurácia de treino, acurácia de teste, precisão, recall,
    # F1-score e matriz de confusão usando as funções importadas acima.
    print(f"\n--- {nome} ---")
    print(f"Acurácia treino: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Acurácia teste : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precisão (teste): {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall (teste)  : {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-score (teste): {f1_score(y_test, y_test_pred):.4f}")
    print("Matriz de confusão (teste):")
    print(confusion_matrix(y_test, y_test_pred))

    # TODO 8: se o modelo tiver predict_proba, mostre as probabilidades das
    # cinco primeiras amostras de teste.
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(x_test[:5])
        print("\nProbabilidades (5 primeiras amostras de teste):")
        for i, prob in enumerate(probabilidades):
            print(f"Amostra {i+1}: classe 0={prob[0]:.3f}, classe 1={prob[1]:.3f}")


def main():
    # TODO 1: carregue o dataset load_breast_cancer().
    dados = load_breast_cancer()

    # TODO 2: separe os dados de entrada em X e o alvo em y.
    X = dados.data
    y = dados.target

    # TODO 3: imprima informações básicas: nomes das primeiras features,
    # classes, quantidade de exemplos e quantidade de features.
    print("Nomes das primeiras 5 features:", dados.feature_names[:5])
    print("Classes:", dados.target_names)
    print("Quantidade de exemplos:", X.shape[0])
    print("Quantidade de features:", X.shape[1])

    # TODO 4: divida X e y em treino e teste com train_test_split.
    # Use test_size=0.25, random_state=42 e stratify=y.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # TODO 9: crie uma regressão logística e uma árvore de decisão.
    log_reg = LogisticRegression(max_iter=2000, random_state=42)
    dec_tree = DecisionTreeClassifier(random_state=42)

    # TODO 10: chame avaliar_modelo para os dois modelos.
    avaliar_modelo("Regressão Logística", log_reg, X_train, X_test, y_train, y_test)
    avaliar_modelo("Árvore de Decisão", dec_tree, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
