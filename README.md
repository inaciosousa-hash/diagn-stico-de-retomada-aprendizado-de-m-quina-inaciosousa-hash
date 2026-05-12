[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ARkoM8Jo)
# Diagnóstico de retomada - Aprendizado de Máquina

Esta atividade serve para mapear o que você já domina em Aprendizado de Máquina depois das atividades anteriores da disciplina.

Responda individualmente. Use suas palavras. Rode o código quando possível. Se usar IA depois da primeira tentativa, registre o uso na seção 8.

Prazo: 11/05/2026 às 23:59, horário de Fortaleza.

## 1. Mapa do que eu lembro

Marque cada tópico como: lembro bem, lembro parcialmente, não lembro, nunca vi ou não tenho certeza.

- vetores, matrizes e produto escalar: lembro bem
- média, desvio padrão e correlação: lembro parcialmente
- probabilidade condicional e Teorema de Bayes: lembro parcialmente
- regressão linear: lembro parcialmente
- classificação supervisionada: não lembro
- treino, teste e validação: lembro bem
- normalização ou padronização de dados: lembro parcialmente
- KNN: não lembro
- árvore de decisão: lembro parcialmente
- matriz de confusão: lembro bem
- acurácia, precisão, recall e F1-score: lembro bem
- overfitting e underfitting: lembro bem
- validação cruzada: não lembro
- Random Forest: lembro parcialmente
- XGBoost ou boosting: não lembro
- `predict_proba()`: lembro parcialmente
- SQL/ETL aplicado a dados: não lembro
- simulação de Monte Carlo: lembro bem

## 2. O que foi trabalhado antes

Explique, em 8 a 12 linhas:

1. quais desses tópicos você lembra de ter trabalhado na disciplina;
3. quais atividades ou exemplos você lembra;
4. o que você conseguiu fazer com autonomia;
5. o que você só conseguiu fazer seguindo roteiro;
6. qual assunto precisa ser retomado com mais urgência.
  Praticamente todos os tópicos foram abordados em um momento ou outro via atividades passadas pelo professor anterior.
Raramente os conteúdos eram lecionados em aula pois o horário era dedicado justamente a resolução das atividades
que eram feitas em grupo que ficavam a cargo de se organizarem e dividirem as tarefas. Os tópicos que marquei com
"lembro bem" na questão anterior são justamente aqueles com que tive participação direta/solo na pesquisa para a
resolução da questão. Creio que todos os tópicos com que trabalhei precisei de um auxílio ou outro, pois eu obtinha o
conhecimento justamente para resolver as atividades passadas. Minha opinião pessoal, mas creio que voltar um pouco para
a base seja interessante para o andamento da disciplina.
  
## 3. Conceitos essenciais

Responda com suas palavras e dê um exemplo simples.

1. O que é aprendizado supervisionado?
   É um tipo de aprendizado em que o modelo é treinado com um conjunto de dados já rotulados, ou seja, que já possuem
uma "resposta". O modelo é treinado nesses dados de modo que seja capaz de reduzir os erros em suas previsões comparado
com os valores reais, para assim desenvolver generalização e poder trabalhar com outros dados.

2. O que é uma tarefa de classificação?
   É a tarefa de classificar se um dado pertence a dado grupo ou classe.

3. O que são features e target?
   Features são os dados trabalhados como "características" para a análise do modelo, target é a resposta que o modelo 
deve fornecer. Ex: Num dos trabalhos que fizemos nós utilizamos um dataset de um grupo de mulheres que informava se elas
tinham ou não diabetes e diversas características como peso, glucose e pressão sanguínea. Essas características eram as
features do dataset enquanto que o target era a resposta para se elas tinham ou não diabetes.

4. Para que serve separar treino e teste?
   Serve para que o modelo possa ser treinado e executado num mesmo dataset.
   
5. O que é overfitting?
   É quando o modelo é extremamente ajustado para o dataset que está sendo utilizado para treino de modo que é capaz de
trabalhar muito bem com ele mas performa mal quando testado com outros datasets.

6. Por que acurácia pode ser uma métrica enganosa?
    Se o dataset que for utilizado para treino tiver dados desbalanceados, o modelo pode ser capaz de identificar corretamente
a classe majoritária muito bem mas se sair mal com as classes minoritárias. Ainda assim, devido a taxa de acertos com a classe
majoritárias, a Acurácia irá apontar o modelo como eficiente.
    
## 4. Diagnóstico prático com Scikit-Learn

No arquivo `diagnostico_ml.py`, use o dataset `load_breast_cancer` do Scikit-Learn e faça:

1. carregue os dados;
2. separe `X` e `y`;
3. divida em treino e teste;
4. treine uma regressão logística;
5. treine uma árvore de decisão;
6. mostre matriz de confusão, acurácia, precisão, recall e F1-score para cada modelo;
7. compare o desempenho em treino e teste;
8. escreva aqui qual modelo generalizou melhor e por quê.

Se não conseguir terminar tudo, registre até onde chegou e qual erro apareceu.

### Resultados

Cole aqui os principais resultados do seu código.

```text

```

### Interpretação

Qual modelo generalizou melhor? Explique usando as métricas e a comparação entre treino e teste.

Resposta:

## 5. Probabilidade e interpretação

Escolha um dos modelos treinados e responda:

1. O modelo produz probabilidade com `predict_proba()`?
2. O que significa uma probabilidade alta para uma classe?
3. Probabilidade alta garante que a previsão está correta? Explique.
4. Em um problema real, qual seria o risco de confiar cegamente nessa previsão?

Resposta:

## 6. Generalização

Compare treino e teste:

1. Há sinal de overfitting?
2. Há sinal de underfitting?
3. O que você tentaria mudar para melhorar o resultado?
4. O que você precisaria estudar melhor para responder com mais segurança?

Resposta:

## 7. Ponto de dificuldade

Escolha um tópico da lista inicial e escreva:

1. o que você entende dele;
2. onde você se confunde;
3. que tipo de explicação ajudaria: exemplo no quadro, notebook guiado, exercício curto, revisão matemática, visualização ou projeto pequeno.

Resposta:

## 8. Uso de IA, se houver

Se você usou IA depois da primeira tentativa, registre:

```text
Pergunta feita:
Resumo da resposta:
Como eu verifiquei:
O que eu alterei na minha resposta:
O que ainda não entendi:
```

## Submissão no Moodle

Depois de finalizar, copie no Moodle:

```text
Repositório:
Commit final:
Autoavaliação: nível atual, maior dificuldade e tópico que precisa ser retomado.
```
