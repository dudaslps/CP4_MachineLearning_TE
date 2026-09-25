# CP4_MachineLearning_TE

# Case iFood — Modelo Preditivo de Resposta a Campanha de Marketing

Projeto de Machine Learning (classificação binária) que parte dos modelos apresentados em aula
e busca **superar o melhor ROC AUC da aula** por meio de Feature Engineering e tuning de
hiperparâmetros com validação cruzada.

## Contexto de negócio

O iFood realizou uma campanha piloto para testar a aceitação de um novo produto (um gadget)
junto a **2.240 clientes** selecionados aleatoriamente e contatados por telefone. O resultado
da piloto:

| Indicador | Valor |
|---|---|
| Custo total da campanha (amostra) | 6.720 MU |
| Receita gerada pelos clientes que aceitaram | 3.674 MU |
| Lucro global da campanha | **-3.046 MU** |
| Taxa de sucesso (conversão) | 15% |

A campanha piloto deu prejuízo: contatar clientes indiscriminadamente gera custo sem garantir
receita. O objetivo do case é usar Machine Learning para prever, entre os clientes, **quais têm
maior probabilidade de responder** à próxima campanha — permitindo direcionar o contato apenas
aos clientes mais propensos, tornando a campanha lucrativa, além de ajudar a entender o perfil
dos clientes com interesse no produto.

## Dataset

`data_1.csv` — 2.240 clientes, 29 colunas, reunindo três tipos de informação:

| Tipo | Exemplos |
|---|---|
| Sociodemográficas | idade, estado civil, filhos, escolaridade, renda anual |
| Comportamentais | canal de compra (loja, catálogo, site), gasto por categoria, recência, aceitação de campanhas anteriores |
| Alvo (target) | `Response` — 1 se o cliente respondeu à campanha piloto, 0 caso contrário |

## Objetivo técnico

Este notebook parte de 3 notebooks de referência apresentados em aula (baselines, boosting e
tuning) e busca **superar o melhor ROC AUC visto em aula**, seguindo quatro etapas:

1. **EDA objetiva**, propondo novas features e justificando as escolhas a partir da correlação
   com o target;
2. **Preparação dos dados e holdout estratificado**, com o conjunto de teste reservado
   exclusivamente para a avaliação final;
3. **Tuning de hiperparâmetros com validação cruzada** (busca manual, Grid Search, Randomized
   Search e Bayesian Search);
4. **Comparação final** entre os modelos da aula e os novos modelos, usando Precision, Recall,
   F1-score e ROC AUC.

## Metodologia

### Feature Engineering

A matriz de correlação com o target mostrou que os sinais mais fortes de resposta eram gasto
por categoria, compras por catálogo e aceitação de campanhas anteriores — todos dispersos em
várias colunas. Isso motivou a criação de features agregadas:

- `TotalMnt` — soma de todas as colunas `Mnt*` (gasto total por categoria);
- `TotalPurchases` — soma das colunas `Num*Purchases` (total de compras por canal);
- `TotalAcceptedCmp` — soma de `AcceptedCmp1..5` (campanhas anteriores aceitas);
- `Age`, `Days_Customer` (tempo de casa), `Kids_Total`/`Has_Child`, `Family_Size`;
- `AvgTicket` (gasto médio por compra) e `Income_per_Person` (renda per capita na família).

Também foram removidos 4 registros de cadastro inconsistentes (idade > 107 anos e renda de
666.666) e agrupadas categorias raras de estado civil ("Alone", "Absurd", "YOLO") em "Other".

### Holdout estratificado

Split 80/20 (`train_test_split(..., stratify=y, random_state=SEED)`), preservando a proporção
da classe positiva (~15%) em treino e teste. O conjunto de teste não é tocado em nenhuma etapa
de tuning ou validação cruzada — só é usado na comparação final.

### Tuning com validação cruzada

Todos os modelos (Regressão Logística, Decision Tree, Random Forest, LightGBM, XGBoost,
CatBoost) foram ajustados com `StratifiedKFold(n_splits=5)`, testando as quatro técnicas de
busca ensinadas em aula (manual, Grid Search, Randomized Search, Bayesian Search).

A principal correção de método em relação à aula: a busca de hiperparâmetros da aula otimizava
`precision`, o que reduzia o ROC AUC dos modelos tunados. Aqui, toda busca otimiza `roc_auc`
diretamente — essa mudança de métrica de otimização foi decisiva para o resultado final.

## Resultado final

Avaliação feita uma única vez no conjunto de teste (`Xtest`/`ytest`), nunca usado durante o
tuning:

| Modelo | Precision | Recall | F1-score | ROC AUC |
|---|---|---|---|---|
| **XGBoost (novo)** | 0.727 | 0.478 | 0.577 | **0.9153** |
| CatBoost (aula) — melhor da aula | 0.844 | 0.403 | 0.545 | 0.9122 |
| XGBoost (aula) | 0.737 | 0.418 | 0.533 | 0.9117 |
| CatBoost (novo) | 0.794 | 0.403 | 0.535 | 0.9096 |
| LightGBM (novo) | 0.702 | 0.493 | 0.579 | 0.9057 |
| Regressão Logística (aula) | 0.448 | 0.776 | 0.568 | 0.8994 |
| Random Forest (novo) | 0.604 | 0.478 | 0.533 | 0.8957 |
| Decision Tree (aula) | 0.463 | 0.284 | 0.352 | 0.7930 |

> Tabela completa (17 modelos) disponível no notebook, seção "Comparação final".

**Desafio cumprido:** o XGBoost com Feature Engineering + tuning por ROC AUC atingiu
**ROC AUC = 0.9153**, superando o melhor modelo da aula (CatBoost, 0.9122) em **+0.0031**,
recuperando recall (+0.075) e mantendo um F1-score mais equilibrado.

## Estrutura do repositório

```
.
├── Case_Ifood_Colearning_CP4_ML.ipynb   # notebook principal (EDA, features, holdout, tuning, comparação final)
├── data_1.csv                           # dataset do case (2.240 clientes)
└── README.md
```

## Como executar

1. Abra o notebook `Case_Ifood_Colearning_CP4_ML.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Faça upload do arquivo `data_1.csv` para o ambiente do Colab (ou monte o Google Drive).
3. Execute as células em ordem, de cima para baixo — a primeira célula instala as bibliotecas
   de boosting (`lightgbm`, `xgboost`, `catboost`) e de busca bayesiana (`scikit-optimize`).

## Tecnologias

- Python, pandas, numpy
- scikit-learn (pipelines, `GridSearchCV`, `RandomizedSearchCV`, `StratifiedKFold`)
- LightGBM, XGBoost, CatBoost
- scikit-optimize (`BayesSearchCV`)
- matplotlib

## Equipe

Arthur Costa Donaire · Felipe Pereira de Jesus · Giovanna Pereira de Oliveira ·
Gustavo Paiva Silva · Maria Eduarda Soares Lopes e Souza
