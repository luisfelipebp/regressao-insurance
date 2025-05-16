# Insurance Cost Prediction - Regressão com Machine Learning

Projeto de comparação de modelos de regressão para previsão de custos de seguro saúde usando o [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

## 📌 Visão Geral
Este projeto avalia o desempenho de 6 algoritmos de regressão para prever o valor do seguro médico individual:

- **Regressão Linear**
- **Support Vector Regression (SVR)**
- **Árvore de Decisão**
- **Random Forest**
- **XGBoost**
- **LightGBM**

**Objetivo:** Identificar o modelo mais eficaz para prever o custo do seguro (`charges`) a partir de dados demográficos e comportamentais.

## 📊 Dataset
- **1.338 amostras** e **7 variáveis**:
  - `age`: Idade do beneficiário
  - `sex`: Sexo (`male`/`female`)
  - `bmi`: Índice de Massa Corporal
  - `children`: Número de dependentes
  - `smoker`: Fumante (`yes`/`no`)
  - `region`: Região (`southwest`, `southeast`, `northwest`, `northeast`)
  - `charges`: Custo do seguro (variável alvo)
- **Pré-processamento:** Codificação de variáveis categóricas em valores numéricos.
- Divisão treino-teste: **70% treino** | **30% teste**

## 🧠 Modelos Avaliados

### 1. Regressão Linear

`LinearRegression()`

**Resultados:**
- R² (treino/teste): *aprox. 0.61 / 0.62*
- Validação cruzada (15 folds): 61.26%
- RMSE (Raiz do erro quadrático médio): 7380.55

### 2. Regressão Linear Múltipla

`LinearRegression()`

**Resultados:**
- R² (treino/teste): *aprox. 0.74 / 0.75*
- Validação cruzada (15 folds): 74.44%
- RMSE (Raiz do erro quadrático médio): 6017.57
- 
### 3. Support Vector Regression (SVR)

`SVR(kernel='rbf')`

**Resultados:**
- R² (treino/teste): *aprox. 0.86 / 0.84*
- Validação cruzada (15 folds): 84.46%
- RMSE (Raiz do erro quadrático médio): 4792.53

### 4. Árvore de Decisão

`DecisionTreeRegressor(max_depth=4, random_state=7)`

**Resultados:**
- R² (treino/teste): *aprox. 0.87 / 0.85*
- Validação cruzada (15 folds): 84.98%
- RMSE (Raiz do erro quadrático médio): 4647.72

### 5. Random Forest

`RandomForestRegressor(n_estimators=50, criterion='squared_error', max_depth=5, random_state = 7)`

**Resultados:**
- R² (treino/teste): *aprox. 0.89 / 0.85*
- Validação cruzada (15 folds):  85.23%
- RMSE (Raiz do erro quadrático médio): 4668.64
- 
### 6. XGBoost

`XGBRegressor(n_estimators=140, max_depth=4, learning_rate=0.05, objective="reg:squarederror", random_state=7)`

**Resultados:**
- R² (treino/teste): *aprox. 0.91 / 0.85
- Validação cruzada (15 folds): 85.85%
- RMSE (Raiz do erro quadrático médio): 4621.63

### 7. LightGBM

LGBMRegressor(num_leaves=50, max_depth=4, learning_rate=0.1, n_estimators=50, random_state=7)`

**Resultados:**
- R² (treino/teste): *aprox. 0.89 / 0.84*
- Validação cruzada (15 folds): 85.91%
- RMSE (Raiz do erro quadrático médio): 4782.14

### 8. CATBOOST

`CatBoostRegressor (iterations=230, learning_rate=0.05, depth = 6, random_state = 7, verbose=False)`

**Resultados:**
- R² (treino/teste): *aprox. 0.91 / 0.85*
- Validação cruzada (15 folds): 85.57%
- RMSE (Raiz do erro quadrático médio): 4594.31


## 🏆 Conclusões Principais

| Modelo            | R² Treino/Teste    |
|-------------------|--------------------|
| **CATBOOST**      | 0.91 / 0.85 🥇     | 
| **LightGBM**      | 0.89 / 0.84 🥈     | 
| **XGBoost**       | 0.91 / 0.85 🥉     | 
| **Random Forest** | 0.85 / 0.83        |
| **Árvore Decisão**| 0.81 / 0.78        | 
| **Regressão Linear** | 0.74 / 0.72      | 
| **SVR**           | 0.68 / 0.65        | 

**Principais Observações:**
- A variável `smoker` apresenta a maior correlação positiva com o custo do seguro.
- Idade (`age`) e IMC (`bmi`) também influenciam significativamente os custos.
- Modelos ensemble (Random Forest, XGBoost, LightGBM) apresentaram melhor capacidade de generalização.

