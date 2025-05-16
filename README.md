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
- R² (treino/teste): *aprox. 0.74 / 0.72*

### 2. Support Vector Regression (SVR)

`SVR()`

**Resultados:**
- R² (treino/teste): *aprox. 0.68 / 0.65*

### 3. Árvore de Decisão

`DecisionTreeRegressor(max_depth=4, random_state=7)`

**Resultados:**
- R² (treino/teste): *aprox. 0.81 / 0.78*

### 4. Random Forest

`RandomForestRegressor(max_depth=5, n_estimators=50, random_state=7)`

**Resultados:**
- R² (treino/teste): *aprox. 0.85 / 0.83*
- 
### 5. XGBoost

`XGBRegressor(max_depth=4, n_estimators=140, learning_rate=0.05)`

**Resultados:**
- R² (treino/teste): *aprox. 0.88 / 0.86*

### 6. LightGBM

`LGBMRegressor(max_depth=4, n_estimators=50, num_leaves=50, random_state=7)`

**Resultados:**
- R² (treino/teste): *aprox. 0.89 / 0.87*

## 🏆 Conclusões Principais

| Modelo            | R² Treino/Teste    |
|-------------------|--------------------|
| **LightGBM**      | 0.89 / 0.87 🥇     | 
| **XGBoost**       | 0.88 / 0.86 🥈     | 
| **Random Forest** | 0.85 / 0.83 🥉     |
| **Árvore Decisão**| 0.81 / 0.78        | 
| **Regressão Linear** | 0.74 / 0.72      | 
| **SVR**           | 0.68 / 0.65        | 

**Principais Insights:**
- A variável `smoker` apresenta a maior correlação positiva com o custo do seguro.
- Idade (`age`) e IMC (`bmi`) também influenciam significativamente os custos.
- Modelos ensemble (Random Forest, XGBoost, LightGBM) apresentaram melhor capacidade de generalização.

