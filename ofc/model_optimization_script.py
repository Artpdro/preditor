import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# Carregar os dados
json_path = 'datatran_consolidado.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Renomear colunas
df.rename(columns={'data_inversa': 'data'}, inplace=True)
df.rename(columns={'dia_semana': 'DiaSemana'}, inplace=True)
df.rename(columns={'tipo_acidente': 'TipoAcidente'}, inplace=True)
df.rename(columns={'condicao_metereologica': 'CondicaoMetereologica'}, inplace=True)

# Limpar dados
df = df[df['TipoAcidente'].astype(str).str.lower() != 'none']
df['TipoAcidente'] = df['TipoAcidente'].str.replace(' ', '')
df["horario"] = pd.to_timedelta(df["horario"]).dt.total_seconds()

# Agrupar para criar o target
df_grouped = df.groupby(["horario", "CondicaoMetereologica", "DiaSemana"]).size().reset_index(name='quantidade_ocorrencias')

# Codificar variáveis categóricas
le_condicao = LabelEncoder()
le_dia_semana = LabelEncoder()

df_grouped['CondicaoMetereologica_encoded'] = le_condicao.fit_transform(df_grouped['CondicaoMetereologica'])
df_grouped['DiaSemana_encoded'] = le_dia_semana.fit_transform(df_grouped['DiaSemana'])

# Definir X e y
X = df_grouped[['horario', 'CondicaoMetereologica_encoded', 'DiaSemana_encoded']]
y = df_grouped['quantidade_ocorrencias']

# Salvar os LabelEncoders
joblib.dump(le_condicao, 'le_condicao.pkl')
joblib.dump(le_dia_semana, 'le_dia_semana.pkl')

# Otimização de hiperparâmetros com GridSearchCV (abordagem mais simples)
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20],
    'min_samples_leaf': [2, 4]
}

model = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2, scoring='r2') # Reduzido cv para 2
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

print(f"Melhores hiperparâmetros: {grid_search.best_params_}")

# Avaliar o melhor modelo usando validação cruzada
scores = cross_val_score(best_model, X, y, cv=3, scoring='r2') # Reduzido cv para 3

print(f"Acurácia média (R-squared) do modelo otimizado: {np.mean(scores):.4f}")
print(f"Desvio padrão da acurácia do modelo otimizado: {np.std(scores):.4f}")

# Treinar o modelo final com os melhores hiperparâmetros em todos os dados
best_model.fit(X, y)

# Salvar o modelo treinado
joblib.dump(best_model, 'random_forest_model.pkl')

print("Modelo Random Forest otimizado e salvo como random_forest_model.pkl")

