import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# Carregar os dados
json_path = '/home/ubuntu/upload/datatran_consolidado.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Amostrar os dados para reduzir o tempo de processamento (20% dos dados)
df = df.sample(frac=0.05, random_state=42).reset_index(drop=True) # Reduzido para 5% para agilizar o GridSearchCV

# Renomear colunas
df.rename(columns={'data_inversa': 'data', 'dia_semana': 'DiaSemana', 'tipo_acidente': 'TipoAcidente', 'condicao_metereologica': 'CondicaoMetereologica', 'uf': 'UF', 'municipio': 'Municipio'}, inplace=True)

# Limpar dados
df = df[df['TipoAcidente'].astype(str).str.lower() != 'none']
df['TipoAcidente'] = df['TipoAcidente'].str.replace(' ', '')
df["horario"] = pd.to_timedelta(df["horario"]).dt.total_seconds()

# Agrupar para criar o target
df_grouped = df.groupby(["horario", "CondicaoMetereologica", "DiaSemana", "UF", "Municipio"]).size().reset_index(name='quantidade_ocorrencias')

# Codificar variáveis categóricas
le_condicao = LabelEncoder()
le_dia_semana = LabelEncoder()
le_uf = LabelEncoder()
le_municipio = LabelEncoder()

df_grouped['CondicaoMetereologica_encoded'] = le_condicao.fit_transform(df_grouped['CondicaoMetereologica'])
df_grouped['DiaSemana_encoded'] = le_dia_semana.fit_transform(df_grouped['DiaSemana'])
df_grouped['UF_encoded'] = le_uf.fit_transform(df_grouped['UF'])
df_grouped['Municipio_encoded'] = le_municipio.fit_transform(df_grouped['Municipio'])

# Definir X e y
X = df_grouped[['horario', 'CondicaoMetereologica_encoded', 'DiaSemana_encoded', 'UF_encoded', 'Municipio_encoded']]
y = df_grouped['quantidade_ocorrencias']

# Salvar os LabelEncoders
joblib.dump(le_condicao, 'le_condicao.pkl')
joblib.dump(le_dia_semana, 'le_dia_semana.pkl')
joblib.dump(le_uf, 'le_uf.pkl')
joblib.dump(le_municipio, 'le_municipio.pkl')

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o modelo
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Definir os parâmetros para GridSearchCV
# Expandindo o espaço de busca e ajustando para um tempo de execução razoável
param_grid = {
    'n_estimators': [50, 100, 150], # Reduzindo o máximo para 150 para economizar tempo
    'max_depth': [10, 20, 30], # Adicionando mais profundidade
    'min_samples_leaf': [1, 5, 10] # Ajustando min_samples_leaf
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Executar a busca em grade
grid_search.fit(X_train, y_train)

# Obter os melhores parâmetros e o melhor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Melhores parâmetros encontrados: {best_params}")

# Avaliar o melhor modelo
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
std_dev = np.std(y_test - y_pred)

print(f"Acurácia (R-squared) do modelo otimizado: {r2:.4f}")
print(f"RMSE do modelo otimizado: {rmse:.4f}")
print(f"Desvio Padrão dos Resíduos do modelo otimizado: {std_dev:.4f}")

# Salvar o melhor modelo treinado
joblib.dump(best_model, 'random_forest_model.pkl')

print("Modelo Random Forest otimizado salvo como random_forest_model.pkl")

