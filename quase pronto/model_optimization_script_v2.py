import pandas as pd
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
import warnings
import holidays

warnings.filterwarnings("ignore")

# Carregar os dados
json_path = 'datatran_consolidado.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Renomear colunas
df.rename(columns={'data_inversa': 'data', 'dia_semana': 'DiaSemana', 'tipo_acidente': 'TipoAcidente', 'condicao_metereologica': 'CondicaoMetereologica', 'uf': 'UF', 'municipio': 'Municipio'}, inplace=True)

# Amostrar os dados para reduzir o tempo de processamento para teste
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Limpar dados
df = df[df['TipoAcidente'].astype(str).str.lower() != 'none']
df['TipoAcidente'] = df['TipoAcidente'].str.replace(' ', '')

# Converter 'data' para datetime
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

# Feature Engineering
# 1. Features de tempo
df['horario_segundos'] = pd.to_timedelta(df['horario']).dt.total_seconds()
df['hora_do_dia'] = df['horario_segundos'] / 3600 # Horas decimais

# Transformação cíclica para horário
df['horario_sin'] = np.sin(2 * np.pi * df['hora_do_dia'] / 24)
df['horario_cos'] = np.cos(2 * np.pi * df['hora_do_dia'] / 24)

# 2. Features de data
df['dia_do_mes'] = df['data'].dt.day
df['mes'] = df['data'].dt.month
df['ano'] = df['data'].dt.year
df['dia_da_semana_num'] = df['data'].dt.dayofweek # 0=Segunda, 6=Domingo
df['dia_do_ano'] = df['data'].dt.dayofyear
df['semana_do_ano'] = df['data'].dt.isocalendar().week.astype(int)
df['trimestre'] = df['data'].dt.quarter

# 3. Feriados
br_holidays = holidays.Brazil()
df['is_holiday'] = df['data'].apply(lambda x: x in br_holidays).astype(int)

# Agrupar para criar o target (removendo TipoAcidente do agrupamento para evitar vazamento de dados)
# O agrupamento deve ser feito apenas pelas features que serão usadas como preditoras
df_grouped = df.groupby(['horario_sin', 'horario_cos', 'CondicaoMetereologica', 'DiaSemana', 'UF', 'Municipio', 'dia_do_mes', 'mes', 'ano', 'dia_da_semana_num', 'is_holiday', 'dia_do_ano', 'semana_do_ano', 'trimestre']).size().reset_index(name='quantidade_ocorrencias')

print(f"Shape do DataFrame agrupado: {df_grouped.shape}")

# Definir features categóricas e numéricas
categorical_features = ['CondicaoMetereologica', 'DiaSemana', 'UF', 'Municipio'] # 'TipoAcidente' é o que queremos prever, não uma feature para o agrupamento inicial
numerical_features = ['horario_sin', 'horario_cos', 'dia_do_mes', 'mes', 'ano', 'dia_da_semana_num', 'is_holiday', 'dia_do_ano', 'semana_do_ano', 'trimestre']

# Criar o preprocessor com OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Definir X e y
X = df_grouped[categorical_features + numerical_features]
y = df_grouped['quantidade_ocorrencias']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar o pré-processamento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Salvar o preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')

# Definir o modelo
lgbm = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1) # verbose=-1 para suprimir mensagens de log

# Definir os parâmetros para RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [20, 31],
    'max_depth': [5, 10],
    'min_child_samples': [20, 30],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [0.1, 0.5]
}

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=lgbm, param_distributions=param_dist, n_iter=2, cv=2, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', random_state=42) # Reduzido n_iter e cv para acelerar o treinamento

# Executar a busca aleatória
random_search.fit(X_train_processed, y_train)

# Obter os melhores parâmetros e o melhor modelo
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print(f"Melhores parâmetros encontrados: {best_params}")

# Avaliar o melhor modelo
y_pred = best_model.predict(X_test_processed)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
std_dev = np.std(y_test - y_pred)

print(f"Acurácia (R-squared) do modelo otimizado: {r2:.4f}")
print(f"RMSE do modelo otimizado: {rmse:.4f}")
print(f"Desvio Padrão dos Resíduos do modelo otimizado: {std_dev:.4f}")

# Salvar o melhor modelo treinado
joblib.dump(best_model, 'lightgbm_model.pkl')

print("Modelo LightGBM otimizado salvo como lightgbm_model.pkl")
