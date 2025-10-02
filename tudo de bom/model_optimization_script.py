import pandas as pd
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor, Pool
import warnings
import datetime

warnings.filterwarnings("ignore")

# Carregar os dados
json_path = 'datatran_consolidado.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Se houver coluna de data, transformar para datetime
# Supondo que 'data' ou outra coluna possua data
if 'data_inversa' in df.columns:
    df['data'] = pd.to_datetime(df['data_inversa'], format='%Y-%m-%d', errors='coerce')
else:
    # Se tiver outra coluna de data, ajustar
    pass

# Remover entradas com data inválida
df = df.dropna(subset=['data'])

# Ordenar por data para evitar vazamento temporal
df = df.sort_values('data').reset_index(drop=True)

# Feature engineering de tempo
df['Ano'] = df['data'].dt.year
df['Mes'] = df['data'].dt.month
df['Dia'] = df['data'].dt.day
df['Hora'] = pd.to_timedelta(df['horario']).dt.total_seconds() / 3600  # transformar em horas, por exemplo
# se "horario" contiver horas
# Crie faixas de horário
df['FaixaHorario'] = pd.cut(df['Hora'], bins=[0,6,12,18,24], labels=[0,1,2,3], include_lowest=True)

# Renomear colunas
df.rename(columns={
    'dia_semana': 'DiaSemana',
    'tipo_acidente': 'TipoAcidente',
    'condicao_metereologica': 'CondicaoMetereologica',
    'uf': 'UF',
    'municipio': 'Municipio'
}, inplace=True)

# Limpar
df = df[df['TipoAcidente'].astype(str).str.lower() != 'none']
df['TipoAcidente'] = df['TipoAcidente'].str.replace(' ', '')

# Agrupar ou criar target
# Talvez não agrupar demais; se "quantidade_ocorrencias" for a target, considerar não agrupar por Município se isso for fragmentar demais
df_grouped = df.groupby(["data", "Hora", "CondicaoMetereologica", "DiaSemana", "UF", "Municipio"]).size().reset_index(name='quantidade_ocorrencias')

# Opcional: transformar target com log se skew
df_grouped['quantidade_log1p'] = np.log1p(df_grouped['quantidade_ocorrencias'])

# Codificar categóricas
from sklearn.preprocessing import LabelEncoder

cat_cols = ['CondicaoMetereologica', 'DiaSemana', 'UF', 'Municipio', 'FaixaHorario']
le = {}
for c in cat_cols:
    le[c] = LabelEncoder()
    df_grouped[c + '_enc'] = le[c].fit_transform(df_grouped[c].astype(str))

# Salvar encoders
for c in cat_cols:
    joblib.dump(le[c], f'le_{c}.pkl')

# Definir X e y
feature_cols = ['Hora', 'CondicaoMetereologica_enc', 'DiaSemana_enc', 'UF_enc', 'Municipio_enc', 'FaixaHorario_enc', 'Mes', 'Ano']
X = df_grouped[feature_cols]
y = df_grouped['quantidade_log1p']  # usar log1p como target

# Dividir train/test com cuidado temporal
# Exemplo: últimos 20% do tempo como teste
split_index = int(len(df_grouped) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Criar Pool do CatBoost para validação
train_pool = Pool(X_train, label=y_train, cat_features=[feature_cols.index(c + '_enc') for c in cat_cols])
valid_pool = Pool(X_test, label=y_test, cat_features=[feature_cols.index(c + '_enc') for c in cat_cols])

# Parâmetros iniciais
base_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'R2',
    'random_seed': 42,
    'early_stopping_rounds': 50,
    'verbose': 100,
    'use_best_model': True
}

# Espaço de busca mais amplo e refinado
from sklearn.model_selection import ParameterSampler

param_dist = {
    'iterations': [1000, 2000, 3000],
    'learning_rate': [0.005, 0.01, 0.02, 0.05],
    'depth': [6, 8, 10, 12],
    'l2_leaf_reg': [1, 3, 5, 7, 10],
    'bagging_temperature': [0.0, 0.5, 1.0, 2.0],
    'random_strength': [0.5, 1, 2, 5],
    'border_count': [32, 64, 128, 254],
    'bootstrap_type': ['Bayesian', 'Poisson', 'Bernoulli'],
    'one_hot_max_size': [2, 5, 10, 20]
}

# Número de iterações de busca
n_iter = 20  # ajustar se tiver mais tempo

best_score = -np.inf
best_params = None
best_model = None

for params in ParameterSampler(param_dist, n_iter=n_iter, random_state=42):
    try:
        model = CatBoostRegressor(**base_params, **params)
        model.fit(train_pool, eval_set=valid_pool)
        preds = model.predict(X_test)
        # desfazer a transformação log para comparação se quiser
        preds_orig = np.expm1(preds)
        y_test_orig = np.expm1(y_test)
        r2 = r2_score(y_test_orig, preds_orig)
        if r2 > best_score:
            best_score = r2
            best_params = params
            best_model = model
            print("→ Novo melhor R2:", best_score, "com params:", best_params)
    except Exception as e:
        print("Erro com params", params, e)
        continue

print("\n=== Melhor pontuação R2 final:", best_score)
print("=== Parâmetros correspondentes:", best_params)

# Avaliação final no conjunto teste
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_true = np.expm1(y_test)

r2_final = r2_score(y_test_true, y_pred)
rmse_final = np.sqrt(mean_squared_error(y_test_true, y_pred))
residuos = y_test_true - y_pred
std_final = np.std(residuos)

print(f"R² final: {r2_final:.4f}")
print(f"RMSE final: {rmse_final:.4f}")
print(f"Desvio padrão dos resíduos: {std_final:.4f}")

# Salvar modelo
joblib.dump(best_model, 'catboost_model_super_optimizado.pkl')
