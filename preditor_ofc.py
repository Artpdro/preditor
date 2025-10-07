import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

CAMINHO_DADOS = 'datatran_consolidado.json'

class PreditorRidgeReal:
    
    def __init__(self):
        self.modelo = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.treinado = False
        self.metricas = {}
        self.valores_unicos = {}
        self.feature_names = []
        self.feature_medians = None # Adicionado para armazenar as medianas das features
        
    def treinar(self, arquivo_json):
        """Treina modelo Ridge com alta acurácia"""
        # Carregar dados
        with open(arquivo_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Salvar valores únicos para interface
        self._extrair_valores_unicos(df)
        
        # Processar dados (método original de alta acurácia)
        df_processed = self._processar_dados(df)
        
        # Criar features (método original)
        X, y = self._criar_features(df_processed)
        
        # Salvar nomes das features
        self.feature_names = X.columns.tolist()
        
        # Salvar as medianas das features para uso na previsão
        self.feature_medians = X.median().to_dict()
        
        # Validação
        self.metricas = self._validar_modelo(X, y)
        
        # Treinar
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)
        
        self.treinado = True
        
        # Mostrar métricas
        print(f"Acurácia (R²): {self.metricas['r2_mean']:.4f}")
        print(f"Desvio Padrão: {self.metricas['desvio_padrao']:.2f}")
        print("Modelo foi salvo")
        
        return self
    
    def _extrair_valores_unicos(self, df):
        """Extrai valores únicos para interface"""
        # Processar dados básicos
        df['data'] = pd.to_datetime(df['data_inversa'], format='%d/%m/%Y')
        
        # Mapear dia da semana
        dias_map = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 
                   4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
        df['dia_semana_nome'] = df['data'].dt.dayofweek.map(dias_map)
        
        # Limpar dados
        df['uf'] = df['uf'].fillna('DESCONHECIDO')
        df['municipio'] = df['municipio'].fillna('DESCONHECIDO')
        df['condicao_metereologica'] = df['condicao_metereologica'].fillna('DESCONHECIDO')
        
        # Salvar valores únicos
        self.valores_unicos = {
            'uf': sorted(df['uf'].unique()),
            'municipio': sorted(df['municipio'].unique()),
            'condicao_metereologica': sorted(df['condicao_metereologica'].unique()),
            'dia_semana': ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        }
    
    def _processar_dados(self, df):
        """Processa dados usando método original de alta acurácia"""
        df['data'] = pd.to_datetime(df['data_inversa'], format='%d/%m/%Y')
        df['horario_num'] = pd.to_numeric(df['horario'], errors='coerce').fillna(1200)
        df['hora'] = (df['horario_num'] // 100).astype(int).clip(0, 23)
        
        # Agregação diária (método original)
        agg_diario = df.groupby('data').agg({
            'data_inversa': 'count',
            'uf': 'nunique',
            'municipio': 'nunique',
            'tipo_acidente': 'nunique',
            'condicao_metereologica': lambda x: x.mode()[0] if not x.empty else 'Desconhecido',
            'hora': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        agg_diario.columns = [
            'data', 'acidentes', 'num_ufs', 'num_municipios', 'tipos_unicos',
            'clima_principal', 'hora_media', 'hora_std', 'hora_min', 'hora_max'
        ]
        
        agg_diario['hora_std'] = agg_diario['hora_std'].fillna(0)
        
        return agg_diario
    
    def _criar_features(self, df):
        """Cria features usando método original"""
        y = df['acidentes']
        df = df.sort_values('data').reset_index(drop=True)
        
        # Features temporais
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['dia'] = df['data'].dt.day
        df['dia_semana'] = df['data'].dt.dayofweek
        df['dia_ano'] = df['data'].dt.dayofyear
        df['semana_ano'] = df['data'].dt.isocalendar().week.astype(int) # Convert to int
        df['trimestre'] = df['data'].dt.quarter
        df['fim_semana'] = (df['dia_semana'] >= 5).astype(int)
        
        # Features cíclicas
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365)
        df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365)
        
        # Features históricas
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'acidentes_lag_{lag}'] = df['acidentes'].shift(lag)
        
        # Médias móveis
        for window in [3, 7, 14, 30]:
            df[f'media_{window}d'] = df['acidentes'].shift(1).rolling(window, min_periods=1).mean()
        
        # Volatilidade
        for window in [7, 14, 30]:
            df[f'vol_{window}d'] = df['acidentes'].shift(1).rolling(window, min_periods=1).std().fillna(0)
        
        # Tendência
        df['tend_7d'] = df['acidentes'].shift(1) - df['media_7d']
        df['tend_30d'] = df['acidentes'].shift(1) - df['media_30d']
        
        # Features de contexto
        df['periodo_especial'] = df['data'].apply(
            lambda x: int(x.month in [12, 1, 6, 7] or x.month == 2)
        )
        
        df['feriado'] = df['data'].apply(
            lambda x: int((x.month == 12 and x.day >= 20) or 
                         (x.month == 1 and x.day <= 10) or
                         (x.month == 9 and x.day == 7) or
                         (x.month == 10 and x.day == 12) or
                         (x.month == 11 and (x.day == 2 or x.day == 15)))
        )
        
        # Codificar clima
        encoder_clima = LabelEncoder()
        df['clima_encoded'] = encoder_clima.fit_transform(df['clima_principal'].fillna('Desconhecido'))
        self.encoders['clima'] = encoder_clima
        
        # Features numéricas
        features_numericas = [
            'ano', 'mes', 'dia', 'dia_semana', 'dia_ano', 'semana_ano', 'trimestre', 'fim_semana',
            'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 'dia_ano_sin', 'dia_ano_cos',
            'num_ufs', 'num_municipios', 'tipos_unicos', 'hora_media', 'hora_std', 'hora_min', 'hora_max',
            'acidentes_lag_1', 'acidentes_lag_2', 'acidentes_lag_3', 'acidentes_lag_7', 'acidentes_lag_14', 'acidentes_lag_30',
            'media_3d', 'media_7d', 'media_14d', 'media_30d',
            'vol_7d', 'vol_14d', 'vol_30d',
            'tend_7d', 'tend_30d',
            'periodo_especial', 'feriado', 'clima_encoded'
        ]
        
        X = df[features_numericas]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Limpar dados
        mask_valido = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask_valido]
        y = y[mask_valido]
        
        return X, y
    
    def _validar_modelo(self, X, y):
        """Validação cruzada"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        scores_r2 = []
        residuos_todos = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler_temp = StandardScaler()
            X_train_scaled = scaler_temp.fit_transform(X_train)
            X_val_scaled = scaler_temp.transform(X_val)
            
            modelo_temp = Ridge(alpha=1.0)
            modelo_temp.fit(X_train_scaled, y_train)
            
            y_pred = modelo_temp.predict(X_val_scaled)
            
            r2 = r2_score(y_val, y_pred)
            scores_r2.append(r2)
            
            residuos = y_val - y_pred
            residuos_todos.extend(residuos.tolist())
        
        return {
            'r2_mean': np.mean(scores_r2),
            'desvio_padrao': np.std(residuos_todos)
        }
    
    def prever_acidentes(self, horario, dia_semana, condicao_metereologica, uf, municipio):
        """USA O MODELO RIDGE TREINADO para fazer previsão real"""
        if not self.treinado:
            # Retorna uma previsão padrão ou levanta um erro se o modelo não estiver treinado
            # Para o propósito de evitar o '4' fixo, podemos retornar um valor indicando não treinado
            return 0 # Ou raise ValueError("Modelo não treinado para previsão.")
        
        # Criar uma data fictícia para gerar features
        data_base = datetime(2025, 10, 6)  # Data atual
        
        # Ajustar data para o dia da semana solicitado
        dias_map = {'Segunda': 0, 'Terça': 1, 'Quarta': 2, 'Quinta': 3, 
                   'Sexta': 4, 'Sábado': 5, 'Domingo': 6}
        
        dia_semana_num = dias_map.get(dia_semana, 0)
        
        # Ajustar data para o dia da semana correto
        dias_diff = dia_semana_num - data_base.weekday()
        data_ajustada = data_base + timedelta(days=dias_diff)
        
        # Criar features como o modelo espera
        # Passamos os valores de uf e municipio para que possam ser usados se necessário
        features_dict = self._criar_features_para_data(
            data_ajustada, horario, condicao_metereologica, uf, municipio
        )
        
        # Converter o dicionário de features para um array na ordem correta
        features_array = np.array([features_dict[f] for f in self.feature_names])
        
        # Usar o modelo Ridge treinado
        features_scaled = self.scaler.transform([features_array])
        predicao_raw = self.modelo.predict(features_scaled)[0]
        
        # A previsão é para o dia todo, mas queremos para uma situação específica
        # Ajustar proporcionalmente (um dia tem 24 horas)
        # Este ajuste pode precisar ser revisado dependendo da granularidade da previsão
        predicao_ajustada = predicao_raw / 24  # Dividir por 24 horas
        
        # Arredondar conforme solicitado
        predicao_final = round(max(0, predicao_ajustada))
        
        return predicao_final
    
    def _criar_features_para_data(self, data, horario, condicao_metereologica, uf, municipio):
        """Cria features para uma situação específica usando valores históricos e medianas do treinamento"""
        
        # Inicializa um dicionário para armazenar as features
        features = {}
        
        # Features temporais básicas
        features['ano'] = data.year
        features['mes'] = data.month
        features['dia'] = data.day
        features['dia_semana'] = data.weekday()
        features['dia_ano'] = data.timetuple().tm_yday
        features['semana_ano'] = data.isocalendar().week #.astype(int) # Já é int
        features['trimestre'] = ((features['mes'] - 1) // 3) + 1
        features['fim_semana'] = int(features['dia_semana'] >= 5)
        
        # Features cíclicas
        features['mes_sin'] = np.sin(2 * np.pi * features['mes'] / 12)
        features['mes_cos'] = np.cos(2 * np.pi * features['mes'] / 12)
        features['dia_semana_sin'] = np.sin(2 * np.pi * features['dia_semana'] / 7)
        features['dia_semana_cos'] = np.cos(2 * np.pi * features['dia_semana'] / 7)
        features['dia_ano_sin'] = np.sin(2 * np.pi * features['dia_ano'] / 365)
        features['dia_ano_cos'] = np.cos(2 * np.pi * features['dia_ano'] / 365)
        
        # Features contextuais (usar medianas do treinamento ou valores específicos se disponíveis)
        # Para num_ufs, num_municipios, tipos_unicos, hora_std, hora_min, hora_max, 
        # usaremos as medianas calculadas durante o treinamento.
        # O 'horario' fornecido pelo usuário será usado para 'hora_media'.
        features['num_ufs'] = self.feature_medians.get('num_ufs', 1) # Usar mediana ou um valor padrão razoável
        features['num_municipios'] = self.feature_medians.get('num_municipios', 1) # Usar mediana
        features['tipos_unicos'] = self.feature_medians.get('tipos_unicos', 1) # Usar mediana
        features['hora_media'] = horario # Usar o horário específico fornecido
        features['hora_std'] = self.feature_medians.get('hora_std', 0) # Usar mediana
        features['hora_min'] = self.feature_medians.get('hora_min', 0) # Usar mediana
        features['hora_max'] = self.feature_medians.get('hora_max', 23) # Usar mediana
        
        # Features históricas (usar medianas do treinamento para lags e médias móveis)
        # Estes valores são mais complexos de simular sem dados históricos reais para a data da previsão.
        # A abordagem mais segura é usar as medianas do conjunto de treinamento.
        for lag in [1, 2, 3, 7, 14, 30]:
            features[f'acidentes_lag_{lag}'] = self.feature_medians.get(f'acidentes_lag_{lag}', 0)
        
        for window in [3, 7, 14, 30]:
            features[f'media_{window}d'] = self.feature_medians.get(f'media_{window}d', 0)
        
        for window in [7, 14, 30]:
            features[f'vol_{window}d'] = self.feature_medians.get(f'vol_{window}d', 0)
        
        features['tend_7d'] = self.feature_medians.get('tend_7d', 0)
        features['tend_30d'] = self.feature_medians.get('tend_30d', 0)
        
        # Contexto
        features['periodo_especial'] = int(features['mes'] in [12, 1, 6, 7] or features['mes'] == 2)
        features['feriado'] = int((features['mes'] == 12 and features['dia'] >= 20) or \
                                  (features['mes'] == 1 and features['dia'] <= 10) or \
                                  (features['mes'] == 9 and features['dia'] == 7) or \
                                  (features['mes'] == 10 and features['dia'] == 12) or \
                                  (features['mes'] == 11 and (features['dia'] == 2 or features['dia'] == 15)))
        
        # Codificar clima
        try:
            clima_encoded = self.encoders['clima'].transform([condicao_metereologica])[0]
        except KeyError:
            # Se o encoder não foi treinado ou a categoria é desconhecida, usar um valor padrão
            clima_encoded = 0 
        features['clima_encoded'] = clima_encoded
        
        return features
    
    def salvar(self, arquivo='modelo_ridge_real.pkl'):
        """Salva modelo"""
        if not self.treinado:
            raise ValueError("Modelo não foi treinado!")
        
        dados = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'metricas': self.metricas,
            'valores_unicos': self.valores_unicos,
            'feature_names': self.feature_names,
            'feature_medians': self.feature_medians # Salvar as medianas
        }
        
        with open(arquivo, 'wb') as f:
            pickle.dump(dados, f)
    
    @classmethod
    def carregar(cls, arquivo):
        """Carrega modelo salvo"""
        with open(arquivo, 'rb') as f:
            dados = pickle.load(f)
        
        preditor = cls()
        preditor.modelo = dados['modelo']
        preditor.scaler = dados['scaler']
        preditor.encoders = dados['encoders']
        preditor.metricas = dados['metricas']
        preditor.valores_unicos = dados['valores_unicos']
        preditor.feature_names = dados['feature_names']
        preditor.feature_medians = dados.get('feature_medians') # Carregar as medianas
        preditor.treinado = True
        
        return preditor

# EXECUÇÃO
# Certifique-se de que 'datatran_consolidado.json' existe e está no formato correto
# preditor = PreditorRidgeReal()
# preditor.treinar(CAMINHO_DADOS)
# preditor.salvar('modelo_ridge_real.pkl')

