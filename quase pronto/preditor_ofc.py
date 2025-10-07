
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import warnings
import holidays

warnings.filterwarnings("ignore")

class SimplifiedAccidentPredictor:
    
    def __init__(self, alpha=1.0):
        self.modelo = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.treinado = False
        self.feature_names = []
        self.feature_medians = None
        self.holidays_br = holidays.Brazil()
        self.r2_score = None
        self.rmse_score = None
        
    def _processar_dados_para_treino(self, df):
        """Processa os dados brutos para o formato de treinamento, agregando por dia."""
        df["data"] = pd.to_datetime(df["data_inversa"], format="%d/%m/%Y")
        df["horario_dt"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce").dt.time
        df["hora"] = df["horario_dt"].apply(lambda x: x.hour if pd.notna(x) else 12)

        # Preencher valores ausentes para evitar erros na agregação
        df["uf"] = df["uf"].fillna("DESCONHECIDO")
        df["municipio"] = df["municipio"].fillna("DESCONHECIDO")
        df["tipo_acidente"] = df["tipo_acidente"].fillna("DESCONHECIDO")
        df["condicao_metereologica"] = df["condicao_metereologica"].fillna("DESCONHECIDO")

        # Agregação diária
        agg_diario = df.groupby("data").agg(
            acidentes=("data_inversa", "count"), # Contagem de acidentes por dia
            uf=("uf", lambda x: x.mode()[0] if not x.empty else "DESCONHECIDO"),
            municipio=("municipio", lambda x: x.mode()[0] if not x.empty else "DESCONHECIDO"),
            tipo_acidente_principal=("tipo_acidente", lambda x: x.mode()[0] if not x.empty else "DESCONHECIDO"),
            condicao_metereologica_principal=("condicao_metereologica", lambda x: x.mode()[0] if not x.empty else "DESCONHECIDO"),
            hora_media=("hora", "mean"),
            hora_std=("hora", "std"),
            hora_min=("hora", "min"),
            hora_max=("hora", "max")
        ).reset_index()
        
        agg_diario["hora_std"] = agg_diario["hora_std"].fillna(0)
        return agg_diario

    def _criar_features(self, df_processed):
        """Cria features temporais e de contexto para o modelo."""
        df = df_processed.sort_values("data").reset_index(drop=True)
        
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.month
        df["dia"] = df["data"].dt.day
        df["dia_semana"] = df["data"].dt.dayofweek
        df["dia_ano"] = df["data"].dt.dayofyear
        df["semana_ano"] = df["data"].dt.isocalendar().week.astype(int)
        df["trimestre"] = df["data"].dt.quarter
        df["fim_semana"] = (df["dia_semana"] >= 5).astype(int)
        
        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
        df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
        df["dia_ano_sin"] = np.sin(2 * np.pi * df["dia_ano"] / 365)
        df["dia_ano_cos"] = np.cos(2 * np.pi * df["dia_ano"] / 365)
        
        df["feriado"] = df["data"].apply(lambda x: int(x in self.holidays_br))
        df["periodo_especial"] = df["data"].apply(
            lambda x: int(x.month in [12, 1, 6, 7] or x.month == 2)
        )

        # Features de Lag
        for lag in [1, 7, 14]:
            df[f"acidentes_lag_{lag}"] = df["acidentes"].shift(lag)
        
        # Features de Média Móvel
        for window in [7, 14]:
            df[f"media_{window}d"] = df["acidentes"].shift(1).rolling(window, min_periods=1).mean()
        
        # Codificação de variáveis categóricas
        for col in ["uf", "municipio", "tipo_acidente_principal", "condicao_metereologica_principal"]:
            encoder = LabelEncoder()
            df[f"{col}_encoded"] = encoder.fit_transform(df[col])
            self.encoders[col] = encoder
        
        # Features numéricas a serem usadas no modelo
        features_numericas = [
            "ano", "mes", "dia", "dia_semana", "dia_ano", "semana_ano", "trimestre", "fim_semana",
            "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos", "dia_ano_sin", "dia_ano_cos",
            "hora_media", "hora_std", "hora_min", "hora_max",
            "feriado", "periodo_especial",
            "acidentes_lag_1", "acidentes_lag_7", "acidentes_lag_14",
            "media_7d", "media_14d",
            "uf_encoded", "municipio_encoded", "tipo_acidente_principal_encoded", "condicao_metereologica_principal_encoded"
        ]
        
        X = df[features_numericas]
        y = df["acidentes"]
        
        X = X.replace([np.inf, -np.inf], np.nan)
        self.feature_medians = X.median().to_dict() # Armazenar medianas para preenchimento futuro
        X = X.fillna(X.median())
        
        return X, y
    
    def _validar_modelo(self, X, y, alpha_val=1.0):
        """Validação cruzada do modelo usando TimeSeriesSplit para calcular métricas."""
        tscv = TimeSeriesSplit(n_splits=5)
        
        scores_r2 = []
        scores_rmse = []
        
        if len(X) < tscv.n_splits + 1:
            print(f"Aviso: Não há amostras suficientes para a validação cruzada. Necessário pelo menos {tscv.n_splits + 1} amostras, mas encontrou {len(X)}. Pulando validação.")
            return {"r2_mean": np.nan, "rmse_mean": np.nan}

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler_temp = StandardScaler()
            X_train_scaled = scaler_temp.fit_transform(X_train)
            X_val_scaled = scaler_temp.transform(X_val)
            
            modelo_temp = Ridge(alpha=alpha_val)
            modelo_temp.fit(X_train_scaled, y_train)
            
            y_pred = modelo_temp.predict(X_val_scaled)
            
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            scores_r2.append(r2)
            scores_rmse.append(rmse)
        
        return {
            "r2_mean": np.mean(scores_r2),
            "rmse_mean": np.mean(scores_rmse)
        }

    def treinar(self, arquivo_json, alpha_val=1.0):
        """Treina o modelo de predição de acidentes e calcula métricas."""
        with open(arquivo_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df_processed = self._processar_dados_para_treino(df.copy())
        
        X, y = self._criar_features(df_processed)
        
        if X.empty:
            raise ValueError("O DataFrame de features (X) está vazio após a criação de features. Não é possível treinar o modelo.")

        self.feature_names = X.columns.tolist()
        
        # Validação do modelo para obter métricas
        metrics = self._validar_modelo(X, y, alpha_val)
        self.r2_score = metrics["r2_mean"]
        self.rmse_score = metrics["rmse_mean"]

        X_scaled = self.scaler.fit_transform(X)
        self.modelo = Ridge(alpha=alpha_val) # Re-instancia o modelo com o alpha correto
        self.modelo.fit(X_scaled, y)
        
        self.treinado = True
        print("Modelo treinado com sucesso.")
        print(f"Acurácia (R²): {self.r2_score:.4f}")
        print(f"Desvio Padrão (RMSE): {self.rmse_score:.2f}")
        return self
    
    def prever_acidentes(self, data_inversa, horario, uf, municipio, tipo_acidente, condicao_metereologica):
        """Faz a predição da quantidade de acidentes para uma data e condições específicas."""
        if not self.treinado:
            raise RuntimeError("O modelo não foi treinado. Chame o método \'treinar\' primeiro.")
        
        data = datetime.strptime(data_inversa, "%d/%m/%Y")
        hora = int(horario.split(":")[0]) # Extrai apenas a hora

        # Criar um DataFrame com os dados de entrada para a previsão
        input_df = pd.DataFrame({
            "data": [data],
            "ano": [data.year],
            "mes": [data.month],
            "dia": [data.day],
            "dia_semana": [data.weekday()],
            "dia_ano": [data.timetuple().tm_yday],
            "semana_ano": [data.isocalendar().week],
            "trimestre": [((data.month - 1) // 3) + 1],
            "fim_semana": [int(data.weekday() >= 5)],
            "mes_sin": [np.sin(2 * np.pi * data.month / 12)],
            "mes_cos": [np.cos(2 * np.pi * data.month / 12)],
            "dia_semana_sin": [np.sin(2 * np.pi * data.weekday() / 7)],
            "dia_semana_cos": [np.cos(2 * np.pi * data.weekday() / 7)],
            "dia_ano_sin": [np.sin(2 * np.pi * data.timetuple().tm_yday / 365)],
            "dia_ano_cos": [np.cos(2 * np.pi * data.timetuple().tm_yday / 365)],
            "feriado": [int(data in self.holidays_br)],
            "periodo_especial": [int(data.month in [12, 1, 6, 7] or data.month == 2)],
            "hora_media": [hora],
            "hora_std": [self.feature_medians.get("hora_std", 0)], # Usar mediana geral
            "hora_min": [self.feature_medians.get("hora_min", 0)], # Usar mediana geral
            "hora_max": [self.feature_medians.get("hora_max", 0)], # Usar mediana geral
            "acidentes_lag_1": [self.feature_medians.get("acidentes_lag_1", 0)], # Usar mediana geral
            "acidentes_lag_7": [self.feature_medians.get("acidentes_lag_7", 0)], # Usar mediana geral
            "acidentes_lag_14": [self.feature_medians.get("acidentes_lag_14", 0)], # Usar mediana geral
            "media_7d": [self.feature_medians.get("media_7d", 0)], # Usar mediana geral
            "media_14d": [self.feature_medians.get("media_14d", 0)], # Usar mediana geral
        })

        # Codificar variáveis categóricas para a previsão
        for col_name_key, encoder_obj in self.encoders.items():
            # Mapear o nome da coluna do encoder para o parâmetro de entrada correspondente
            if col_name_key == "uf":
                val = uf
            elif col_name_key == "municipio":
                val = municipio
            elif col_name_key == "tipo_acidente_principal":
                val = tipo_acidente
            elif col_name_key == "condicao_metereologica_principal":
                val = condicao_metereologica
            else:
                val = "DESCONHECIDO" # Fallback

            try:
                input_df[f"{col_name_key}_encoded"] = encoder_obj.transform([val])
            except ValueError:
                # Se o valor não foi visto durante o treino, usa 0 ou um valor padrão
                input_df[f"{col_name_key}_encoded"] = 0 
        
        # Garantir que todas as features estejam presentes e na ordem correta
        X_predict = pd.DataFrame(columns=self.feature_names)
        for col in self.feature_names:
            if col in input_df.columns:
                X_predict[col] = input_df[col]
            else:
                X_predict[col] = self.feature_medians.get(col, 0) # Preencher com mediana se a feature não foi gerada

        X_predict_scaled = self.scaler.transform(X_predict)
        prediction = self.modelo.predict(X_predict_scaled)
        
        # A predição não pode ser negativa
        return max(0, round(prediction[0]))

# Exemplo de uso (para teste interno, pode ser removido na entrega final)
if __name__ == "__main__":
    predictor = SimplifiedAccidentPredictor(alpha=1.0) # Ajustando o alpha para 0.5
    predictor.treinar("datatran_consolidado.json", alpha_val=0.5)
    
    # Salvar o modelo treinado e seus componentes
    with open("accident_predictor.pkl", "wb") as f:
        pickle.dump({
            "model": predictor.modelo,
            "scaler": predictor.scaler,
            "encoders": predictor.encoders,
            "feature_names": predictor.feature_names,
            "feature_medians": predictor.feature_medians,
            "holidays_br": predictor.holidays_br,
            "r2_score": predictor.r2_score,
            "rmse_score": predictor.rmse_score
        }, f)

    print("Modelo salvo em accident_predictor.pkl")



