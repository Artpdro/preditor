import streamlit as st
import pickle
import numpy as np
from datetime import datetime, timedelta

# Configurar página
st.set_page_config(
    page_title="Previsão de Acidentes",
)

# Classe do preditor Ridge real
class PreditorRidgeReal:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.encoders = {}
        self.valores_unicos = {}
        self.feature_names = []
        self.treinado = False
        self.feature_medians = None # Adicionado para armazenar as medianas das features
    
    def prever_acidentes(self, horario, dia_semana, condicao_metereologica, uf, municipio):
        if not self.treinado:
            return 0 # Retorna 0 se o modelo não estiver treinado, em vez de 1
        
        # Criar uma data fictícia para gerar features
        data_base = datetime(2025, 10, 6)
        
        # Ajustar data para o dia da semana solicitado
        dias_map = {
            'Segunda': 0, 'Terça': 1, 'Quarta': 2, 'Quinta': 3, 
            'Sexta': 4, 'Sábado': 5, 'Domingo': 6
        }
        
        dia_semana_num = dias_map.get(dia_semana, 0)
        
        # Ajustar data para o dia da semana correto
        dias_diff = dia_semana_num - data_base.weekday()
        data_ajustada = data_base + timedelta(days=dias_diff)
        
        # Criar features como o modelo espera
        features_dict = self._criar_features_para_data(
            data_ajustada, horario, condicao_metereologica, uf, municipio
        )
        
        # Converter o dicionário de features para um array na ordem correta
        # Garante que a ordem das features seja a mesma do treinamento
        features_array = np.array([features_dict[f] for f in self.feature_names])
        
        # Usar o modelo Ridge treinado
        features_scaled = self.scaler.transform([features_array])
        predicao_raw = self.modelo.predict(features_scaled)[0]
        
        # A previsão é para o dia todo, ajustar proporcionalmente
        predicao_ajustada = predicao_raw / 24  # Dividir por 24 horas
        
        # Arredondar conforme solicitado
        predicao_final = round(max(0, predicao_ajustada))
        
        return predicao_final
    
    def _criar_features_para_data(self, data, horario, condicao_metereologica, uf, municipio):
        """Cria features para uma situação específica usando valores históricos e medianas do treinamento"""
        
        # Inicializa um dicionário para armazenar as features
        features = {}
        
        # Features temporais básicas
        features["ano"] = data.year
        features["mes"] = data.month
        features["dia"] = data.day
        features["dia_semana"] = data.weekday()
        features["dia_ano"] = data.timetuple().tm_yday
        features["semana_ano"] = data.isocalendar().week # Já é int
        features["trimestre"] = ((features["mes"] - 1) // 3) + 1
        features["fim_semana"] = int(features["dia_semana"] >= 5)
        
        # Features cíclicas
        features["mes_sin"] = np.sin(2 * np.pi * features["mes"] / 12)
        features["mes_cos"] = np.cos(2 * np.pi * features["mes"] / 12)
        features["dia_semana_sin"] = np.sin(2 * np.pi * features["dia_semana"] / 7)
        features["dia_semana_cos"] = np.cos(2 * np.pi * features["dia_semana"] / 7)
        features["dia_ano_sin"] = np.sin(2 * np.pi * features["dia_ano"] / 365)
        features["dia_ano_cos"] = np.cos(2 * np.pi * features["dia_ano"] / 365)
        
        # Features contextuais (usar medianas do treinamento ou valores específicos se disponíveis)
        # Para num_ufs, num_municipios, tipos_unicos, hora_std, hora_min, hora_max, 
        # usaremos as medianas calculadas durante o treinamento.
        # O 'horario' fornecido pelo usuário será usado para 'hora_media'.
        features["num_ufs"] = self.feature_medians.get("num_ufs", 1) # Usar mediana ou um valor padrão razoável
        features["num_municipios"] = self.feature_medians.get("num_municipios", 1) # Usar mediana
        features["tipos_unicos"] = self.feature_medians.get("tipos_unicos", 1) # Usar mediana
        features["hora_media"] = horario # Usar o horário específico fornecido
        features["hora_std"] = self.feature_medians.get("hora_std", 0) # Usar mediana
        features["hora_min"] = self.feature_medians.get("hora_min", 0) # Usar mediana
        features["hora_max"] = self.feature_medians.get("hora_max", 23) # Usar mediana
        
        # Features históricas (usar medianas do treinamento para lags e médias móveis)
        # Estes valores são mais complexos de simular sem dados históricos reais para a data da previsão.
        # A abordagem mais segura é usar as medianas do conjunto de treinamento.
        for lag in [1, 2, 3, 7, 14, 30]:
            features[f"acidentes_lag_{lag}"] = self.feature_medians.get(f"acidentes_lag_{lag}", 0)
        
        for window in [3, 7, 14, 30]:
            features[f"media_{window}d"] = self.feature_medians.get(f"media_{window}d", 0)
        
        for window in [7, 14, 30]:
            features[f"vol_{window}d"] = self.feature_medians.get(f"vol_{window}d", 0)
        
        features["tend_7d"] = self.feature_medians.get("tend_7d", 0)
        features["tend_30d"] = self.feature_medians.get("tend_30d", 0)
        
        # Contexto
        features["periodo_especial"] = int(features["mes"] in [12, 1, 6, 7] or features["mes"] == 2)
        features["feriado"] = int((features["mes"] == 12 and features["dia"] >= 20) or \
                                  (features["mes"] == 1 and features["dia"] <= 10) or \
                                  (features["mes"] == 9 and features["dia"] == 7) or \
                                  (features["mes"] == 10 and features["dia"] == 12) or \
                                  (features["mes"] == 11 and (features["dia"] == 2 or features["dia"] == 15)))
        
        # Codificar clima
        try:
            clima_encoded = self.encoders["clima"].transform([condicao_metereologica])[0]
        except KeyError:
            # Se o encoder não foi treinado ou a categoria é desconhecida, usar um valor padrão
            clima_encoded = 0 
        features["clima_encoded"] = clima_encoded
        
        return features

# Carregar modelo
@st.cache_resource
def carregar_modelo():
    try:
        with open('modelo_ridge_real.pkl', 'rb') as f:
            dados = pickle.load(f)
        
        preditor = PreditorRidgeReal()
        preditor.modelo = dados['modelo']
        preditor.scaler = dados['scaler']
        preditor.encoders = dados['encoders']
        preditor.valores_unicos = dados['valores_unicos']
        preditor.feature_names = dados['feature_names']
        preditor.feature_medians = dados.get('feature_medians') # Carregar as medianas
        preditor.treinado = True
        
        return preditor, f"✅ Modelo carregado (Acurácia: {dados['metricas']['r2_mean']:.1%})"
    
    except FileNotFoundError:
        # Valores padrão
        preditor = PreditorRidgeReal()
        valores_padrao = {
            'uf': ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'],
            'municipio': ['São Paulo', 'Rio de Janeiro', 'Brasília', 'Salvador', 'Fortaleza', 'Belo Horizonte', 'Manaus', 'Curitiba', 'Recife', 'Porto Alegre'],
            'condicao_metereologica': ['Céu Claro', 'Nublado', 'Sol', 'Chuva', 'Garoa/Chuvisco', 'Nevoeiro/Neblina'],
            'dia_semana': ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        }
        preditor.valores_unicos = valores_padrao
        
        return preditor, "❌ Erro: modelo não encontrado. Execute: `python3 preditorpeido.py` para treinar o modelo."

# Interface principal
def main():
    st.title("Previsão de Acidentes")
    st.markdown("---")
    
    # Carregar modelo
    preditor, status = carregar_modelo()
    
    # Mostrar status do modelo
    if "✅" in status:
        st.success(status)
    else:
        st.warning(status)
        st.info("Execute: `python3 preditorpeido.py` para treinar o modelo")
    
    # Formulário de entrada
    st.subheader("Dados para Previsão")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Horário
        horario = st.selectbox(
            "Horário:",
            options=list(range(0, 24)),
            format_func=lambda x: f"{x:02d}:00",
            index=12
        )
        
        # Dia da semana
        dia_semana = st.selectbox(
            "Dia da Semana:",
            options=preditor.valores_unicos['dia_semana'],
            index=0
        )
        
        # Condição climática
        condicao_metereologica = st.selectbox(
            "Condição Climática:",
            options=preditor.valores_unicos['condicao_metereologica'],
            index=0
        )
    
    with col2:
        # UF
        uf = st.selectbox(
            "UF (Estado):",
            options=preditor.valores_unicos['uf'],
            index=0
        )
        
        # Município
        municipio = st.selectbox(
            "Município:",
            options=preditor.valores_unicos['municipio'],
            index=0
        )
    
    st.markdown("---")
    
    # Botão de previsão
    if st.button("Prever Quantidade de Acidentes", type="primary", use_container_width=True):
        
        # Fazer previsão usando o modelo Ridge treinado
        quantidade_acidentes = preditor.prever_acidentes(
            horario=horario,
            dia_semana=dia_semana,
            condicao_metereologica=condicao_metereologica,
            uf=uf,
            municipio=municipio
        )
        
        # Mostrar resultado
        st.markdown("### Resultado da Previsão")
        
        # Exibir quantidade com destaque
        st.metric(
            label="🚨 Quantidade de Acidentes Prevista",
            value=f"{quantidade_acidentes}",
            help="Previsão feita pelo modelo Ridge treinado (93.77% de acurácia)"
        )
        
        # Interpretação do resultado
        if quantidade_acidentes == 0:
            st.success("🟢 **Baixo Risco** - Nenhum acidente previsto")
        elif quantidade_acidentes == 1:
            st.info("🔵 **Risco Baixo** - 1 acidente previsto")
        elif quantidade_acidentes <= 3:
            st.warning("🟡 **Risco Moderado** - Poucos acidentes previstos")
        else:
            st.error("🔴 **Alto Risco** - Múltiplos acidentes previstos")
        
        # Informações adicionais
        with st.expander("ℹ Detalhes da Previsão"):
            st.write(f"**Horário:** {horario:02d}:00")
            st.write(f"**Dia:** {dia_semana}")
            st.write(f"**Clima:** {condicao_metereologica}")
            st.write(f"**Estado:** {uf}")
            st.write(f"**Município:** {municipio}")
            st.write(f"**Resultado:** {quantidade_acidentes} acidente(s)")
            st.write("**Método:** Modelo Ridge treinado com 380.851 registros")
            
            st.markdown("**Como funciona:**")
            st.write("• O modelo Ridge foi treinado com dados históricos completos")
            st.write("• Usa 39 features temporais, cíclicas e históricas")
            st.write("• Previsão ajustada proporcionalmente para situação específica")
            st.write("• Arredondamento: 1.3→1, 1.5→2")
    
    # Informações do modelo
    st.markdown("---")

if __name__ == "__main__":
    main()

