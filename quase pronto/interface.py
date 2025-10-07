import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from preditorpeido import SimplifiedAccidentPredictor

# Carregar o modelo treinado e seus componentes
@st.cache_resource
def load_model():
    try:
        with open("accident_predictor.pkl", "rb") as f:
            data = pickle.load(f)
        
        # Reconstruir o objeto SimplifiedAccidentPredictor
        predictor = SimplifiedAccidentPredictor()
        predictor.modelo = data["model"]
        predictor.scaler = data["scaler"]
        predictor.encoders = data["encoders"]
        predictor.feature_names = data["feature_names"]
        predictor.feature_medians = data["feature_medians"]
        predictor.holidays_br = data["holidays_br"]
        predictor.treinado = True
        predictor.r2_score = data.get("r2_score", None)
        predictor.rmse_score = data.get("rmse_score", None)
        return predictor
    except FileNotFoundError:
        st.error("O arquivo do modelo 'accident_predictor.pkl' não foi encontrado. Por favor, treine o modelo primeiro.")
        return None

# Carregar as opções de UF, Município e Condição Climática
@st.cache_data
def load_options():
    with open("uf_options.json", "r", encoding="utf-8") as f:
        uf_options = json.load(f)
    
    with open("municipios_por_uf.json", "r", encoding="utf-8") as f:
        municipios_por_uf = json.load(f)
    
    with open("condicoes_metereologicas_options.json", "r", encoding="utf-8") as f:
        condicoes_metereologicas_options = json.load(f)
    
    return uf_options, municipios_por_uf, condicoes_metereologicas_options

predictor = load_model()
uf_options, municipios_por_uf, condicoes_metereologicas_options = load_options()

st.title("Preditor de Acidentes de Trânsito")
st.write("Preencha os campos abaixo para prever o número de acidentes.")

if predictor:
    # Entradas do usuário
    uf = st.selectbox("UF", uf_options)
    
    # Filtrar municípios com base na UF selecionada
    municipios_filtrados = municipios_por_uf.get(uf, ["DESCONHECIDO"])
    municipio = st.selectbox("Município", municipios_filtrados)
    
    horario = st.time_input("Horário", datetime.now().time()).strftime("%H:%M:%S")
    
    condicao_metereologica = st.selectbox("Condição Climática", condicoes_metereologicas_options)

    if st.button("Prever Acidentes"):
        try:
            # Usar a data atual para a predição
            data_inversa = datetime.now().strftime("%d/%m/%Y")
            
            # Usar o tipo de acidente mais comum como padrão
            tipo_acidente_default = list(predictor.encoders.get("tipo_acidente_principal", LabelEncoder()).classes_)[0] if "tipo_acidente_principal" in predictor.encoders else "DESCONHECIDO"
            
            predicao = predictor.prever_acidentes(
                data_inversa=data_inversa,
                horario=horario,
                uf=uf,
                municipio=municipio,
                tipo_acidente=tipo_acidente_default,
                condicao_metereologica=condicao_metereologica
            )
            st.success(f"A previsão de acidentes para as condições informadas é: **{predicao}**")
        except Exception as e:
            st.error(f"Ocorreu um erro ao fazer a predição: {e}")
            st.warning("Certifique-se de que o modelo foi treinado e que os dados de entrada são válidos.")

    st.sidebar.header("Métricas do Modelo")
    st.sidebar.write(f"**R² Score:** {predictor.r2_score:.4f}" if predictor.r2_score is not None else "**R² Score:** N/A")
    st.sidebar.write(f"**RMSE:** {predictor.rmse_score:.2f}" if predictor.rmse_score is not None else "**RMSE:** N/A")
else:
    st.warning("O modelo não pôde ser carregado. Por favor, verifique se o arquivo 'accident_predictor.pkl' existe e se o modelo foi treinado corretamente.")
