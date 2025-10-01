import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json

# Configurar a página
st.set_page_config(
    page_title="Previsão de Acidentes de Trânsito",
    page_icon="🚗",
    layout="centered"
)

# Título da aplicação
st.title("🚗 Previsão de Acidentes de Trânsito")
st.markdown("---")

# Carregar o modelo e os encoders
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load("random_forest_model.pkl")
        le_condicao = joblib.load("le_condicao.pkl")
        le_dia_semana = joblib.load("le_dia_semana.pkl")
        le_uf = joblib.load("le_uf.pkl")
        le_municipio = joblib.load("le_municipio.pkl")

        # Carregar dados brutos para mapeamento UF-Município
        json_path = '/home/ubuntu/upload/datatran_consolidado.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        df_raw = pd.DataFrame(raw_data)
        df_raw.rename(columns={'uf': 'UF', 'municipio': 'Municipio'}, inplace=True)
        
        # Criar mapeamento UF -> Municípios
        uf_municipio_map = {}
        for uf in df_raw['UF'].unique():
            municipios_uf = df_raw[df_raw['UF'] == uf]['Municipio'].unique()
            # Filtrar apenas municípios que estão no encoder
            municipios_validos = [m for m in municipios_uf if m in le_municipio.classes_]
            uf_municipio_map[uf] = sorted(municipios_validos)

        return model, le_condicao, le_dia_semana, le_uf, le_municipio, uf_municipio_map
    except FileNotFoundError as e:
        st.error(f"Modelo, encoders ou arquivo de dados não encontrados: {e}")
        return None, None, None, None, None, None

model, le_condicao, le_dia_semana, le_uf, le_municipio, uf_municipio_map = load_model_and_data()

if model is not None:
    st.success("Modelo, encoders e dados carregados com sucesso!")
    
    # Informações do modelo atualizadas
    st.markdown("### 📊 Informações do Modelo")
    st.info("""
    **Acurácia (R-squared):** 0.0412  
    **RMSE:** 0.1197  
    **Desvio Padrão dos Resíduos:** 0.1197  
    **Algoritmo:** Random Forest Regressor  
    **Hiperparâmetros Otimizados:** max_depth=10, min_samples_leaf=10, n_estimators=50  
    **Amostra de dados:** 5% do dataset completo (otimizada via GridSearchCV)  
    **Preditores:** Horário, Condição Meteorológica, Dia da Semana, UF, Município
    """)
    
    st.markdown("---")
    st.markdown("### 🔮 Fazer Previsão")
    
    # Criar campos de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        horario_input = st.time_input(
            "⏰ Horário do Acidente",
            value=None,
            help="Selecione o horário em que deseja prever acidentes"
        )
    
    with col2:
        condicoes_disponiveis = le_condicao.classes_
        condicao_input = st.selectbox(
            "🌤️ Condição Meteorológica",
            options=condicoes_disponiveis,
            help="Selecione a condição meteorológica"
        )
    
    dias_disponiveis = le_dia_semana.classes_
    dia_semana_input = st.selectbox(
        "📅 Dia da Semana",
        options=dias_disponiveis,
        help="Selecione o dia da semana"
    )

    uf_options = ["Todos"] + sorted([uf for uf in le_uf.classes_ if uf in uf_municipio_map and len(uf_municipio_map[uf]) > 0])

    uf_input = st.selectbox(
        "🗺️ UF (Estado)",
        options=uf_options,
        key="uf_selector",
        help="Selecione a Unidade Federativa ou 'Todos' para uma previsão geral"
    )

    municipio_input = None
    if uf_input == "Todos":
        st.selectbox(
            "🏙️ Município",
            options=["Não aplicável quando UF = 'Todos'"],
            disabled=True,
            key="municipio_selector",
            help="Campo desabilitado quando 'Todos' está selecionado para UF"
        )
    else:
        municipio_options = uf_municipio_map.get(uf_input, [])

        if municipio_options:
            # Garantir que o município selecionado anteriormente ainda é válido
            current_municipio_index = 0
            if 'municipio_selector' in st.session_state and st.session_state.municipio_selector in municipio_options:
                current_municipio_index = municipio_options.index(st.session_state.municipio_selector)

            municipio_input = st.selectbox(
                "🏙️ Município",
                options=municipio_options,
                index=current_municipio_index,
                key="municipio_selector",
                help="Selecione o Município"
            )
        else:
            st.selectbox(
                "🏙️ Município",
                options=["Nenhum município disponível"],
                disabled=True,
                key="municipio_selector",
                help="Nenhum município disponível para esta UF"
            )

    # Botão para fazer previsão
    submitted = st.button("🔍 Fazer Previsão", use_container_width=True)
    
    if submitted:
        if horario_input is not None:
            # Converter horário para segundos
            horario_segundos = horario_input.hour * 3600 + horario_input.minute * 60 + horario_input.second
            
            # Codificar as variáveis categóricas
            condicao_encoded = le_condicao.transform([condicao_input])[0]
            dia_semana_encoded = le_dia_semana.transform([dia_semana_input])[0]
            
            # Lidar com 'Todos' para UF
            if uf_input == "Todos":
                # Fazer previsões para todas as UFs e calcular a média
                predicoes_uf = []
                for uf in le_uf.classes_:
                    if uf in uf_municipio_map and len(uf_municipio_map[uf]) > 0:
                        uf_encoded = le_uf.transform([uf])[0]
                        # Usar o primeiro município da UF para a previsão
                        municipio_for_uf = uf_municipio_map[uf][0]
                        municipio_encoded = le_municipio.transform([municipio_for_uf])[0]
                        
                        input_data = np.array([[horario_segundos, condicao_encoded, dia_semana_encoded, uf_encoded, municipio_encoded]])
                        predicao_uf = model.predict(input_data)[0]
                        predicoes_uf.append(predicao_uf)
                
                predicao = np.mean(predicoes_uf)
                uf_display = "Todos os Estados"
                municipio_display = "(Média calculada)"
            else:
                if municipio_input not in ["Não aplicável quando UF = 'Todos'", "Nenhum município disponível"] and municipio_input is not None:
                    uf_encoded = le_uf.transform([uf_input])[0]
                    municipio_encoded = le_municipio.transform([municipio_input])[0]
                    
                    # Criar array para previsão
                    input_data = np.array([[horario_segundos, condicao_encoded, dia_semana_encoded, uf_encoded, municipio_encoded]])
                    
                    # Fazer previsão
                    predicao = model.predict(input_data)[0]
                    uf_display = uf_input
                    municipio_display = municipio_input
                else:
                    st.error("Por favor, selecione um município válido.")
                    st.stop()
            
            # Exibir resultado
            st.markdown("---")
            st.markdown("### 📈 Resultado da Previsão")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Quantidade Prevista",
                    value=f"{predicao:.2f}",
                    help="Número estimado de acidentes"
                )
            
            with col2:
                st.metric(
                    label="Horário",
                    value=horario_input.strftime("%H:%M:%S")
                )
            
            with col3:
                st.metric(
                    label="Dia da Semana",
                    value=dia_semana_input
                )
            
            # Informações adicionais
            st.markdown("---")
            st.markdown("### ℹ️ Informações da Previsão")
            st.info(f"""
            **Condição Meteorológica:** {condicao_input}  
            **Dia da Semana:** {dia_semana_input}  
            **UF:** {uf_display}  
            **Município:** {municipio_display}  
            **Horário:** {horario_input.strftime("%H:%M:%S")} ({horario_segundos} segundos)
            """)

            # Interpretação do resultado
            if predicao < 1:
                st.success("🟢 **Baixo risco** - Poucos acidentes esperados neste horário e condições.")
            elif predicao < 3:
                st.warning("🟡 **Risco moderado** - Número moderado de acidentes esperados.")
            else:
                st.error("🔴 **Alto risco** - Muitos acidentes esperados. Atenção redobrada!")
            
            # Mostrar detalhes técnicos em um expander
            with st.expander("🔧 Detalhes Técnicos"):
                if uf_input == "Todos":
                    st.write(f"**Método:** Média das previsões para {len(predicoes_uf)} UFs disponíveis")
                    st.write(f"**UFs consideradas:** {len(predicoes_uf)}")
                else:
                    st.write(f"**UF codificada:** {le_uf.transform([uf_input])[0]}")
                    if municipio_input:
                        st.write(f"**Município codificado:** {le_municipio.transform([municipio_input])[0]}")
                
                st.write(f"**Condição codificada:** {condicao_encoded}")
                st.write(f"**Dia da semana codificado:** {dia_semana_encoded}")
            
        else:
            st.error("Por favor, selecione um horário válido.")

else:
    st.error("Não foi possível carregar o modelo. Verifique se os arquivos do modelo estão presentes.")

# Informações sobre o modelo
st.markdown("---")
st.markdown("### 📋 Sobre o Modelo")
st.markdown("""
Este modelo de Machine Learning utiliza **Random Forest** para prever a quantidade de acidentes de trânsito 
com base em condições específicas. O modelo foi treinado com dados reais de acidentes e considera:

- **Horário do dia** (em segundos)
- **Condições meteorológicas** (Céu Claro, Nublado, Sol, etc.)
- **Dia da semana**
- **Unidade Federativa (UF)**
- **Município**

A opção **"Todos"** para UF calcula a média das previsões para todos os estados disponíveis no dataset.

**Nota sobre a acurácia:** O modelo apresenta uma acurácia baixa (R² = 0.0412), o que indica que a predição 
de acidentes é um problema complexo que pode requerer mais dados, features adicionais ou modelos mais sofisticados.
""")

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido com Streamlit** | Modelo Random Forest para previsão de acidentes de trânsito")


# Informações sobre o modelo
st.markdown("---")
st.markdown("### 📋 Sobre o Modelo")
st.markdown("""
Este modelo de Machine Learning utiliza **Random Forest** para prever a quantidade de acidentes de trânsito 
com base em condições específicas. O modelo foi treinado com dados reais de acidentes e considera:

- **Horário do dia** (em segundos)
- **Condições meteorológicas** (Céu Claro, Nublado, Sol, etc.)
- **Dia da semana**
- **Unidade Federativa (UF)**
- **Município**

A opção **"Todos"** para UF calcula a média das previsões para todos os estados disponíveis no dataset.

**Nota sobre a acurácia:** O modelo apresenta uma acurácia baixa (R² = 0.0412), o que indica que a predição 
de acidentes é um problema complexo que pode requerer mais dados, features adicionais ou modelos mais sofisticados.
""")

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido com Streamlit** | Modelo Random Forest para previsão de acidentes de trânsito")
