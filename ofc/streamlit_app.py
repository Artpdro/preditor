import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
def load_model():
    try:
        model = joblib.load('random_forest_model.pkl')
        le_condicao = joblib.load('le_condicao.pkl')
        le_dia_semana = joblib.load('le_dia_semana.pkl')
        return model, le_condicao, le_dia_semana
    except FileNotFoundError:
        st.error("Modelo não encontrado. Execute o script de treinamento primeiro.")
        return None, None, None

model, le_condicao, le_dia_semana = load_model()

if model is not None:
    st.success("Modelo carregado com sucesso!")
    
    # Informações do modelo
    st.markdown("### 📊 Informações do Modelo")
    st.info("""
    **Acurácia (R-squared):** 0.0317  
    **Desvio Padrão:** 0.0270  
    **Algoritmo:** Random Forest Regressor  
    **Hiperparâmetros:** max_depth=10, min_samples_leaf=4, n_estimators=150
    """)
    
    st.markdown("---")
    st.markdown("### 🔮 Fazer Previsão")
    
    # Criar formulário para entrada de dados
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Campo para horário
            horario_input = st.time_input(
                "⏰ Horário do Acidente",
                value=None,
                help="Selecione o horário em que deseja prever acidentes"
            )
        
        with col2:
            # Campo para condição meteorológica
            condicoes_disponiveis = le_condicao.classes_
            condicao_input = st.selectbox(
                "🌤️ Condição Meteorológica",
                options=condicoes_disponiveis,
                help="Selecione a condição meteorológica"
            )
        
        # Campo para dia da semana
        dias_disponiveis = le_dia_semana.classes_
        dia_semana_input = st.selectbox(
            "📅 Dia da Semana",
            options=dias_disponiveis,
            help="Selecione o dia da semana"
        )
        
        # Botão para fazer previsão
        submitted = st.form_submit_button("🔍 Fazer Previsão", use_container_width=True)
        
        if submitted:
            if horario_input is not None:
                # Converter horário para segundos
                horario_segundos = horario_input.hour * 3600 + horario_input.minute * 60 + horario_input.second
                
                # Codificar as variáveis categóricas
                condicao_encoded = le_condicao.transform([condicao_input])[0]
                dia_semana_encoded = le_dia_semana.transform([dia_semana_input])[0]
                
                # Criar array para previsão
                input_data = np.array([[horario_segundos, condicao_encoded, dia_semana_encoded]])
                
                # Fazer previsão
                predicao = model.predict(input_data)[0]
                
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
                
                # Interpretação do resultado
                if predicao < 1:
                    st.success("🟢 **Baixo risco** - Poucos acidentes esperados neste horário e condições.")
                elif predicao < 3:
                    st.warning("🟡 **Risco moderado** - Número moderado de acidentes esperados.")
                else:
                    st.error("🔴 **Alto risco** - Muitos acidentes esperados. Atenção redobrada!")
                
                # Informações adicionais
                st.markdown("---")
                st.markdown("### ℹ️ Informações Adicionais")
                st.info(f"""
                **Condição Meteorológica:** {condicao_input}  
                **Horário em segundos:** {horario_segundos}  
                **Condição codificada:** {condicao_encoded}  
                **Dia da semana codificado:** {dia_semana_encoded}
                """)
            else:
                st.error("Por favor, selecione um horário válido.")

else:
    st.error("Não foi possível carregar o modelo. Verifique se os arquivos do modelo estão presentes.")

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido com Streamlit** | Modelo Random Forest para previsão de acidentes de trânsito")
