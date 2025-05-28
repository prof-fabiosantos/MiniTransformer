import streamlit as st
from ui.sidebar import mostrar_sidebar
from ui.layout import layout_principal
from core.inferencia import inicializar_modelo, gerar_resposta, visualizar_pesos

inicializar_modelo()
mostrar_sidebar()
layout_principal()

if st.session_state.get("gerar", False):
    gerar_resposta()

if st.session_state.get("resposta_gerada", False):
    if st.button("Visualizar pesos do modelo"):
        visualizar_pesos()
