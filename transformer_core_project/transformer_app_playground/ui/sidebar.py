import streamlit as st

def mostrar_sidebar():
    total_params = st.session_state.get("total_params", None)
    if total_params:
        st.sidebar.markdown(f"**Parâmetros do modelo:** {total_params:,}")
