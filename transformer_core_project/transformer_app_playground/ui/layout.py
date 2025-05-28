import streamlit as st

def layout_principal():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("assets/logo_transformercore.png", width=150)
    with col2:
        st.title("Playground")

    pergunta = st.text_input("Faça uma pergunta:", "qual a capital do brasil?")
    num_tokens = st.slider("Número de palavras a gerar", 1, 30, 10)
    temperature = st.slider("Temperatura", 0.1, 2.0, 1.0, step=0.1)
    top_k = st.slider("Top-k", 0, 50, 5)
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9)

    st.session_state.update({
        "pergunta": pergunta,
        "num_tokens": num_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "gerar": st.button("Gerar resposta")
    })
