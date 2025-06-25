#© 2025 Prof. Dr. Fabio Santos. Uso permitido apenas para fins educacionais e não comerciais.
# transformer_chatbot_app.py
import streamlit as st
import torch
import torch.nn.functional as F
import json
from model import SimpleTransformer
import numpy as np
from PIL import Image

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')
    return logits

col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/logo_transformercore.png", width=150)
with col2:
    st.title("Playground")


# Carrega vocabulário e modelo
with open("vocab_transformer.json", "r", encoding="utf-8") as vf:
    token_to_id = json.load(vf)
    id_to_token = {int(i): t for t, i in token_to_id.items()}

vocab_size = len(token_to_id)
model = SimpleTransformer(vocab_size=vocab_size, embed_dim=128, num_heads=8, num_layers=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("modelo_transformer.pt", map_location=device))
model.to(device)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
st.sidebar.markdown(f"**Parâmetros do modelo:** {total_params:,}")


# Interface
pergunta = st.text_input("Faça uma pergunta:", "qual a capital do brasil?")
num_tokens = st.slider("Número de palavras a gerar", 1, 30, 10)
temperature = st.slider("Temperatura", 0.1, 2.0, 1.0, step=0.1)
top_k = st.slider("Top-k", 0, 50, 5)
top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9)

if st.button("Gerar resposta"):
    prompt = f"pergunta: {pergunta.strip().lower()} resposta:"
    tokens = prompt.split()
    if any(t not in token_to_id for t in tokens):
        desconhecidas = [t for t in tokens if t not in token_to_id]
        st.error(f"Palavras fora do vocabulário: {', '.join(desconhecidas)}")
    else:
        input_ids = torch.tensor([[token_to_id[t] for t in tokens]]).to(device)
        generated = input_ids
        for _ in range(num_tokens):
            logits, _ = model(generated)
            logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        saida = " ".join([id_to_token[i] for i in generated[0].tolist()])
        st.write("### Resposta gerada:")
        st.success(saida)
