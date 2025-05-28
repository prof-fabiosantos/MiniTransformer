# Playground
# Importa a biblioteca Streamlit para criar interfaces web
import streamlit as st

# Importa o PyTorch, uma biblioteca para aprendizado profundo
import torch

# Importa funções de ativação como softmax do PyTorch
import torch.nn.functional as F

# Importa o módulo json para ler o vocabulário salvo em arquivo
import json

# Importa o modelo Transformer definido no arquivo model.py
from model import SimpleTransformer

# Importa NumPy para manipulação de arrays
import numpy as np

# Importa a biblioteca PIL para lidar com imagens
from PIL import Image

# Define a função para aplicar filtragem top-k e top-p nos logits
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    # Limita top_k ao tamanho do vetor de logits
    top_k = min(top_k, logits.size(-1))
    
    # Aplica top-k: mantém apenas os top_k maiores logits
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    # Aplica top-p (nucleus sampling): mantém apenas os logits cujo somatório acumulado de probabilidade <= top_p
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')
    
    return logits  # Retorna os logits filtrados

# Divide a interface em duas colunas com proporção 1:5
col1, col2 = st.columns([1, 5])

# Coloca a logo na primeira coluna
with col1:
    st.image("assets/logo_transformercore.png", width=150)

# Coloca o título "Playground" na segunda coluna
with col2:
    st.title("Playground")

# Carrega o vocabulário do modelo salvo em JSON
with open("vocab_transformer.json", "r", encoding="utf-8") as vf:
    token_to_id = json.load(vf)
    id_to_token = {int(i): t for t, i in token_to_id.items()}  # Inverte o dicionário

# Calcula o tamanho do vocabulário
vocab_size = len(token_to_id)

# Instancia o modelo Transformer com os parâmetros definidos
model = SimpleTransformer(vocab_size=vocab_size, embed_dim=128, num_heads=8, num_layers=4)

# Define o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega os pesos do modelo treinado
model.load_state_dict(torch.load("modelo_transformer.pt", map_location=device))

# Move o modelo para o dispositivo
model.to(device)

# Coloca o modelo em modo de avaliação (desativa dropout etc.)
model.eval()

# Mostra o número total de parâmetros do modelo na barra lateral
total_params = sum(p.numel() for p in model.parameters())
st.sidebar.markdown(f"**Parâmetros do modelo:** {total_params:,}")

# Cria a interface para entrada do usuário
pergunta = st.text_input("Faça uma pergunta:", "qual a capital do brasil?")

# Slider para escolher o número de palavras a gerar
num_tokens = st.slider("Número de palavras a gerar", 1, 30, 10)

# Slider para escolher a temperatura (diversidade)
temperature = st.slider("Temperatura", 0.1, 2.0, 1.0, step=0.1)

# Slider para definir o top-k
top_k = st.slider("Top-k", 0, 50, 5)

# Slider para definir o top-p
top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9)

# Quando o botão "Gerar resposta" é clicado
if st.button("Gerar resposta"):
    # Prepara o prompt com a pergunta
    prompt = f"pergunta: {pergunta.strip().lower()} resposta:"
    tokens = prompt.split()  # Tokeniza a entrada
    
    # Verifica se há tokens fora do vocabulário
    if any(t not in token_to_id for t in tokens):
        desconhecidas = [t for t in tokens if t not in token_to_id]
        st.error(f"Palavras fora do vocabulário: {', '.join(desconhecidas)}")
    else:
        # Converte tokens para IDs e move para o dispositivo
        input_ids = torch.tensor([[token_to_id[t] for t in tokens]]).to(device)
        generated = input_ids
        
        # Gera novos tokens com base no modelo
        for _ in range(num_tokens):
            logits, _ = model(generated)  # Obtém os logits do modelo
            logits = logits[:, -1, :] / temperature  # Aplica temperatura apenas ao último token
            filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)  # Converte logits em probabilidades
            next_token = torch.multinomial(probs, num_samples=1)  # Amostra o próximo token
            generated = torch.cat([generated, next_token], dim=1)  # Anexa o novo token
        
        # Converte IDs de volta para texto
        saida = " ".join([id_to_token[i] for i in generated[0].tolist()])
        
        # Exibe a resposta gerada na interface
        st.write("### Resposta gerada:")
        st.success(saida)
