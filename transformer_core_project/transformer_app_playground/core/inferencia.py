
import torch
import torch.nn.functional as F
import json
from model import SimpleTransformer
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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

def inicializar_modelo():
    with open("vocab_transformer.json", "r", encoding="utf-8") as vf:
        token_to_id = json.load(vf)
        id_to_token = {int(i): t for t, i in token_to_id.items()}

    vocab_size = len(token_to_id)
    model = SimpleTransformer(vocab_size=vocab_size, embed_dim=128, num_heads=8, num_layers=4, return_attention=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("modelo_transformer.pt", map_location=device))
    model.to(device)
    model.eval()

    st.session_state.update({
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "model": model,
        "device": device,
        "total_params": sum(p.numel() for p in model.parameters())
    })

def gerar_resposta():
    pergunta = st.session_state["pergunta"].strip().lower()
    tokens = pergunta.split()
    token_to_id = st.session_state["token_to_id"]

    if any(t not in token_to_id for t in tokens):
        desconhecidas = [t for t in tokens if t not in token_to_id]
        st.error(f"Palavras fora do vocabulário: {', '.join(desconhecidas)}")
        return

    input_ids = torch.tensor([[token_to_id[t] for t in tokens]]).to(st.session_state["device"])
    generated = input_ids
    model = st.session_state["model"]
    attentions_all = []

    for _ in range(st.session_state["num_tokens"]):
        logits, attn_weights = model(generated)
        logits = logits[:, -1, :] / st.session_state["temperature"]
        filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=st.session_state["top_k"], top_p=st.session_state["top_p"])
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        attentions_all.append(attn_weights)

    id_to_token = st.session_state["id_to_token"]
    st.session_state["resposta_gerada"] = True
    st.session_state["tokens"] = [id_to_token[i] for i in generated[0].tolist()]
    st.session_state["atencao"] = attentions_all
    st.session_state["embeddings"] = model.embed.weight.detach().cpu().numpy()

    saida = " ".join(st.session_state["tokens"])
    st.write("### Resposta gerada:")
    st.success(saida)

def visualizar_pesos():
    if "atencao" in st.session_state:
        st.write("### Visualização da Cabelas de Atenção")
        tokens = st.session_state["tokens"]
        attn_all = st.session_state["atencao"][-1][-1][0]
        n_heads = attn_all.shape[0]

        for i in range(0, n_heads, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < n_heads:
                    with cols[j]:
                        attn = attn_all[i + j]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(attn.detach().cpu().numpy(), cmap="plasma")
                        ax.set_title(f"Cabeça {i + j}")
                        ax.set_xticks(range(len(tokens)))
                        ax.set_yticks(range(len(tokens)))
                        ax.set_xticklabels(tokens, rotation=45, ha="right")
                        ax.set_yticklabels(tokens)
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)

    if "atencao" in st.session_state:
        st.write("### Probabilidades de Saída dos Tokens da Resposta")
        tokens = st.session_state["tokens"]
        device = st.session_state["device"]
        model = st.session_state["model"]
        token_to_id = st.session_state["token_to_id"]
        input_ids = torch.tensor([[token_to_id[t] for t in tokens]]).to(device)

        with torch.no_grad():
            logits, _ = model(input_ids)
            probs = torch.softmax(logits[0], dim=-1)

        id_to_token = st.session_state["id_to_token"]
        for i, prob_dist in enumerate(probs):
            topk = torch.topk(prob_dist, 5)
            token_real = tokens[i]
            st.markdown(f"---\n**🎯 Token gerado:** `{token_real}`")
            for idx, prob in zip(topk.indices, topk.values):
                token_pred = id_to_token[idx.item()]
                bar = "█" * int(prob.item() * 20)
                st.markdown(f"- `{token_pred}`: **{prob.item():.4f}** {bar}")
