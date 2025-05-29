
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
from train_transformer_main import QADataset, SimpleTransformer, top_k_top_p_filtering

st.title("🧠 Treinamento Transformer")

st.sidebar.header("🔧 Hiperparâmetros")
seq_len = st.sidebar.slider("Tamanho da sequência", 5, 100, 10)
batch_size = st.sidebar.slider("Tamanho do Batch", 1, 64, 5)
embed_dim = st.sidebar.selectbox("Dimensão do Embedding", [64, 128, 256], index=1)
num_heads = st.sidebar.selectbox("Número de Cabeças", [2, 4, 8], index=2)
num_layers = st.sidebar.slider("Número de Camadas", 1, 12, 4)
learning_rate = st.sidebar.number_input("Taxa de Aprendizado", value=5e-4, format="%.5f")
num_epochs = st.sidebar.slider("Épocas", 1, 100, 15)

start_button = st.sidebar.button("🚀 Iniciar Treinamento")

if start_button:
    st.write("### ⏳ Carregando dados e modelo...")
    dataset = QADataset("textDataset.txt", seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleTransformer(
        vocab_size=len(dataset.token_to_id),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        st.write(f"📉 Época {epoch+1}: Loss médio = {avg_loss:.4f}")

    torch.save(model.state_dict(), "modelo_transformer.pt")
    with open("vocab_transformer.json", "w", encoding="utf-8") as f:
        json.dump(dataset.token_to_id, f)
    st.success("✅ Modelo treinado e salvo com sucesso.")

    st.write("### 📈 Gráfico de perda")
    fig, ax = plt.subplots()
    ax.plot(loss_history, label="Perda média")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)
