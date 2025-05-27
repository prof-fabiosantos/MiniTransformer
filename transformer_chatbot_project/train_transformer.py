# transformer_chatbot_treino.py (com arquitetura ampliada)
import torch  # Biblioteca principal para tensores e deep learning
import torch.nn as nn  # Submódulo com definições de camadas
import torch.nn.functional as F  # Submódulo com funções como softmax, cross_entropy etc.
from torch.utils.data import Dataset, DataLoader  # Utilitários para criar datasets e iteradores
import numpy as np  # Biblioteca para manipulação numérica
import json  # Para salvar e carregar arquivos JSON
from model import SimpleTransformer  # Importa o modelo Transformer definido externamente
import random  # Biblioteca padrão de aleatoriedade

# Função para aplicar top-k e top-p (nucleus) filtering nos logits antes da amostragem

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    top_k = min(top_k, logits.size(-1))  # Garante que top_k não seja maior que o número de tokens
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)  # Pega os top-k maiores logits
        min_values = values[:, -1].unsqueeze(-1)  # Pega o menor valor entre os top-k
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)  # Zera os menores
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # Ordena os logits em ordem decrescente
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # Soma cumulativa das probabilidades
        sorted_indices_to_remove = cumulative_probs > top_p  # Identifica onde excede top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()  # Garante pelo menos um token
        sorted_indices_to_remove[:, 0] = 0  # Sempre mantém o mais provável
        indices_to_remove = sorted_indices[sorted_indices_to_remove]  # Índices a remover
        logits[0, indices_to_remove] = float('-inf')  # Atribui -inf para remover esses logits
    return logits

# Define um dataset para pares pergunta/resposta
class QADataset(Dataset):
    def __init__(self, filepath, seq_len=20):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()  # Lê todas as linhas do arquivo

        tokens = []
        for line in lines:
            line = line.strip().lower()  # Remove espaços e converte para minúsculas
            if not line: continue
            tokens.extend(line.split() + ['<eos>'])  # Tokeniza e adiciona token de fim de sequência
        vocab = sorted(set(tokens))  # Cria vocabulário único ordenado
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}  # Mapeia token para ID
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}  # Mapeia ID para token
        self.data = [self.token_to_id[tok] for tok in tokens]  # Converte os tokens em IDs
        self.seq_len = seq_len  # Tamanho da janela de entrada

    def __len__(self):
        return len(self.data) - self.seq_len  # Número total de amostras

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len])  # Entrada
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1])  # Rótulo (próximo token)
        return x, y

if __name__ == "__main__":
    dataset = QADataset("textDataset.txt", seq_len=10)  # Cria dataset com janelas de 10 tokens
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # DataLoader com batch size 4

    # Instancia o modelo Transformer
    model = SimpleTransformer(
        vocab_size=len(dataset.token_to_id),
        embed_dim=128,
        num_heads=8,
        num_layers=4
    )

    # Calcula número total de parâmetros treináveis
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo criado com {total_params:,} parâmetros.")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # Otimizador Adam
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU se disponível

    model.train()  # Modo treinamento
    model.to(device)  # Move o modelo para GPU ou CPU

    for epoch in range(15):  # Treina por 15 épocas
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)  # Move dados para o dispositivo
            logits, _ = model(x)  # Faz a previsão
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # Calcula perda
            optimizer.zero_grad()  # Zera gradientes antigos
            loss.backward()  # Backpropagation
            optimizer.step()  # Atualiza os pesos
            total_loss += loss.item()  # Acumula perda
        print(f"Época {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")  # Mostra estatísticas

    torch.save(model.state_dict(), "modelo_transformer.pt")  # Salva pesos do modelo
    with open("vocab_transformer.json", "w", encoding="utf-8") as f:
        json.dump(dataset.token_to_id, f)  # Salva vocabulário
    print("Modelo e vocabulário salvos com sucesso.")

    # Geração de texto com controle de temperatura e amostragem
    model.eval()  # Modo avaliação
    prompt = "pergunta: qual a capital do brasil? resposta:"  # Frase inicial
    tokens = prompt.lower().split()  # Tokeniza a frase
    input_ids = torch.tensor([[dataset.token_to_id[t] for t in tokens]]).to(device)  # Converte para IDs

    generated = input_ids  # Começa com o prompt
    temperature = 1.0  # Temperatura padrão
    top_k = 5  # Considera os 5 tokens mais prováveis
    top_p = 0.9  # Considera o núcleo de 90% da probabilidade
    max_tokens = 10  # Gera até 10 tokens

    for _ in range(max_tokens):
        logits, _ = model(generated)  # Predição
        logits = logits[:, -1, :] / temperature  # Aplica temperatura no último token
        filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)  # Filtra
        probs = F.softmax(filtered_logits, dim=-1)  # Normaliza
        next_token = torch.multinomial(probs, num_samples=1)  # Amostra o próximo token
        generated = torch.cat([generated, next_token], dim=1)  # Adiciona ao prompt

    texto_final = " ".join([dataset.id_to_token[i] for i in generated[0].tolist()])  # Reconstrói o texto
    print("\nTexto gerado:")
    print(texto_final)  # Mostra o resultado final

