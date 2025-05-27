# Este software é licenciado apenas para fins educacionais e não pode ser utilizado para fins comerciais.

import torch  # Biblioteca principal para tensores e deep learning
import torch.nn as nn  # Submódulo com camadas
import torch.nn.functional as F  # Funções como softmax, gelu, etc.
import numpy as np  # Biblioteca para operações matemáticas

# Camada de atenção com múltiplas cabeças
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0  # Garante que a divisão entre cabeças seja inteira
        self.head_dim = embed_dim // num_heads  # Dimensão por cabeça
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)  # Projeção conjunta para Q, K e V
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # Projeção final após concatenação das cabeças

    def forward(self, x, mask=True):
        B, T, C = x.shape  # B=batch, T=seq len, C=embed dim
        qkv = self.qkv_proj(x)  # Projeta Q, K, V de uma vez
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # Separa e reorganiza
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Divide Q, K e V
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # Escora atenção
        if mask:
            tri_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)  # Máscara triangular superior
            attn_scores = attn_scores.masked_fill(tri_mask, float('-inf'))  # Aplica máscara
        attn_weights = F.softmax(attn_scores, dim=-1)  # Converte em pesos com softmax
        out = torch.matmul(attn_weights, V)  # Aplica os pesos às V
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # Junta as cabeças
        return self.out_proj(out), attn_weights  # Aplica projeção final e retorna pesos de atenção

# Feedforward do bloco Transformer (MLP)
class TransformerMLP(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # Expande dimensionalidade
        self.act = nn.GELU()  # Função de ativação não-linear
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # Reduz de volta para embed_dim

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))  # MLP com ativação GELU

# Bloco Transformer completo com atenção + normalização + MLP
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_hidden=256):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)  # Atenção
        self.norm1 = nn.LayerNorm(embed_dim)  # Normalização pré-atenção
        self.mlp = TransformerMLP(embed_dim, mlp_hidden)  # Feedforward
        self.norm2 = nn.LayerNorm(embed_dim)  # Normalização pré-MLP

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x))  # Atenção com residual
        x = x + attn_out  # Conexão residual
        mlp_out = self.mlp(self.norm2(x))  # MLP com normalização
        x = x + mlp_out  # Conexão residual final
        return x

# Modelo Transformer simples completo
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  # Embedding de tokens
        self.pos_emb = nn.Parameter(torch.randn(1, 100, embed_dim))  # Embedding de posição
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])  # Lista de blocos Transformer
        self.head = nn.Linear(embed_dim, vocab_size)  # Camada final para previsão de tokens

    def forward(self, x):
        B, T = x.shape  # B=batch, T=seq len
        x = self.embed(x) + self.pos_emb[:, :T, :]  # Soma embeddings com posições
        for block in self.blocks:
            x = block(x)  # Aplica cada bloco Transformer
        logits = self.head(x)  # Projeta para o vocabulário
        return logits, None  # Retorna logits
