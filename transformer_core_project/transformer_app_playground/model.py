#© 2025 Prof. Dr. Fabio Santos. Uso permitido apenas para fins educacionais e não comerciais.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=True, return_attn=False):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask:
            tri_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            attn_scores = attn_scores.masked_fill(tri_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return (out, attn_weights) if return_attn else (out, None)

class TransformerMLP(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_hidden=256):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(embed_dim, mlp_hidden)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, return_attn=False):
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x, return_attn=return_attn)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, attn_weights

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, return_attention=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)
        self.return_attention = return_attention

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos_emb[:, :T, :]
        attn_all = []

        for block in self.blocks:
            x, attn = block(x, return_attn=self.return_attention)
            if self.return_attention:
                attn_all.append(attn)

        logits = self.head(x)
        return logits, attn_all if self.return_attention else None


