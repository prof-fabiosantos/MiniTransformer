# 🧠 MiniTransformer

**MiniTransformer** é uma implementação educacional e compacta da arquitetura Transformer, projetada para tarefas de NLP como geração de texto e perguntas/respostas. Leve, interpretável e pronta para personalização, ela é ideal para estudantes, entusiastas e experimentos locais.

![Logo MiniTransformer](./logo_minitransformer.png)

---

## 🚀 Recursos
- Arquitetura Transformer simplificada com PyTorch puro
- Camada de atenção multi-cabeça (multi-head self-attention)
- Conexões residuais com normalização e MLP
- Projeções integradas de Q, K e V
- Geração de texto com temperatura, top-k e top-p
- Dataset estilo QA baseado em texto plano

---

## 🧱 Arquitetura Interna
O `MiniTransformer` é composto pelos seguintes blocos principais:

- **Embedding de palavras e posições**
- **`MultiHeadSelfAttention`**: atenção com projeção conjunta QKV, máscara causal e concatenação de cabeças
- **`TransformerMLP`**: MLP com GELU e projeções lineares
- **`TransformerBlock`**: bloco completo com LayerNorm, residual e atenção + MLP
- **`SimpleTransformer`**: empilhamento de blocos, seguido por uma projeção linear para logits

Exemplo de configuração:
```python
model = SimpleTransformer(
    vocab_size=len(vocab),
    embed_dim=128,
    num_heads=8,
    num_layers=4
)
```

---

## 📁 Estrutura do projeto
```bash
transformer_chatbot_project/
├── model.py                # Arquitetura do MiniTransformer
├── train_transformer_v3.py  # Script de treino e geração
├── textDataset.txt        # Dados de treinamento com pares QA
├── modelo_transformer.pt  # Modelo treinado
├── vocab_transformer.json # Vocabulário serializado
└── TransformerChatbotTreino.ipynb  # Versão notebook interativo
```

---

## 🏁 Como usar

```bash
pip install torch streamlit numpy

# Treinar modelo
python train_transformer_v3.py

# Rodar app Streamlit (caso implementado)
streamlit run transformer_app.py
```

---

## 📌 Parâmetros típicos
- Camadas: 4
- Cabeças de atenção: 8
- Embedding: 128
- Hidden MLP: 256
- Tokens máximos por prompt: 100
- Parâmetros totais: ~806.000

---

## ✨ Exemplo de uso
```text
prompt: pergunta: quanto é 2 mais 2? resposta:
output: 4 <eos>
```

---

## 📚 Referências
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- Baseado em práticas de modelos como GPT-2

---

Desenvolvido com ❤️ para aprendizado, prototipagem e experimentação.

> "Construa seu próprio modelo, compreenda cada atenção."
