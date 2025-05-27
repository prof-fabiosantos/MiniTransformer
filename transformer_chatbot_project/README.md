# Chatbot com Mini Transformer

Este projeto implementa um chatbot leve baseado na arquitetura Transformer, treinado com pares de pergunta e resposta, usando PyTorch e Streamlit.

---

## 🚀 Funcionalidades

* Treinamento com dados estilo FAQ
* Geração de respostas com controle por:

  * `temperature`
  * `top-k`
  * `top-p` (nucleus sampling)
* Interface interativa com Streamlit

---

## 📦 Requisitos

```bash
pip install torch streamlit numpy
```

---

## 🧠 Treinamento do modelo

1. Crie um arquivo `textDataset.txt` com perguntas e respostas, uma por linha:

```
pergunta: qual a capital do brasil? resposta: brasília.
pergunta: quanto é 2 mais 2? resposta: 4.
```

2. Execute o script de treinamento:

```bash
python train_transformer_v3.py
```

Isso irá gerar os arquivos:

* `modelo_transformer.pt`
* `vocab_transformer.json`

---

## 💬 Usando o Chatbot (Streamlit)

Execute o app com:

```bash
streamlit run transformer_chabot_v2.py
```

### Interface permite ajustar:

* Quantidade de palavras geradas
* Temperatura da amostragem
* Top-k e Top-p

Digite uma pergunta como:

```
qual a capital do brasil?
```

E veja a resposta gerada com base no modelo treinado.

---

## 📁 Estrutura do projeto

```
├── transformer_chatbot_treino.py      # Script de treino
├── transformer_chatbot_app.py         # Interface web com Streamlit
├── model.py                           # Arquitetura Transformer
├── textDataset.txt                    # Dataset de treino
├── modelo_transformer.pt              # Modelo salvo
├── vocab_transformer.json             # Vocabulário salvo
```

---

## 📌 Observações

* O modelo foi feito para fins didáticos.
* Pode ser expandido com embeddings melhores e tokenizer de subpalavras.

---

## 🧑‍💻 Autor

Este projeto foi gerado com o apoio do ChatGPT e adaptações personalizadas para mini Transformers.
