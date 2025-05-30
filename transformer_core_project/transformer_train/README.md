
# 🧠 Transformer Train App

Este projeto contém dois arquivos principais:

- `train_transformer_main.py`: responsável por treinar um modelo Transformer simples.
- `train_transformer_app.py`: aplicativo em Streamlit que permite configurar e executar o treinamento de forma interativa.

---

## 📂 Sobre o `train_transformer_main.py`

O arquivo `train_transformer_main.py` é responsável por **treinar um modelo Transformer simples** para tarefas de geração de texto com base em pares de pergunta e resposta. Ele segue uma arquitetura compacta e educacional, ideal para fins de aprendizado, prototipagem e experimentação.

### Funcionalidades:

- **Pré-processamento do dataset** (`textDataset.txt`).
- **Definição do modelo Transformer** com número ajustável de camadas, cabeças, embeddings.
- **Treinamento supervisionado** com otimização via `Adam` e `cross_entropy`.
- **Salvamento dos pesos do modelo** e vocabulário treinado.
- **Geração de texto** a partir de um prompt, usando top-k e top-p sampling.

---

## 🌐 Sobre o `train_transformer_app.py`

O arquivo `train_transformer_app.py` é um **frontend interativo criado com Streamlit** que permite ao usuário configurar e treinar o modelo Transformer sem editar o código.

### Funcionalidades:

- Interface com **sliders e caixas de seleção** para ajustar os hiperparâmetros.
- Visualização **dinâmica** do batch atual (tokens e palavras).
- **Heatmap de atenção** gerado a cada batch.
- **Gráfico de perda** após o treinamento.
- **Salvamento automático** do modelo e vocabulário.

---

## ▶️ Como Executar

1. Instale as dependências:

```bash
pip install streamlit matplotlib torch
```

2. Certifique-se de ter o arquivo `textDataset.txt` na mesma pasta.
3. Execute o aplicativo:

```bash
streamlit run train_transformer_app.py
```

---

Desenvolvido pelo Prof. Fabio Santos (EST/UEA )para fins educacionais e demonstrações de IA generativa com Transformers.
