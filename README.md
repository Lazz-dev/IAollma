# Projeto IA Generativa com LangChain e Ollama

Este projeto demonstra como usar a biblioteca LangChain com o modelo LLaMA 3 do Ollama para criar um sistema de perguntas e respostas baseado em texto.

## 🚀 Funcionalidades

- Carregamento e segmentação de textos
- Geração de embeddings com Ollama
- Armazenamento vetorial com FAISS
- Respostas interativas com IA generativa

## 🧠 Modelo utilizado

- [LLaMA 3](https://ollama.com/library/llama3), executado via [Ollama](https://ollama.com/)

## 📁 Requisitos

- Python 3.10+
- Ollama instalado e rodando localmente
- Modelo `llama3` baixado com:  
  ```bash
  ollama pull llama3
- todas as dependencias instaladas com o comando no terminal, comando usado: pip install langchain langchain-community langchain-ollama faiss-cpu
