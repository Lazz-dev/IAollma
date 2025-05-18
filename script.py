from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carregando o texto
loader = TextLoader("texto_base.txt", encoding='utf-8')
documents = loader.load()

# Dividindo em pedaços menores
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Gerando embeddings com o modelo do Ollama
embeddings = OllamaEmbeddings(model="llama3")
db = FAISS.from_documents(docs, embeddings)

# Criando a cadeia de perguntas e respostas
retriever = db.as_retriever()
llm = ChatOllama(model="llama3")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Loop de perguntas
while True:
    pergunta = input("Digite uma pergunta (ou 'sair' para encerrar):\n→ ")
    if pergunta.lower() == "sair":
        break
    resposta = qa.invoke({"query": pergunta})
    print("Resposta:", resposta["result"])
