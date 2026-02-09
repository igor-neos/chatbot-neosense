from pathlib import Path
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Configuração da API
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# Lê o arquivo de conhecimento
docs_path = Path("docs/manuais_neosense.txt")
raw_text = docs_path.read_text(encoding="utf-8")

# 🔹 CHUNKING (ESSENCIAL)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

chunks = text_splitter.split_text(raw_text)

# Gera o FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)

# Salva o índice
vectorstore.save_local("faiss_index_neosense")

print(f"✅ FAISS gerado com {len(chunks)} chunks")
