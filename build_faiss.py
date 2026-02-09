from pathlib import Path
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# ⚠️ Usa a MESMA chave e o MESMO modelo do app
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# Fonte da base
docs_path = Path("docs/manuais_neosense.txt")

texts = [docs_path.read_text(encoding="utf-8")]

# Cria o FAISS
vectorstore = FAISS.from_texts(texts, embeddings)

# Salva localmente
vectorstore.save_local("faiss_index_neosense")

print("✅ FAISS gerado com sucesso")
