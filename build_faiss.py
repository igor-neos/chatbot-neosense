from pathlib import Path
import time
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# -----------------------
# CONFIG API
# -----------------------
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# -----------------------
# LOAD DOCS
# -----------------------
docs_path = Path("docs/manuais_neosense.txt")
raw_text = docs_path.read_text(encoding="utf-8")

# -----------------------
# CHUNKING
# -----------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = text_splitter.split_text(raw_text)

# -----------------------
# BUILD FAISS IN BATCHES
# -----------------------
BATCH_SIZE = 10      # 👈 crítico para evitar 504
SLEEP_SECONDS = 1.2  # 👈 respeita rate limit

vectorstore = None

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]

    if vectorstore is None:
        vectorstore = FAISS.from_texts(batch, embeddings)
    else:
        vectorstore.add_texts(batch)

    time.sleep(SLEEP_SECONDS)

# -----------------------
# SAVE INDEX
# -----------------------
vectorstore.save_local("faiss_index_neosense")

print(f"✅ FAISS gerado com {len(chunks)} chunks em batches")
