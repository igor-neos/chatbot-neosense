# ingestao_github.py - V7.2: Modelo Correto (gemini-embedding-001)

import os
import time
import shutil
import subprocess
from pathlib import Path

# --- CONFIGURAÇÕES ---
REPO_URL = "https://github.com/GCNeos/neos-documentacao.git"
BRANCH = "main"
PASTA_INDICE = "faiss_index_neosense"
PASTA_TEMP_CLONE = "./temp_docs_repo"

# Autenticação
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if GITHUB_TOKEN and "github.com" in REPO_URL and "@" not in REPO_URL:
    REPO_URL = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    # Tenta pegar dos secrets do Streamlit se estiver rodando localmente e a env var falhar
    try:
        import streamlit as st
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        print("❌ ERRO: GOOGLE_API_KEY não definida.")
        exit()

# Imports LangChain
try:
    from langchain_community.document_loaders import GitLoader
except ImportError:
    from langchain_community.document_loaders.git import GitLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print(f"🚀 Iniciando ingestão V7.2 (Modelo: models/gemini-embedding-001)...")

# ---------------------------------------------------------
# 1. Clone manual do repositório
# ---------------------------------------------------------
def clonar_repositorio():
    if os.path.exists(PASTA_TEMP_CLONE):
        print("🧹 Limpando pasta temporária antiga...")
        def on_rm_error(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            os.unlink(path)
        shutil.rmtree(PASTA_TEMP_CLONE, onerror=on_rm_error)

    print("📥 Clonando repositório...")
    cmd = [
        "git", "clone",
        "--depth", "1",
        "--branch", BRANCH,
        "--single-branch",
        REPO_URL,
        PASTA_TEMP_CLONE
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print("✅ Clone realizado com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro no Git: {e}")
        exit()

# ---------------------------------------------------------
# 2. Análise de caminho (SOMENTE metadados)
# ---------------------------------------------------------
def analisar_caminho(path):
    path_lower = path.lower()

    produto = "neosense"
    versao = "atual"
    prioridade = "media"

    if "neosense agenda do vendedor" in path_lower:
        produto = "agenda_vendedor"
    elif "neosense aplicativo" in path_lower and "e-commerce" not in path_lower:
        produto = "app_mobile"
    elif "portal web" in path_lower:
        produto = "portal_web"
    elif "neosense e-commerce" in path_lower:
        produto = "ecommerce"
    elif "neosense crm" in path_lower:
        produto = "crm"

    if "legado" in path_lower:
        versao = "legado"
    elif "redesign" in path_lower:
        versao = "redesign"

    caminho = (
        path.replace(".md", "")
            .replace("\\", "/")
            .split("manuais-neosense/")[-1]
    )

    return produto, versao, prioridade, caminho

# ---------------------------------------------------------
# 3. Execução principal
# ---------------------------------------------------------

clonar_repositorio()

print("📂 Lendo arquivos...")
loader = GitLoader(
    repo_path=PASTA_TEMP_CLONE,
    branch=BRANCH,
    file_filter=lambda f: f.endswith(".md")
)
raw_docs = loader.load()
print(f"📄 Arquivos encontrados: {len(raw_docs)}")

processed_docs = []

# Splitter estrutural (semântica)
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "titulo"),
        ("##", "secao"),
        ("###", "subsecao")
    ]
)

# Splitter de tamanho (controle, não motor)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

print("🔧 Processando documentos...")

for doc in raw_docs:
    source = doc.metadata.get("source", "")
    produto, versao, prioridade, caminho = analisar_caminho(source)

    # ⚠️ NÃO injetamos metadados no texto para não poluir
    texto_limpo = doc.page_content

    # Remove restos comuns do GitBook
    lixo = [
        "<figure", "</figure>", "<img", "alt=\"\"", "figcaption",
        "http://localhost", ".gitbook/assets"
    ]
    for item in lixo:
        texto_limpo = texto_limpo.replace(item, "")

    doc.page_content = texto_limpo

    md_chunks = markdown_splitter.split_text(doc.page_content)

    for chunk in md_chunks:
        chunk.metadata.update({
            "produto": produto,
            "versao": versao,
            "prioridade": prioridade,
            "caminho": caminho
        })

    processed_docs.extend(md_chunks)

final_chunks = text_splitter.split_documents(processed_docs)
print(f"🧩 Chunks finais: {len(final_chunks)}")

# ---------------------------------------------------------
# 4. Vetorização
# ---------------------------------------------------------
print("🧠 Criando índice FAISS...")

if os.path.exists(PASTA_INDICE):
    shutil.rmtree(PASTA_INDICE)

# --- CORREÇÃO: Usando o modelo validado pelo script de diagnóstico ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

batch_size = 100
vector_store = None

for i in range(0, len(final_chunks), batch_size):
    batch = final_chunks[i:i + batch_size]
    try:
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
        time.sleep(1)
        print(f"   Lote {i//batch_size + 1} processado.")
    except Exception as e:
        print(f"⚠️ Erro no lote: {e}")

if vector_store:
    vector_store.save_local(PASTA_INDICE)
    print("✅ Índice FAISS criado com sucesso!")

# Limpeza
try:
    def on_rm_error(func, path, exc_info):
        import stat
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)
    shutil.rmtree(PASTA_TEMP_CLONE, onerror=on_rm_error)
except:
    pass