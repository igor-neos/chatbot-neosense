# app.py - Versão 3.9.3 (Firebase via JSON local, Gemini 2.5, SimpleRAG + JIRA + Redação)

import os
import ssl
import certifi
import re
from pathlib import Path
from datetime import datetime
import streamlit as st
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- CORREÇÃO SSL PARA AMBIENTES CORPORATIVOS ---
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
# --- FIM ---

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from jira import JIRA

import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------
# ⚙️ CONFIGURAÇÃO DA PÁGINA
# -----------------------
st.set_page_config(
    page_title="Chatbot Neosense",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# 🔐 CONTROLE DE ACESSO
# -----------------------
def check_password():
    if st.session_state.get("password_correct", False):
        return True
    password = st.text_input("Digite a senha para acessar:", type="password")
    if not password:
        st.stop()
    if password == st.secrets["APP_PASSWORD"]:
        st.session_state.password_correct = True
        st.rerun()
    else:
        st.error("😕 Senha incorreta.")
        st.stop()


check_password()

# -----------------------
# 🔑 CONFIGURAÇÃO API
# -----------------------
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("Chave GOOGLE_API_KEY ausente nos segredos.")
    st.stop()

# -----------------------
# 🔥 FIREBASE (local JSON OU secrets do Streamlit Cloud)
# -----------------------
import tempfile
import json
from pathlib import Path

try:
    if not firebase_admin._apps:
        cred = None

        # 1) Tenta primeiro o arquivo local (desenvolvimento)
        cred_path_local = Path(__file__).parent / "firebase_credentials.json"
        if cred_path_local.exists():
            # Modo dev: arquivo JSON local (NÃO versionado no Git)
            cred = credentials.Certificate(str(cred_path_local))
        else:
            # 2) Se não tiver arquivo, tenta pegar das secrets (Streamlit Cloud)
            try:
                cred_dict = dict(st.secrets["firebase_creds"])

                # Ajusta quebras de linha da private_key, se vier com "\n" escapado
                pk = cred_dict.get("private_key", "")
                if "\\n" in pk:
                    cred_dict["private_key"] = pk.replace("\\n", "\n").strip()

                cred = credentials.Certificate(cred_dict)
            except Exception as e:
                st.error(
                    "Credenciais do Firebase não encontradas.\n\n"
                    "Localmente, garanta que o arquivo 'firebase_credentials.json' "
                    "está na mesma pasta do app.py.\n"
                    "No Streamlit Cloud, configure a seção [firebase_creds] em Settings > Secrets."
                )
                st.stop()

        firebase_admin.initialize_app(cred)

    db = firestore.client()

except Exception as e:
    st.error(f"Erro ao conectar ao Firebase: {e}")
    st.stop()

# -----------------------
# 🔥 LOGGING
# -----------------------
def log_to_firestore(
    pergunta,
    resposta=None,
    erro=None,
    modo="normal",
    question_type=None,
    ticket=None,
):
    try:
        data = {
            "timestamp": datetime.now().isoformat(),
            "pergunta": pergunta,
            "resposta": resposta,
            "erro": erro,
            "feedback": None,
            "modo": modo,
            "question_type": question_type,
            "ticket": ticket,
        }
        _, doc_ref = db.collection("chat_logs").add(data)
        return doc_ref.id
    except Exception as e:
        st.warning(f"⚠️ Erro ao salvar log no Firestore: {e}")
        return None


def update_feedback_in_firestore(doc_id, feedback):
    try:
        db.collection("chat_logs").document(doc_id).update({"feedback": feedback})
    except Exception as e:
        st.warning(f"⚠️ Erro ao atualizar feedback: {e}")

# -----------------------
# 🧠 FUNÇÕES GERAIS
# -----------------------
NOT_FOUND_MSG = "Não encontrei essa informação na base de conhecimento."


def detect_question_type(q: str) -> str:
    q = q.lower()
    if "diferença" in q or "diferenca" in q:
        return "comparação"
    if "relação" in q or "relacao" in q or "ligação" in q or "ligacao" in q:
        return "relacional"
    if q.startswith("como ") or "como " in q:
        return "procedimento"
    return "conceitual"


SYN_MAP = {
    r"senha(s)?": "senha login acesso recuperar credenciais esqueci senha token",
    r"login": "login acesso autenticação entrar conectar token",
    r"campanha(s)?": "campanha ofertas descontos promoção marketing criar editar configurar vantagem público geral",
    r"miss(ões|ao)": "missões metas período apuração gasto referência pontuação objetivo missão fidelidade incremento",
    r"agenda do vendedor": "agenda do vendedor carteira clientes contato campanhas aplicativo vendedor dashboard vendas material apoio",
    r"beneficio(s)?": "benefício recompensa catálogo missão criar editar adicionar prêmio catálogo de benefícios fidelidade",
    r"pré[- ]cadastro": "pré-cadastro pre cadastro cadastro inicial cliente parcial loja oferta lead",
    r"catálogo": "catálogo benefícios recompensas prêmios lista registrar adicionar item resgate disponível utilizado",
    r"pdv": "pdv integração ponto de venda loja código externo id_externo_organizacao saldo recompensa imprimir cupom",
}


def expand_query(user_query: str) -> str:
    expanded = user_query
    for patt, exp in SYN_MAP.items():
        if re.search(patt, user_query, flags=re.IGNORECASE):
            expanded += f" ({exp})"
    return expanded


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def log_unanswered(question: str, aug_query: str):
    try:
        logs_dir = Path("logs")
        ensure_dir(logs_dir)
        with open(
            logs_dir / "perguntas_nao_respondidas.csv", "a", encoding="utf-8"
        ) as f:
            ts = datetime.now().isoformat()
            question_limpa = question.replace('"', "'")
            aug_query_limpa = aug_query.replace('"', "'")
            f.write(f'"{ts}","{question_limpa}","{aug_query_limpa}"\n')
    except Exception:
        pass

# -----------------------
# 🎫 Integração com JIRA
# -----------------------
def extract_ticket_id_from_input(text_input):
    if not text_input:
        return None
    match = re.search(r"([A-Z]+-[0-9]+)", text_input.upper())
    return match.group(1) if match else None


@st.cache_data(ttl=600)
def fetch_jira_data(ticket_id):
    try:
        jira_options = {"server": st.secrets["JIRA_SERVER"]}
        jira = JIRA(
            options=jira_options,
            basic_auth=(st.secrets["JIRA_USERNAME"], st.secrets["JIRA_API_TOKEN"]),
        )
        issue = jira.issue(ticket_id)
        return (
            f"Dados do Ticket JIRA: {ticket_id}\n"
            f"- Título: {issue.fields.summary}\n"
            f"- Status: {issue.fields.status.name}\n"
            f"- Responsável: {issue.fields.assignee.displayName if issue.fields.assignee else 'Ninguém atribuído'}\n"
            f"- Descrição: {issue.fields.description if issue.fields.description else 'Sem descrição.'}"
        )
    except Exception as e:
        return f"Erro ao buscar o ticket {ticket_id}: {e}"

# -----------------------
# ✍️ MODO REDAÇÃO
# -----------------------
def detect_redacao_mode(prompt: str) -> bool:
    p = prompt.lower()
    gatilhos = [
        "crie um rascunho",
        "crie um rascunho da documentação",
        "documente este ticket",
        "documente o ticket",
        "gerar documentação",
        "gerar rascunho de documentação",
        "escreva a documentação",
        "faça a documentação",
        "crie a documentação",
        "documente a funcionalidade",
        "quero documentar",
    ]
    if any(g in p for g in gatilhos):
        return True
    # Gatilho mais genérico: rascunho + algo de doc/ticket/funcionalidade
    if "rascunho" in p and (
        "documenta" in p
        or "documentação" in p
        or "documentacao" in p
        or "ticket" in p
        or "funcionalidade" in p
    ):
        return True
    return False


REDACAO_PROMPT = """
Você é um redator técnico sênior da equipe de documentação da Neos Tecnologia.
Transforme o conteúdo técnico em texto claro, objetivo e escaneável para o GitBook interno do Neosense CRM.

Regras de escrita:
- Use títulos curtos e objetivos.
- Mantenha tom técnico, direto e consistente.
- Explique: o que é a funcionalidade, por que ela existe e como o usuário usa.
- Estruture nas seções: Introdução, Configuração, Funcionamento e 💡 Dica/Observação.
- Combine o módulo e o título principal em uma única linha, no formato:
  **[Módulo: NomeDoMódulo] Nome da funcionalidade**
- Não invente funcionalidades que não aparecem nas fontes.
- Não repita blocos inteiros no final.
- Não inclua instruções internas como "esta seção fala sobre".

--- DADOS DO TICKET (fonte primária):
{jira_data}

--- CONTEXTO DA BASE DE CONHECIMENTO (RAG):
{rag_context}

Agora escreva a documentação técnica final:
"""

# -----------------------
# 🧩 DETECÇÃO DE PERGUNTAS SOBRE TICKET JIRA
# -----------------------
def detect_jira_ticket_question(prompt: str) -> bool:
    p = prompt.lower().strip()

    # Padrões que claramente se referem a "ticket" de venda, não JIRA
    crm_ticket_padroes = [
        "ticket médio",
        "ticket medio",
        "tickets loja",
        "ticket loja",
        "desconto no ticket",
        "campanha de desconto no ticket",
    ]
    if any(b in p for b in crm_ticket_padroes):
        return False

    # 1) Gatilhos explícitos
    gatilhos = [
        "resuma este ticket",
        "resuma o ticket",
        "resumo do ticket",
        "faça um resumo do ticket",
        "resuma esta atividade",
        "resuma esta atividade do jira",
        "resuma a atividade do jira",
        "atividade do jira",
        "plano de teste",
        "planos de teste",
        "o que mudou",
        "o que foi alterado",
        "como era antes",
        "como é a partir desta atividade",
        "impacto desta atividade",
        "impactos desta atividade",
    ]
    if any(g in p for g in gatilhos):
        return True

    # 2) Frases do tipo "segundo o ticket", "de acordo com este ticket"
    if ("ticket" in p or "jira" in p) and any(
        h in p
        for h in [
            "segundo",
            "de acordo",
            "este ticket",
            "esse ticket",
            "neste ticket",
            "deste ticket",
            "no ticket",
        ]
    ):
        return True

    # 3) Perguntas genéricas sobre "o que este ticket..." quando há JIRA
    if "ticket" in p and any(
        k in p
        for k in [
            "resumo",
            "resuma",
            "explique",
            "descreva",
            "fale sobre",
            "o que este ticket",
            "o que esse ticket",
        ]
    ):
        return True

    return False


def has_jira_context() -> bool:
    ctx = st.session_state.get("jira_context")
    return bool(ctx)

# -----------------------
# 📚 RAG (FAISS + MMR) - com SimpleRAG + filtro de confiança
# -----------------------
class SimpleRAG:
    def __init__(self, retriever, llm, prompt_template: PromptTemplate, vectorstore=None):
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt_template = prompt_template
        # limiar de distância/similaridade (scores muito altos = pouco similares)
        self.similarity_threshold = 0.5

    def _format_docs(self, docs):
        pieces = []
        for d in docs:
            if hasattr(d, "page_content"):
                pieces.append(d.page_content)
            else:
                pieces.append(str(d))
        return "\n\n".join(pieces)

    def invoke(self, inputs: dict):
        query = inputs.get("query") if isinstance(inputs, dict) else inputs

        docs = []
        scores = None

        try:
            results = None
            if self.vectorstore is not None and hasattr(
                self.vectorstore, "similarity_search_with_score"
            ):
                results = self.vectorstore.similarity_search_with_score(query, k=10)
            elif hasattr(self.retriever, "similarity_search_with_score"):
                results = self.retriever.similarity_search_with_score(query, k=10)

            if results is not None:
                docs = [r[0] for r in results]
                scores = [r[1] for r in results]
            else:
                if hasattr(self.retriever, "get_relevant_documents"):
                    docs = self.retriever.get_relevant_documents(query)
                else:
                    docs = self.retriever.invoke(query)

            top_score = scores[0] if scores else None
            debug_info = {
                "query": str(query),
                "count": len(docs),
                "top_score": top_score,
                "snippet": (
                    docs[0].page_content[:800]
                    if docs and hasattr(docs[0], "page_content")
                    else ""
                ),
                "error": None,
            }
        except Exception as e:
            docs = []
            scores = None
            debug_info = {
                "query": str(query),
                "count": 0,
                "top_score": None,
                "snippet": "",
                "error": str(e),
            }

        # guarda debug para inspecionar depois (quando SHOW_DEBUG=True)
        try:
            st.session_state["rag_debug"] = debug_info
        except Exception:
            pass

        # filtro de confiança: se o doc mais próximo ainda ficou "longe", retorna NOT_FOUND
        if scores and scores[0] is not None:
            try:
                if scores[0] > self.similarity_threshold:
                    return {"result": NOT_FOUND_MSG}
            except Exception:
                pass

        if not docs:
            return {"result": NOT_FOUND_MSG}

        context_text = self._format_docs(docs)
        prompt_text = self.prompt_template.format(context=context_text, question=query)

        try:
            resp = self.llm.invoke(prompt_text)
            if hasattr(resp, "content"):
                text = resp.content.strip()
            elif isinstance(resp, dict) and "text" in resp:
                text = resp["text"].strip()
            else:
                text = str(resp).strip()
        except Exception as e:
            text = f"❌ Erro ao invocar o LLM: {e}"

        # Pequeno fallback: se por acaso o modelo devolver NOT_FOUND_MSG mas havia contexto,
        # mostramos um snippet cru.
        if text.strip() == NOT_FOUND_MSG and docs:
            snippet = context_text[:800]
            text = (
                "Encontrei algumas informações relacionadas na base de conhecimento:\n\n"
                f"{snippet}"
            )

        return {"result": text}


@st.cache_resource
def load_rag_chain():
    # 1) Embeddings – MESMO modelo do processar_documentos.py
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    # 2) Caminho do índice FAISS (mesma pasta do app.py)
    index_path = Path(__file__).parent / "faiss_index_neosense"

    # 2.1) Função auxiliar para (re)criar o índice a partir do TXT
    def build_index_from_txt() -> FAISS:
        # 👉 AJUSTE AQUI o caminho/nome do seu TXT, se for diferente
        txt_path = Path(__file__).parent / "docs" / "manuais_neosense.txt"

        if not txt_path.exists():
            st.error(
                "Arquivo de base de conhecimento 'docs/manuais_neosense.txt' "
                "não foi encontrado no repositório.\n\n"
                "Certifique-se de que ele está versionado no GitHub (e não ignorado no .gitignore)."
            )
            st.stop()

        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Quebra em chunks (simplificado, mas eficaz)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200,
            separators=["\n### ", "\n## ", "\n#", "\n\n", ".", " "],
        )
        docs = splitter.create_documents([full_text])

        vectorstore_local = FAISS.from_documents(docs, embeddings)
        # Salva o índice para próximos boots (ambiente atual)
        vectorstore_local.save_local(str(index_path))
        return vectorstore_local

    # 2.2) Tenta carregar o índice existente; se falhar, recria
    if index_path.exists():
        try:
            vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            # Erro clássico de pickle/pydantic/langchain -> recria
            st.warning(
                "Índice FAISS existente está incompatível com a versão atual das bibliotecas. "
                "Recriando índice a partir do TXT..."
            )
            shutil.rmtree(index_path, ignore_errors=True)
            vectorstore = build_index_from_txt()
    else:
        # Primeiro deploy / índice ausente
        vectorstore = build_index_from_txt()

    # 3) Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.45},
    )

    # 4) LLM para responder usando o contexto
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.25,
        top_p=0.9,
    )

    # 5) Prompt do RAG (mantive o seu, só copiado daqui)
    PROMPT = PromptTemplate(
        template="""
Você é o **Chatbot Neosense**, um assistente técnico especialista no sistema **Neosense CRM**.
Baseie sua resposta exclusivamente no conteúdo abaixo.
- Seja direto, objetivo e didático.
- Se houver referência clara a um módulo (ex: [Módulo: Missões]), mencione isso no começo da resposta.
- Se a resposta tiver mais de 10 linhas, finalize com: "💡 Em resumo:" seguido de um resumo claro.

---
FONTES DE CONHECIMENTO:
{context}
---
PERGUNTA:
{question}
---
RESPOSTA:
""",
        input_variables=["context", "question"],
    )

    return SimpleRAG(
        retriever=retriever,
        llm=llm,
        prompt_template=PROMPT,
        vectorstore=vectorstore,
    )


qa_chain = load_rag_chain()

# -----------------------
# 💬 INTERFACE DO USUÁRIO
# -----------------------
st.sidebar.header("Consultar ticket JIRA 🎫")
st.sidebar.caption("Insira o código ou link do ticket para obter contexto adicional.")
jira_ticket_input = st.sidebar.text_input(
    "Código ou link do ticket", placeholder="Ex: NEOSDEV-1234"
)

st.sidebar.divider()
with st.sidebar.expander("Dicas para criar boas perguntas"):
    st.markdown(
        """
1. **Seja específico:**
Ex: “Como criar campanha de desconto no ticket para todas as lojas?”

2. **Use termos do Neosense CRM:**
Nomes de módulos ou funções ajudam a precisão.

3. **Inclua contexto:**
Informe o código do ticket se a dúvida for sobre JIRA.

4. **Pergunte em sequência:**
Faça perguntas de acompanhamento quando necessário.
"""
    )

SHOW_DEBUG = False

if SHOW_DEBUG:
    with st.sidebar.expander("Debug RAG (base de conhecimento)", expanded=False):
        debug = st.session_state.get("rag_debug")
        if not debug:
            st.caption("Nenhuma consulta RAG registrada ainda.")
        else:
            st.markdown(f"**Última query:** `{debug.get('query', '')}`")
            st.markdown(f"**Documentos encontrados:** {debug.get('count', 0)}")
            if debug.get("top_score") is not None:
                st.markdown(f"**Score do 1º doc:** `{debug.get('top_score')}`")
            if debug.get("error"):
                st.markdown(f"**Erro:** `{debug['error']}`")
            if debug.get("snippet"):
                st.markdown("**Trecho do 1º documento:**")
                st.code(debug["snippet"])

# -----------------------
# TÍTULO E SUGESTÕES
# -----------------------
st.title("🤖 Chatbot Neosense")
st.caption("Assistente virtual da Neos Tecnologia.")

suggestions = [
    "O que é e como funciona a Agenda do Vendedor?",
    "Como recuperar a senha do aplicativo?",
    "Como criar campanha de desconto no ticket para público geral?",
    "Qual a diferença entre período de apuração e tempo do gasto de referência em missões?",
]
cols = st.columns(2)
for i, s in enumerate(suggestions):
    if cols[i % 2].button(s, use_container_width=True):
        st.session_state.chat_history = st.session_state.get("chat_history", [])
        st.session_state.chat_history.append({"role": "user", "content": s})
        st.session_state["pending_response"] = True
        st.rerun()

# -----------------------
# ESTADO DE CONVERSA
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_response" not in st.session_state:
    st.session_state.pending_response = False
if "jira_context" not in st.session_state:
    st.session_state.jira_context = None

# -----------------------
# RENDERIZAÇÃO DO HISTÓRICO
# -----------------------
for i, msg in enumerate(st.session_state.chat_history):
    role = msg["role"]
    content = msg.get("content", "")
    avatar_icon = "🧑" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar_icon):
        if msg.get("modo") == "redacao":
            st.text_area("📝 Rascunho de documentação:", content, height=400)
        else:
            st.markdown(content)

        if role == "assistant":
            doc_id = msg.get("doc_id")
            feedback_submitted = msg.get("feedback_submitted", False)

            if doc_id and not feedback_submitted:
                c1, c2, _ = st.columns([1, 1, 10])
                with c1:
                    if st.button("👍", key=f"up_{doc_id}"):
                        update_feedback_in_firestore(doc_id, "up")
                        st.session_state.chat_history[i]["feedback_submitted"] = True
                        st.rerun()
                with c2:
                    if st.button("👎", key=f"down_{doc_id}"):
                        update_feedback_in_firestore(doc_id, "down")
                        st.session_state.chat_history[i]["feedback_submitted"] = True
                        st.rerun()

            elif doc_id and feedback_submitted:
                st.caption("Feedback registrado. Obrigado!")

# -----------------------
# CAPTURA NOVA PERGUNTA
# -----------------------
if prompt := st.chat_input("Olá, sou o Neobot. Como posso te ajudar?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.pending_response = True
    st.rerun()

# -----------------------
# GERAÇÃO DE RESPOSTA
# -----------------------
if (
    st.session_state.pending_response
    and st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
):
    user_prompt = st.session_state.chat_history[-1]["content"]

    # Atualiza contexto JIRA se houver algo na sidebar
    if jira_ticket_input:
        st.session_state.jira_context = fetch_jira_data(
            extract_ticket_id_from_input(jira_ticket_input)
        )

    question_type = detect_question_type(user_prompt)
    rag_query = expand_query(user_prompt)

    # 1) MODO REDAÇÃO (prioritário)
    if detect_redacao_mode(user_prompt):
        with st.spinner("Gerando rascunho de documentação..."):
            llm_red = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.3,
                top_p=0.9,
            )

            contexto = qa_chain.invoke({"query": rag_query})
            rag_context = contexto.get("result", NOT_FOUND_MSG)

            redacao_prompt = REDACAO_PROMPT.format(
                jira_data=st.session_state.jira_context or "Nenhum ticket informado.",
                rag_context=rag_context,
            )

            doc_id = None
            try:
                resposta_modelo = llm_red.invoke(redacao_prompt)
                output_text = (
                    resposta_modelo.content.strip()
                    if hasattr(resposta_modelo, "content")
                    else str(resposta_modelo).strip()
                )
                doc_id = log_to_firestore(
                    pergunta=user_prompt,
                    resposta=output_text,
                    erro=None,
                    modo="redacao",
                    question_type=question_type,
                    ticket=st.session_state.jira_context,
                )
            except Exception as e:
                output_text = f"❌ Erro ao gerar rascunho: {e}"
                doc_id = log_to_firestore(
                    pergunta=user_prompt,
                    resposta=None,
                    erro=str(e),
                    modo="redacao",
                    question_type=question_type,
                    ticket=st.session_state.jira_context,
                )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": output_text,
                "doc_id": doc_id,
                "feedback_submitted": False,
                "modo": "redacao",
            }
        )
        st.session_state.pending_response = False
        st.rerun()

    # 2) FLUXO ESPECIAL: TICKET JIRA + RAG
    elif has_jira_context() and detect_jira_ticket_question(user_prompt):
        with st.spinner("Analisando ticket JIRA e base de conhecimento..."):
            llm_jira = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.25,
                top_p=0.9,
            )

            jira_data = st.session_state.jira_context

            try:
                contexto_rag = qa_chain.invoke({"query": rag_query})
                rag_context = contexto_rag.get("result", NOT_FOUND_MSG)
            except Exception as e:
                rag_context = f"Não foi possível recuperar contexto da base de conhecimento (erro: {e})."

            prompt_jira = (
                "Você é um analista de produto da Neos Tecnologia, trabalhando com o sistema Neosense CRM.\n\n"
                "Você receberá:\n"
                "1) Os dados REAIS (ou a tentativa de busca) de um ticket JIRA.\n"
                "2) Um resumo de contexto da base de conhecimento do Neosense CRM (RAG).\n\n"
                "Use SEMPRE as duas fontes para responder, deixando claro o que é:\n"
                "- funcionamento atual do sistema (base de conhecimento)\n"
                "- mudança proposta ou problema descrito no ticket (dados do ticket).\n\n"
                "Regras IMPORTANTES:\n"
                "- Se a pergunta usar expressões como 'segundo o ticket', 'de acordo com o ticket', "
                "'neste ticket', 'deste ticket' ou perguntar o valor de um campo específico, "
                "RESPONDA SOMENTE com base em 'DADOS DO TICKET JIRA'.\n"
                "- Se a informação não estiver claramente descrita nos 'DADOS DO TICKET JIRA', diga que "
                "o ticket não traz essa informação. Não tente inferir a partir do contexto RAG.\n\n"
                "Tarefas principais:\n"
                "1) Se a pergunta pedir RESUMO do ticket, resuma:\n"
                "- contexto\n- problema/necessidade\n- solução proposta\n"
                "- principais regras de negócio\n- impactos no sistema ou no usuário final.\n\n"
                "2) Se a pergunta citar 'plano de teste', descreva um plano de teste em alto nível:\n"
                "- objetivos do teste\n- principais cenários\n- exemplos de casos de teste\n"
                "- critérios de aceite.\n\n"
                "3) Se a pergunta falar de 'antes' e 'depois', explique:\n"
                "- como o sistema funciona hoje (ANTES), com base no CONTEXTO RAG;\n"
                "- o que muda (DEPOIS), com base nos DADOS DO TICKET.\n\n"
                "Responda sempre em português, em tópicos, e não invente funcionalidades "
                "que não estejam no ticket ou na base de conhecimento.\n\n"
                "--- CONTEXTO DA BASE DE CONHECIMENTO (RAG) ---\n"
                f"{rag_context}\n\n"
                "--- DADOS DO TICKET JIRA ---\n"
                f"{jira_data}\n\n"
                "--- PERGUNTA DO USUÁRIO ---\n"
                f"{user_prompt}\n\n"
                "Agora responda:\n"
            )

            doc_id = None
            try:
                resposta_modelo = llm_jira.invoke(prompt_jira)
                output_text = (
                    resposta_modelo.content.strip()
                    if hasattr(resposta_modelo, "content")
                    else str(resposta_modelo).strip()
                )
                doc_id = log_to_firestore(
                    pergunta=user_prompt,
                    resposta=output_text,
                    erro=None,
                    modo="normal",
                    question_type=question_type,
                    ticket=st.session_state.jira_context,
                )
            except Exception as e:
                output_text = (
                    f"❌ Erro ao responder usando ticket + base de conhecimento: {e}"
                )
                doc_id = log_to_firestore(
                    pergunta=user_prompt,
                    resposta=None,
                    erro=str(e),
                    modo="normal",
                    question_type=question_type,
                    ticket=st.session_state.jira_context,
                )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": output_text,
                "doc_id": doc_id,
                "feedback_submitted": False,
                "modo": "normal",
            }
        )
        st.session_state.pending_response = False
        st.rerun()

    # 3) MODO NORMAL (apenas base de conhecimento)
    else:
        doc_id = None
        with st.spinner("Buscando na base de conhecimento..."):
            try:
                resposta_modelo = qa_chain.invoke({"query": rag_query})
                output_text = resposta_modelo.get("result", NOT_FOUND_MSG).strip()

                if output_text.strip() == NOT_FOUND_MSG:
                    log_unanswered(user_prompt, rag_query)

                doc_id = log_to_firestore(
                    pergunta=user_prompt,
                    resposta=output_text,
                    erro=None,
                    modo="normal",
                    question_type=question_type,
                    ticket=st.session_state.jira_context,
                )

            except Exception as e:
                output_text = f"❌ Erro ao gerar resposta: {e}"
                doc_id = log_to_firestore(
                    pergunta=user_prompt,
                    resposta=None,
                    erro=str(e),
                    modo="normal",
                    question_type=question_type,
                    ticket=st.session_state.jira_context,
                )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": output_text,
                "doc_id": doc_id,
                "feedback_submitted": False,
                "modo": "normal",
            }
        )
        st.session_state.pending_response = False
        st.rerun()
