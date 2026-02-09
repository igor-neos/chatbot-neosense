# app.py - Versão 5.3 (Correção de Estado Limpo e Logs Completos)
# TESTE

if st.secrets.get("BUILD_FAISS") == "true":
    from build_faiss import *
    st.success("FAISS gerado. Remova BUILD_FAISS.")
    st.stop()
# TESTE

import os
import ssl
import certifi
import re
import warnings
from pathlib import Path
from datetime import datetime
import streamlit as st
import shutil

# --- Supressão de avisos ---
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# --- Import LangChain ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Correção SSL ---
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

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
# 🔥 FIREBASE
# -----------------------
try:
    if not firebase_admin._apps:
        cred = None
        cred_path_local = Path(__file__).parent / "firebase_credentials.json"
        if cred_path_local.exists():
            cred = credentials.Certificate(str(cred_path_local))
        else:
            try:
                cred_dict = dict(st.secrets["firebase_creds"])
                pk = cred_dict.get("private_key", "")
                if "\\n" in pk:
                    cred_dict["private_key"] = pk.replace("\\n", "\n").strip()
                cred = credentials.Certificate(cred_dict)
            except Exception:
                st.error("Credenciais do Firebase não encontradas.")
                st.stop()
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    st.error(f"Erro ao conectar ao Firebase: {e}")
    st.stop()

# -----------------------
# 🔥 LOGGING & FEEDBACK
# -----------------------
def log_to_firestore(pergunta, resposta=None, erro=None, modo="normal", question_type=None, ticket=None, timestamp_inicio=None):
    try:
        agora = datetime.now()
        inicio = timestamp_inicio if timestamp_inicio else agora
        duracao = (agora - inicio).total_seconds()

        data = {
            "timestamp_pergunta": inicio.isoformat(),
            "timestamp_resposta": agora.isoformat(),
            "duracao_segundos": duracao,
            "pergunta": pergunta,
            "resposta": resposta,
            "erro": erro,
            "feedback": None,
            "modo": modo,
            "question_type": question_type,
            "ticket": ticket, # Grava o ticket (ou None se estiver vazio)
        }
        _, doc_ref = db.collection("chat_logs").add(data)
        return doc_ref.id
    except Exception as e:
        return None

def update_feedback_in_firestore(doc_id, feedback):
    try:
        db.collection("chat_logs").document(doc_id).update({"feedback": feedback})
    except Exception:
        pass

# -----------------------
# 🧠 FUNÇÕES GERAIS
# -----------------------
NOT_FOUND_MSG = "Não encontrei essa informação na base de conhecimento."

def detect_question_type(q: str) -> str:
    q = q.lower()
    if "diferença" in q or "diferenca" in q: return "comparação"
    if "relação" in q or "ligação" in q: return "relacional"
    if q.startswith("como "): return "procedimento"
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
        with open(logs_dir / "perguntas_nao_respondidas.csv", "a", encoding="utf-8") as f:
            ts = datetime.now().isoformat()
            f.write(f'"{ts}","{question}","{aug_query}"\n')
    except Exception: pass

# -----------------------
# 🎫 Integração com JIRA
# -----------------------
def extract_ticket_id_from_input(text_input):
    if not text_input: return None
    match = re.search(r"([A-Z]+-[0-9]+)", text_input.upper())
    return match.group(1) if match else None

@st.cache_data(ttl=600)
def fetch_jira_data(ticket_id):
    try:
        jira_options = {"server": st.secrets["JIRA_SERVER"]}
        jira = JIRA(options=jira_options, basic_auth=(st.secrets["JIRA_USERNAME"], st.secrets["JIRA_API_TOKEN"]))
        issue = jira.issue(ticket_id)
        return (
            f"Dados do Ticket JIRA: {ticket_id}\n"
            f"- Título: {issue.fields.summary}\n"
            f"- Status: {issue.fields.status.name}\n"
            f"- Descrição: {issue.fields.description if issue.fields.description else 'Sem descrição.'}"
        )
    except Exception as e:
        return f"Erro ao buscar o ticket {ticket_id}: {e}"

# -----------------------
# ✍️ MODO REDAÇÃO
# -----------------------
def detect_redacao_mode(prompt: str) -> bool:
    p = prompt.lower()
    gatilhos = ["crie um rascunho", "documente este ticket", "gerar documentação", "escreva a documentação"]
    if any(g in p for g in gatilhos): return True
    if "rascunho" in p and ("documenta" in p or "ticket" in p or "funcionalidade" in p): return True
    return False

REDACAO_PROMPT = """
Você é um redator técnico sênior da equipe de documentação da Neos Tecnologia.
Transforme o conteúdo técnico em texto claro, objetivo e escaneável para o GitBook interno do Neosense CRM.

Regras de escrita:
- Use títulos curtos e objetivos.
- Mantenha tom técnico, direto e consistente.
- Estruture nas seções: Introdução, Configuração, Funcionamento e 💡 Dica/Observação.
- Combine o módulo e o título principal em uma única linha, no formato:
  **[Módulo: NomeDoMódulo] Nome da funcionalidade**

--- DADOS DO TICKET (fonte primária):
{jira_data}

--- CONTEXTO DA BASE DE CONHECIMENTO (RAG):
{rag_context}

Agora escreva a documentação técnica final:
"""

def has_jira_context() -> bool:
    return bool(st.session_state.get("jira_context"))

# -----------------------
# 📚 RAG (FAISS + MMR)
# -----------------------
class SimpleRAG:
    def __init__(self, retriever, llm, prompt_template: PromptTemplate, vectorstore=None):
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt_template = prompt_template

        # Quanto MENOR, mais similar (FAISS)
        self.max_acceptable_score = 0.85  

        self.condense_prompt = PromptTemplate.from_template(
            """Dada a conversa a seguir e uma pergunta de acompanhamento, reescreva a pergunta para que seja independente, mantendo o contexto original.
            Histórico: {chat_history}
            Pergunta: {question}
            Pergunta Independente:"""
        )

    def _format_docs(self, docs):
        pieces = []
        for d in docs:
            content = d.page_content if hasattr(d, "page_content") else str(d)
            produto = d.metadata.get("produto", "")
            caminho = d.metadata.get("caminho_logico", "")
            
            fonte_display = ""
            if produto and caminho:
                fonte_display = f"\n🔍 Fonte: {produto} > {caminho}"
            elif d.metadata.get("source"):
                fonte_display = f"\n🔍 Fonte: {d.metadata.get('source')}"
            
            pieces.append(f"{content}{fonte_display}")
        return "\n\n".join(pieces)

    def _format_chat_history(self, history):
        buffer = []
        for msg in history[-4:]:
            role = "Humano" if msg["role"] == "user" else "Assistente"
            buffer.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(buffer)

    def invoke(self, inputs: dict):
        query = inputs.get("query") if isinstance(inputs, dict) else inputs
        chat_history_raw = inputs.get("chat_history", []) if isinstance(inputs, dict) else []

        search_query = query
        if chat_history_raw:
            try:
                history_text = self._format_chat_history(chat_history_raw)
                condense_input = self.condense_prompt.format(
                    chat_history=history_text, question=query
                )
                rephrased = self.llm.invoke(condense_input)
                search_query = (
                    rephrased.content.strip()
                    if hasattr(rephrased, "content")
                    else str(rephrased)
                )
            except Exception:
                pass

        try:
            results = None
            if self.vectorstore:
                results = self.vectorstore.similarity_search_with_score(
                    search_query, k=10
                )
            elif self.retriever:
                results = self.retriever.similarity_search_with_score(
                    search_query, k=10
                )

            if not results:
                return {"result": NOT_FOUND_MSG}

            # Filtra apenas chunks realmente relevantes
            filtered_docs = [
                doc for doc, score in results if score <= self.max_acceptable_score
            ]

            if not filtered_docs:
                return {"result": NOT_FOUND_MSG}

            # Debug opcional
            try:
                st.session_state["rag_debug"] = {
                    "query": search_query,
                    "count": len(filtered_docs),
                    "best_score": min(score for _, score in results),
                    "snippet": filtered_docs[0].page_content[:400],
                }
            except Exception:
                pass

            context_text = self._format_docs(filtered_docs)
            prompt_text = self.prompt_template.format(
                context=context_text, question=query
            )

            resp = self.llm.invoke(prompt_text)
            text = (
                resp.content.strip()
                if hasattr(resp, "content")
                else str(resp).strip()
            )

        except Exception as e:
            text = f"❌ Erro ao processar resposta: {e}"

        return {"result": text}


@st.cache_resource
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    index_path = Path(__file__).parent / "faiss_index_neosense"

    def build_index_empty():
        st.warning("Índice FAISS não encontrado. Por favor, faça o deploy da pasta 'faiss_index_neosense'.")
        return FAISS.from_texts(["O sistema está sem base de conhecimento."], embeddings)

    if index_path.exists():
        try:
            vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Erro ao carregar FAISS: {e}")
            raise
    else:
        vectorstore = build_index_empty()

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 40, "lambda_mult": 0.5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2, top_p=0.9)

    PROMPT = PromptTemplate(
        template="""
        
Você é o **Chatbot Neosense**, um assistente técnico especialista no ecossistema **Neosense CRM**.
Você tem acesso à documentação de: CRM (Redesign e Legado), Aplicativo, E-commerce, Agenda do Vendedor e Portal Web.
Baseie sua resposta exclusivamente no conteúdo abaixo.
- Seja direto, objetivo e didático.
- Se houver referência clara a um módulo (ex: [Módulo: Missões]), mencione isso no começo da resposta.
- Se a resposta tiver mais de 10 linhas, finalize com: "💡 Em resumo:" seguido de um resumo claro.
4. Responda **somente** com base na documentação fornecida.
5. Se a pergunta não tiver relação com os sistemas da Neos Tecnologia
ou não houver informação suficiente no contexto, responda exatamente: "Não encontrei essa informação na base de conhecimento."

--- CONTEXTO ENCONTRADO ---
{context}
---------------------------

PERGUNTA: {question}
RESPOSTA:
""",
        input_variables=["context", "question"],
    )

    return SimpleRAG(retriever=retriever, llm=llm, prompt_template=PROMPT, vectorstore=vectorstore)


qa_chain = load_rag_chain()

# -----------------------
# 💬 INTERFACE DO USUÁRIO
# -----------------------
st.sidebar.header("Consultar ticket JIRA 🎫")
jira_ticket_input = st.sidebar.text_input("Insira o código ou link do ticket e tecle **ENTER**", placeholder="Ex: NEOSDEV-1234")

st.sidebar.divider()
with st.sidebar.expander("Dicas para criar boas perguntas"):
    st.markdown("""
1. **Seja específico:**
    Ex: “Como criar campanha de desconto no ticket para público geral em todas as lojas?”
2. **Use termos do ambiente Neosense:**
    Nomes de módulos e funcionalidades ajudam na precisão.
3. **Inclua contexto JIRA:**
    Informe o código do ticket se a dúvida for sobre itens do JIRA.
4. **Pergunte em sequência:**
    Faça perguntas de acompanhamento.
""")

st.title("🤖 Chatbot Neosense")
st.caption("Assistente virtual do ambiente Neosense.")

suggestions = [
    "O que é e como funciona a Agenda do Vendedor?",
    "Como recuperar a senha do aplicativo?",
    "Como criar campanha de desconto no ticket para público geral?",
    "Qual a diferença entre período de apuração e tempo do gasto de referência em missões?",
]
cols = st.columns(2)
for i, s in enumerate(suggestions):
    if cols[i % 2].button(s, use_container_width=True):
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": s})
        st.session_state["pending_response"] = True
        st.rerun()

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "pending_response" not in st.session_state: st.session_state.pending_response = False
if "jira_context" not in st.session_state: st.session_state.jira_context = None

for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        if msg.get("modo") == "redacao":
            st.text_area("📝 Rascunho:", msg.get("content", ""), height=400)
        else:
            st.markdown(msg.get("content", ""))

        if msg["role"] == "assistant":
            doc_id = msg.get("doc_id")
            feedback_submitted = msg.get("feedback_submitted", False)
            if doc_id and not feedback_submitted:
                c1, c2, _ = st.columns([1, 1, 10])
                if c1.button("👍", key=f"up_{i}"):
                    update_feedback_in_firestore(doc_id, "up")
                    st.session_state.chat_history[i]["feedback_submitted"] = True
                    st.rerun()
                if c2.button("👎", key=f"down_{i}"):
                    update_feedback_in_firestore(doc_id, "down")
                    st.session_state.chat_history[i]["feedback_submitted"] = True
                    st.rerun()
            elif doc_id and feedback_submitted:
                st.caption("Obrigado pelo feedback!")

if prompt := st.chat_input("Olá, eu sou o Neobot. Como posso te ajudar?"):
    st.session_state.start_time = datetime.now()
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.pending_response = True
    st.rerun()

# -----------------------
# GERAÇÃO DE RESPOSTA
# -----------------------
if st.session_state.pending_response and st.session_state.chat_history[-1]["role"] == "user":
    user_prompt = st.session_state.chat_history[-1]["content"]

    # --- CORREÇÃO DE ESTADO (V5.3): Limpeza explícita ---
    if jira_ticket_input:
        st.session_state.jira_context = fetch_jira_data(extract_ticket_id_from_input(jira_ticket_input))
    else:
        st.session_state.jira_context = None # Garante que está limpo se o input estiver vazio
    # ---------------------------------------------------

    question_type = detect_question_type(user_prompt)
    rag_query = expand_query(user_prompt)

    # 1) MODO REDAÇÃO (Prioridade Máxima)
    if detect_redacao_mode(user_prompt):
        doc_id = None
        with st.spinner("Gerando rascunho de documentação..."):
            llm_red = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.3)
            contexto = qa_chain.invoke({"query": rag_query, "chat_history": st.session_state.chat_history[:-1]})
            rag_context = contexto.get("result", NOT_FOUND_MSG)
            
            try:
                resp = llm_red.invoke(REDACAO_PROMPT.format(jira_data=st.session_state.jira_context or "N/A", rag_context=rag_context))
                output_text = resp.content.strip()
                # --- CORREÇÃO DE LOG (V5.3): Adicionado ticket ---
                doc_id = log_to_firestore(user_prompt, output_text, modo="redacao", ticket=st.session_state.jira_context, question_type=question_type, timestamp_inicio=st.session_state.get("start_time"))
                # -------------------------------------------------
            except Exception as e:
                output_text = f"❌ Erro: {e}"
        
        st.session_state.chat_history.append({"role": "assistant", "content": output_text, "doc_id": doc_id, "modo": "redacao"})
        st.session_state.pending_response = False
        st.rerun()

    # 2) TICKET JIRA + RAG (MODO HÍBRIDO)
    elif has_jira_context():
        doc_id = None
        with st.spinner("Analisando ticket JIRA e base de conhecimento..."):
            llm_jira = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.25)
            
            try:
                ctx = qa_chain.invoke({"query": rag_query, "chat_history": st.session_state.chat_history[:-1]})
                rag_context = ctx.get("result", NOT_FOUND_MSG)
            except: rag_context = "Erro ao buscar contexto."
            
            prompt_jira = f"""
            Você é um Analista Técnico Sênior do Neosense CRM.
            
            FONTES DE INFORMAÇÃO:
            1. 🎫 DADOS DO TICKET (A verdade sobre o que é NOVO ou MUDANÇA):
            {st.session_state.jira_context}
            
            2. 📚 BASE DE CONHECIMENTO (A verdade sobre o que JÁ EXISTE):
            {rag_context}
            
            ---
            INSTRUÇÕES:
            - Sua missão é responder a pergunta do usuário combinando essas duas fontes.
            - Se a pergunta for sobre algo novo (ex: "Como funciona a nova segmentação?"), USE OS DADOS DO TICKET como fonte principal.
            - Se a pergunta for geral, use a Base de Conhecimento.
            - NÃO exiba trechos de código cru, JSON ou tags como {{% hint %}}. Formate tudo em Markdown limpo.
            - Se o ticket trouxer regras de negócio (ex: "Prazo 90 dias"), destaque isso na resposta.
            
            PERGUNTA DO USUÁRIO:
            {user_prompt}
            """
            
            try:
                resp = llm_jira.invoke(prompt_jira)
                output_text = resp.content.strip()
                doc_id = log_to_firestore(user_prompt, output_text, modo="normal", ticket=st.session_state.jira_context, question_type=question_type, timestamp_inicio=st.session_state.get("start_time"))
            except Exception as e:
                output_text = f"❌ Erro: {e}"
                
        st.session_state.chat_history.append({"role": "assistant", "content": output_text, "doc_id": doc_id, "modo": "normal"})
        st.session_state.pending_response = False
        st.rerun()

    # 3) MODO NORMAL (Apenas Base de Conhecimento)
    else:
        with st.spinner("Buscando na base de conhecimento..."):
            try:
                resp = qa_chain.invoke({"query": rag_query, "chat_history": st.session_state.chat_history[:-1]})
                output_text = resp.get("result", NOT_FOUND_MSG).strip()
                
                if "hint style" in output_text or "{%" in output_text:
                     output_text = output_text.replace("{% hint style=\"info\" %}", "> ℹ️ ").replace("{% endhint %}", "")

                if output_text == NOT_FOUND_MSG: log_unanswered(user_prompt, rag_query)
                
                doc_id = log_to_firestore(user_prompt, output_text, modo="normal", question_type=question_type, timestamp_inicio=st.session_state.get("start_time"))
            except Exception as e:
                output_text = f"❌ Erro: {e}"
                doc_id = None
        
        st.session_state.chat_history.append({"role": "assistant", "content": output_text, "doc_id": doc_id, "modo": "normal"})
        st.session_state.pending_response = False
        st.rerun()