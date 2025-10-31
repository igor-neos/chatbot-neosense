# app.py - Versão 3.4.5

import os
import ssl
import certifi
import re
from pathlib import Path
from datetime import datetime
import streamlit as st

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
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
# 🔥 FIREBASE (corrigido)
# -----------------------
try:
    if not firebase_admin._apps:
        # Lê o bloco [firebase_creds] do secrets.toml e converte para dict
        cred_dict = dict(st.secrets["firebase_creds"])

        # 🔧 Corrige as quebras de linha escapadas na chave privada
        if "private_key" in cred_dict and isinstance(cred_dict["private_key"], str):
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

        # Inicializa o Firebase com o dicionário corrigido
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)

    db = firestore.client()

except Exception as e:
    st.error(f"Erro ao conectar ao Firebase: {e}")
    st.stop()


# -----------------------
# 🔥 LOGGING
# -----------------------
def log_to_firestore(pergunta, resposta=None, erro=None, modo="normal", question_type=None, ticket=None):
    """
    Cria um documento em chat_logs e retorna o doc_id.
    """
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
        # CORREÇÃO: Firestore .add() -> (update_time, document_reference)
        _ , doc_ref = db.collection("chat_logs").add(data)
        return doc_ref.id
    except Exception as e:
        st.warning(f"⚠️ Erro ao salvar log no Firestore: {e}")
        return None

def update_feedback_in_firestore(doc_id, feedback):
    """
    Atualiza o campo 'feedback' de um log existente (👍 ou 👎).
    """
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

# Expansão semântica para melhorar recall
SYN_MAP = {
    r"\bsenha(s)?\b": "senha login acesso recuperar credenciais esqueci senha token",
    r"\blogin\b": "login acesso autenticação entrar conectar token",
    r"\bcampanha(s)?\b": "campanha ofertas descontos promoção marketing criar editar configurar vantagem público geral",
    r"\bmiss(ões|ao)\b": "missões metas período apuração gasto referência pontuação objetivo missão fidelidade incremento",
    r"\bagenda do vendedor\b": "agenda do vendedor carteira clientes contato campanhas aplicativo vendedor dashboard vendas material apoio",
    r"\bbeneficio(s)?\b": "benefício recompensa catálogo missão criar editar adicionar prêmio catálogo de benefícios fidelidade",
    r"\bpré[- ]cadastro\b": "pré-cadastro pre cadastro cadastro inicial cliente parcial loja oferta lead",
    r"\bcatálogo\b": "catálogo benefícios recompensas prêmios lista registrar adicionar item resgate disponível utilizado",
    r"\bpdv\b": "pdv integração ponto de venda loja código externo id_externo_organizacao saldo recompensa imprimir cupom",
}

def expand_query(user_query: str) -> str:
    expanded = user_query
    for patt, exp in SYN_MAP.items():
        if re.search(patt, user_query, flags=re.IGNORECASE):
            expanded += f" ({exp})"
    return expanded

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# CORREÇÃO: Função log_unanswered corrigida para evitar SyntaxError
def log_unanswered(question: str, aug_query: str):
    try:
        logs_dir = Path("logs")
        ensure_dir(logs_dir)
        with open(
            logs_dir / "perguntas_nao_respondidas.csv",
            "a",
            encoding="utf-8",
        ) as f:
            ts = datetime.now().isoformat()
            
            # 1. PreparamOS as variáveis ANTES da f-string
            question_limpa = question.replace("\"", "'")
            aug_query_limpa = aug_query.replace("\"", "'")
            
            # 2. A f-string agora é simples e limpa
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
    """
    Retorna uma string formatada com os dados principais do ticket JIRA.
    Se falhar, retorna o erro como string (para não quebrar a IA).
    """
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
    return any(g in prompt.lower() for g in gatilhos)

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
# 📚 RAG (FAISS + MMR)
# -----------------------
@st.cache_resource
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )

    vectorstore = FAISS.load_local(
        "faiss_index_neosense",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.45},
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-preview-05-20", # Mantendo o modelo da v3.4.4
        google_api_key=api_key,
        temperature=0.25,
        top_p=0.9,
    )

    PROMPT = PromptTemplate(
        template="""
Você é o **Chatbot Neosense**, um assistente técnico especialista no sistema **Neosense CRM**.
Baseie sua resposta exclusivamente no conteúdo abaixo.
- Seja direto, objetivo e didático.
- Se houver referência clara a um módulo (ex: [Módulo: Missões]), mencione isso no começo da resposta.
- Se a resposta tiver mais de 10 linhas, finalize com: "💡 Em resumo:" seguido de um resumo claro.
- Se não houver informação, responda exatamente:
  "Não encontrei essa informação na base de conhecimento."

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

    qa_chain_local = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain_local

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
1. **Seja específico:** Ex: “Como criar campanha de desconto no ticket para todas as lojas?”
2. **Use termos do Neosense CRM:** Nomes de módulos ou funções ajudam a precisão.
3. **Inclua contexto:** Informe o código do ticket se a dúvida for sobre JIRA.
4. **Pergunte em sequência:** Faça perguntas de acompanhamento quando necessário.
"""
    )

st.title("🤖 Chatbot Neosense")
st.caption("Assistente virtual da Neos Tecnologia.")

# sugestões fixas de perguntas
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
        # CORREÇÃO TESTE: Usar o novo formato de histórico (dicionário)
        st.session_state.chat_history.append({"role": "user", "content": s})
        st.session_state["pending_response"] = True
        st.rerun()

# -----------------------
# ESTADO DE CONVERSA
# -----------------------
# CORREÇÃO TESTE: Inicializa o histórico como lista de dicionários
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_response" not in st.session_state:
    st.session_state.pending_response = False
if "jira_context" not in st.session_state:
    st.session_state.jira_context = None

# -----------------------------------------------------------
# CORREÇÃO TESTE: Renderização do histórico com botões persistentes
# -----------------------------------------------------------
for i, msg in enumerate(st.session_state.chat_history):
    role = msg["role"]
    content = msg.get("content", "") # Pega o 'content' do dicionário
    avatar_icon = "🧑" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar_icon):
        # Lógica especial para exibir rascunho de redação
        if msg.get("modo") == "redacao":
            st.text_area("📝 Rascunho de documentação:", content, height=400)
        else:
            st.markdown(content)

        # SE FOR ASSISTENTE, RENDERIZAR BOTÕES DE FEEDBACK
        if role == "assistant":
            doc_id = msg.get("doc_id")
            feedback_submitted = msg.get("feedback_submitted", False)
            
            # Só mostra os botões se o doc_id existir E o feedback ainda não foi enviado
            if doc_id and not feedback_submitted:
                c1, c2, _ = st.columns([1, 1, 10]) # Colunas para botões pequenos
                with c1:
                    # Usamos o doc_id como parte da chave para torná-la única
                    if st.button("👍", key=f"up_{doc_id}"):
                        update_feedback_in_firestore(doc_id, "up")
                        # Marca que o feedback foi enviado para esta mensagem
                        st.session_state.chat_history[i]["feedback_submitted"] = True
                        st.rerun() # Reroda para os botões sumirem
                with c2:
                    if st.button("👎", key=f"down_{doc_id}"):
                        update_feedback_in_firestore(doc_id, "down")
                        # Marca que o feedback foi enviado para esta mensagem
                        st.session_state.chat_history[i]["feedback_submitted"] = True
                        st.rerun() # Reroda para os botões sumirem
            
            # Se o feedback já foi dado, mostra uma confirmação
            elif doc_id and feedback_submitted:
                st.caption("Feedback registrado. Obrigado!")

# captura nova pergunta
if prompt := st.chat_input("Olá, sou o Neobot. Como posso te ajudar?"):
    # CORREÇÃO TESTE: Usar o novo formato de histórico (dicionário)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.pending_response = True
    st.rerun()

# -----------------------
# GERAÇÃO DE RESPOSTA
# -----------------------
if (
    st.session_state.pending_response
    and st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user" # Verificação de Dicionário
):
    user_prompt = st.session_state.chat_history[-1]["content"] # Pega 'content' do Dicionário
    full_prompt = user_prompt

    # contexto JIRA (aproveita sempre o mais recente)
    if jira_ticket_input:
        st.session_state.jira_context = fetch_jira_data(
            extract_ticket_id_from_input(jira_ticket_input)
        )

    if st.session_state.jira_context:
        full_prompt = (
            f"Com base nos dados REAIS do ticket JIRA: "
            f"'{st.session_state.jira_context}', responda: '{user_prompt}'"
        )

    question_type = detect_question_type(user_prompt)
    augmented_query = expand_query(full_prompt)

    # ---------------------------------
    # MODO REDAÇÃO
    # ---------------------------------
    if detect_redacao_mode(user_prompt):
        llm_red = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-preview-05-20", # Mantendo o modelo da v3.4.4
            google_api_key=api_key,
            temperature=0.3,
            top_p=0.9,
        )

        contexto = qa_chain.invoke({"query": user_prompt})
        rag_context = contexto.get("result", NOT_FOUND_MSG)

        redacao_prompt = REDACAO_PROMPT.format(
            jira_data=st.session_state.jira_context
            or "Nenhum ticket informado.",
            rag_context=rag_context,
        )

        doc_id = None # Inicializa doc_id
        try:
            resposta_modelo = llm_red.invoke(redacao_prompt)
            output_text = resposta_modelo.content.strip()
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

        # CORREÇÃO TESTE: Adiciona ao histórico no formato dicionário
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": output_text,
            "doc_id": doc_id,
            "feedback_submitted": False,
            "modo": "redacao" # Flag especial para renderizar como text_area
        })

        st.session_state.pending_response = False
        st.rerun() # Reroda para exibir a resposta e os botões

    # ---------------------------------
    # MODO NORMAL (Pergunta & Resposta)
    # ---------------------------------
    else:
        doc_id = None # Inicializa doc_id
        # Removemos o 'with st.chat_message...' daqui, 
        # pois o loop principal vai renderizar
        with st.spinner("Buscando na base de conhecimento..."):
            try:
                resposta_modelo = qa_chain.invoke({"query": augmented_query})
                output_text = resposta_modelo.get("result", NOT_FOUND_MSG).strip()

                if output_text.strip() == NOT_FOUND_MSG:
                    log_unanswered(user_prompt, augmented_query)

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

            # CORREÇÃO TESTE: Adiciona ao histórico no formato dicionário
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": output_text,
                "doc_id": doc_id,
                "feedback_submitted": False,
                "modo": "normal"
            })

        st.session_state.pending_response = False
        st.rerun() # Reroda para exibir a resposta e os botões