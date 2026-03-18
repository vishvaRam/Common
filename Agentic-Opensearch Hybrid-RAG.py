# hybrid_rag_agent.py
# pip install langchain langchain-community langgraph langchain-openai opensearch-py PyMuPDF
#%%
import os
import glob
from typing import Annotated, Literal, List

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OPENSEARCH_URL   = os.getenv("OPENSEARCH_URL",  "http://192.9.200.28:9200")
OPENSEARCH_USER  = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS  = os.getenv("OPENSEARCH_PASS", "admin")

LLM_MODEL        = "Qwen3.5-9B-AutoRound-MTP"
LLM_BASE_URL     = "http://192.9.200.28:8525/v1"

EMBEDDING_MODEL    = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BASE_URL = "http://192.9.200.29:1234/v1"

PDF_DIR          = "./pdfs"
INDEX_NAME       = "qwen_document_index"
PIPELINE_NAME    = "hybrid_rag_pipeline"
TOP_K            = 15
#%%
# ─────────────────────────────────────────────
# SHARED OPENSEARCH KWARGS (reused everywhere)
# ─────────────────────────────────────────────
OS_KWARGS = dict(
    opensearch_url=OPENSEARCH_URL,
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=EMBEDDING_BASE_URL,
    api_key="EMPTY",
)

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key="EMPTY",
    temperature=0.1,
    max_completion_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
#%%
# ─────────────────────────────────────────────
# ROUTING SCHEMA
# ─────────────────────────────────────────────
class RouteDecision(BaseModel):
    """Route query to retrieval or direct LLM answer."""
    route: Literal["retrieve", "direct"] = Field(
        description=(
            "'retrieve' — query needs information from uploaded documents. "
            "'direct'   — general knowledge, greetings, simple facts."
        )
    )
    reasoning: str = Field(description="One-line explanation.")

router_llm = llm.with_structured_output(RouteDecision)

# ─────────────────────────────────────────────
# PDF INGESTION
# ─────────────────────────────────────────────
def load_pdfs(pdf_dir: str) -> List[Document]:
    docs = []
    pdf_files = list(set(
        glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True) +
        glob.glob(os.path.join(pdf_dir, "*.pdf"))
    ))
    if not pdf_files:
        print(f"[WARN] No PDFs found in '{pdf_dir}'")
        return docs

    for path in pdf_files:
        print(f"[INFO] Loading: {path}")
        pdf = fitz.open(path)
        for page_num, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(path),
                        "page": page_num + 1,
                    },
                ))
        pdf.close()

    print(f"[INFO] Loaded {len(docs)} pages from {len(pdf_files)} PDF(s).")
    return docs

#%%
def ingest_pdfs(pdf_dir: str) -> OpenSearchVectorSearch:
    """Load PDFs → chunk → index into OpenSearch. Returns connected vectorstore."""
    raw_docs = load_pdfs(pdf_dir)

    if not raw_docs:
        print("[WARN] No docs to ingest. Connecting to existing index.")
        return connect_vectorstore()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[INFO] Indexing {len(chunks)} chunks into '{INDEX_NAME}' …")

    # from_documents embeds + bulk-indexes all chunks
    vectorstore = OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        engine="lucene",          # lucene supports hybrid search
        **OS_KWARGS,
    )
    print(f"[INFO] Indexed {len(chunks)} chunks successfully.")
    return vectorstore


def connect_vectorstore() -> OpenSearchVectorSearch:
    """Connect to an existing OpenSearch index (no re-ingest)."""
    return OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        **OS_KWARGS,
    )

#%%
# ─────────────────────────────────────────────
# HYBRID SEARCH PIPELINE SETUP
# ─────────────────────────────────────────────
def setup_hybrid_pipeline(vectorstore: OpenSearchVectorSearch):
    """
    Create the normalization-processor search pipeline via LangChain's
    built-in configure_search_pipelines() — no manual JSON needed.
    keyword_weight=0.3 (BM25), vector_weight=0.7 (kNN).
    """
    try:
        vectorstore.configure_search_pipelines(
            pipeline_name=PIPELINE_NAME,
            keyword_weight=0.3,
            vector_weight=0.7,
        )
        print(f"[INFO] Hybrid pipeline '{PIPELINE_NAME}' configured.")
    except Exception as e:
        print(f"[WARN] Pipeline setup skipped (may already exist): {e}")


# ─────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────
def build_hybrid_retriever(vectorstore: OpenSearchVectorSearch):
    """
    Use OpenSearchVectorSearch native hybrid search:
      search_type='hybrid_search'  → fires BM25 + kNN in one query
      search_pipeline              → applies normalization + score merging server-side
    No EnsembleRetriever, no in-memory BM25 needed.
    """
    return vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K,
            "search_type": "hybrid_search",
            "search_pipeline": PIPELINE_NAME,
        }
    )

#%%
# ─────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────
class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    query: str = ""
    route: str = ""
    retrieved_docs: List[Document] = Field(default_factory=list)
    answer: str = ""

    class Config:
        arbitrary_types_allowed = True

#%%
# ─────────────────────────────────────────────
# GRAPH NODES
# ─────────────────────────────────────────────
def router_node(state: AgentState) -> dict:
    print(f"\n[ROUTER] Query: {state.query!r}")
    decision: RouteDecision = router_llm.invoke([
        SystemMessage(content=(
            "You are a routing assistant. Decide if the user query needs searching "
            "internal uploaded documents, or can be answered from general knowledge. "
            "Respond with the structured output only."
        )),
        HumanMessage(content=state.query),
    ]) # type: ignore
    print(f"[ROUTER] → {decision.route} | {decision.reasoning}")
    return {"route": decision.route}


def retrieve_node(state: AgentState, retriever) -> dict:
    print(f"[RETRIEVE] Hybrid search: {state.query!r}")
    docs = retriever.invoke(state.query)
    print(f"[RETRIEVE] Got {len(docs)} docs.")
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        score = d.metadata.get("_score", "?")
        print(f"  [{i+1}] {src} p{page} score={score}")
    return {"retrieved_docs": docs}


def generate_rag_node(state: AgentState) -> dict:
    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','?')} | Page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in state.retrieved_docs
    )
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "document context below. If the answer is not in the context, say "
            "'I could not find this in the provided documents.'\n\n"
            f"CONTEXT:\n{context}"
        )),
        HumanMessage(content=state.query),
    ])
    print(f"[GENERATE-RAG] Done. ({len(response.content)} chars)")
    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state.query),
            AIMessage(content=response.content),
        ],
    }


def generate_direct_node(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the question concisely."),
        HumanMessage(content=state.query),
    ])
    print(f"[GENERATE-DIRECT] Done. ({len(response.content)} chars)")
    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state.query),
            AIMessage(content=response.content),
        ],
    }


# ────────────────────────────────────────────
# CONDITIONAL EDGE
# ─────────────────────────────────────────────
def route_edge(state: AgentState) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state.route == "retrieve" else "generate_direct"


# ─────────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────────
def build_graph(retriever):
    graph = StateGraph(AgentState)

    graph.add_node("router",          router_node)
    graph.add_node("retrieve",        lambda s: retrieve_node(s, retriever))
    graph.add_node("generate_rag",    generate_rag_node)
    graph.add_node("generate_direct", generate_direct_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_edge,
        {
            "retrieve":        "retrieve",
            "generate_direct": "generate_direct",
        },
    )

    graph.add_edge("retrieve",        "generate_rag")
    graph.add_edge("generate_rag",    END)
    graph.add_edge("generate_direct", END)

    return graph.compile()
#%%

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Decide: ingest fresh or connect to existing ──────────────
    pdf_files = list(set(
        glob.glob(os.path.join(PDF_DIR, "**/*.pdf"), recursive=True) +
        glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    ))

    if pdf_files:
        print(f"[INFO] Found {len(pdf_files)} PDF(s). Ingesting …")
        vectorstore = ingest_pdfs(PDF_DIR)
    else:
        print("[INFO] No PDFs found. Connecting to existing OpenSearch index …")
        vectorstore = connect_vectorstore()

    # ── Setup hybrid pipeline (idempotent) ────────────────────────
    setup_hybrid_pipeline(vectorstore)

    # ── Build native hybrid retriever ─────────────────────────────
    retriever = build_hybrid_retriever(vectorstore)

    # ── Compile LangGraph ─────────────────────────────────────────
    app = build_graph(retriever)

    print("\n" + "=" * 60)
    print("  Agentic Hybrid RAG  |  LangGraph + OpenSearch")
    print("=" * 60)
    print("  search_type : hybrid_search (BM25 + kNN, server-side RRF)")
    print(f"  pipeline    : {PIPELINE_NAME}")
    print(f"  index       : {INDEX_NAME}")
    print(f"  top_k       : {TOP_K}")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    # ── Chat loop ─────────────────────────────────────────────────
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = app.invoke(AgentState(query=query))
        print(f"\nAssistant: {result['answer']}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
    
#%%
