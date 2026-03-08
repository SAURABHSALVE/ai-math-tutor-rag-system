"""RAG Pipeline using LangChain + FAISS + OpenAI text-embedding-3-small."""

import os
import json
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter ## updated import path since langchain v0.0.148
from langchain_core.documents import Document ## updated import path since langchain v0.0.148
import config


def _load_knowledge_base() -> List[Document]:
    """Load all markdown files from knowledge_base/ as LangChain Documents."""
    documents = []
    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not os.path.exists(kb_dir):
        return documents
    for fname in os.listdir(kb_dir):
        if fname.endswith(".md"):
            fpath = os.path.join(kb_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract topic tag from filename prefix (e.g., "calculus_integrals.md" -> "calculus")
            name_without_ext = fname.rsplit(".", 1)[0]
            topic = name_without_ext.split("_", 1)[0] if "_" in name_without_ext else name_without_ext
            documents.append(Document(page_content=content, metadata={"source": fname, "topic": topic}))
    return documents


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )


def build_vector_store() -> FAISS:
    """Build FAISS index from knowledge base documents."""
    documents = _load_knowledge_base()
    if not documents:
        raise ValueError("No knowledge base documents found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
    vectorstore.save_local(config.VECTOR_STORE_PATH)

    return vectorstore


def load_vector_store() -> FAISS:
    """Load existing FAISS index, or build if missing."""
    index_path = os.path.join(config.VECTOR_STORE_PATH, "index.faiss")
    if not os.path.exists(index_path):
        return build_vector_store()

    embeddings = _get_embeddings()
    return FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


_cached_vectorstore: FAISS | None = None


def get_vector_store() -> FAISS:
    """Return the cached FAISS vector store, loading/building once."""
    global _cached_vectorstore
    if _cached_vectorstore is None:
        _cached_vectorstore = load_vector_store()
    return _cached_vectorstore


def retrieve(query: str, top_k: int = None, topic: str = None) -> List[dict]:
    """Retrieve top-k relevant chunks for a query.

    If *topic* is provided, results from matching topic files are boosted
    by fetching extra candidates and prioritising those whose metadata
    ``topic`` tag matches.
    """
    top_k = top_k or config.TOP_K_RETRIEVAL
    vectorstore = get_vector_store()

    if topic:
        # Fetch extra candidates so we can prioritise topic-matched chunks
        fetch_k = top_k * 3
        results = vectorstore.similarity_search_with_score(query, k=fetch_k)
        # Partition into topic-matched and others
        matched = []
        rest = []
        for doc, score in results:
            if doc.metadata.get("topic") == topic:
                matched.append((doc, score))
            else:
                rest.append((doc, score))
        # Take topic-matched first, then fill remaining slots with others
        results = (matched + rest)[:top_k]
    else:
        results = vectorstore.similarity_search_with_score(query, k=top_k)

    output = []
    for doc, score in results:
        output.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(float(score), 4),
        })
    return output
