from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .identity import company_display_name
from .models import TranscriptRecord


def index_transcripts(
    records: list[TranscriptRecord],
    *,
    persist_dir: Path,
    embedding_model: str,
    ollama_base_url: str | None = None,
    collection_name: str = "ir_transcripts",
    chunk_size: int = 1800,
    chunk_overlap: int = 250,
) -> int:
    if not records:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    embedding_kwargs = {"model": embedding_model}
    if ollama_base_url:
        embedding_kwargs["base_url"] = ollama_base_url
    embeddings = OllamaEmbeddings(**embedding_kwargs)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    docs: list[Document] = []
    for record in records:
        metadata = {
            "symbol": record.company.symbol,
            "company": company_display_name(record.company),
            "source_url": str(record.source_url),
            "title": record.title,
        }
        if record.fiscal_period:
            metadata["fiscal_period"] = record.fiscal_period
        if record.call_date:
            metadata["call_date"] = record.call_date.isoformat()

        docs.extend(splitter.split_documents([Document(page_content=record.text, metadata=metadata)]))

    vectorstore.add_documents(docs)
    return len(docs)
