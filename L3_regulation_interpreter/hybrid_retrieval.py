"""
L3: Hybrid Retrieval

Current goal:
- develop retrieval locally against a JSON regulation corpus
- keep the same interfaces easy to swap to Azure AI Search later

Design:
- input JSON stores source regulation content, not embeddings
- this module chunks, indexes, and ranks clauses/sections at runtime
- ranking is currently lexical plus metadata-aware; later the vector part can
  be replaced by Azure AI Search hybrid retrieval without changing callers
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config import get_config

config = get_config()

DEFAULT_LOCAL_CORPUS_PATH = os.environ.get(
    "LOCAL_REGULATION_CORPUS_PATH",
    str(Path(__file__).with_name("regulation_corpus.json")),
)
DEFAULT_TOP_K = int(os.environ.get("L3_TOP_K", "5"))
DEFAULT_CHUNK_WORDS = int(os.environ.get("L3_CHUNK_WORDS", "400"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("L3_CHUNK_OVERLAP", "100"))


def _tokenize(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_search_query(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts the transaction/event context into a retrieval query object.

    This is intentionally simple and auditable for now. Later the same query
    object can be translated into Azure AI Search keyword + vector inputs.
    """
    channel = str(event.get("channel", "")).upper()
    amount = event.get("amount")
    amount_band = "small"
    try:
        numeric_amount = float(amount)
        if numeric_amount >= 1_000_000:
            amount_band = "very_large"
        elif numeric_amount >= 200_000:
            amount_band = "large"
        elif numeric_amount >= 50_000:
            amount_band = "medium"
    except (TypeError, ValueError):
        numeric_amount = None

    keywords = [
        channel.lower(),
        "rbi",
        "compliance",
        "transaction",
        "payment",
        amount_band,
    ]

    for key in ("sender_bank", "receiver_bank", "receiver_type", "purpose_code", "channel"):
        value = event.get(key)
        if value:
            keywords.extend(_tokenize(value))

    for key in ("scenario_tag", "funds_in_out_pattern", "l3_investigation_notes", "purpose_code_declared"):
        value = event.get(key)
        if value:
            keywords.extend(_tokenize(value))

    for key in ("t2_watchlist_hit", "t3_risk_label", "geo_country", "transaction_type"):
        value = event.get(key)
        if value:
            keywords.extend(_tokenize(value))

    # Add L2 triggers to keywords
    l2_triggers = event.get("l2_triggers_fired", [])
    if l2_triggers:
        for t in l2_triggers:
            keywords.extend(_tokenize(t))

    scenario = event.get("scenario_tag", "")
    if not scenario:
        if any("C5" in t for t in l2_triggers):
            scenario = "CROSS_BORDER_LRS Liberalised Remittance Scheme LRS limits"
        elif any("C1_high_value" in t for t in l2_triggers):
            scenario = "high value transaction enhanced due diligence monitoring"
        elif any("C1" in t for t in l2_triggers):
            scenario = "structuring smurfing transaction splitting"
        else:
            scenario = " ".join(l2_triggers).replace("_", " ")
    scenario = scenario.replace("_", " ")
    
    investigation = event.get("l3_investigation_notes", "")
    channel_str = event.get("channel", channel)
    txn_type = event.get("transaction_type", "").replace("_", " ")
    
    query_text = f"search_query: What are the RBI regulatory guidelines, reporting thresholds, and KYC requirements for suspected {scenario} via {channel_str} {txn_type}? Context: {investigation}"

    return {
        "channel": channel,
        "amount": numeric_amount,
        "amount_band": amount_band,
        "keywords": _dedupe_preserve_order(keywords),
        "query_text": query_text,
    }


def _window_words(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    stride = max(chunk_words - overlap_words, 1)
    while start < len(words):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window).strip())
        start += stride
    return chunks


def chunk_regulation_document(
    document: Dict[str, Any],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Converts one regulation document into searchable chunks.

    Expected source shape:
    {
      "document_id": "...",
      "title": "...",
      "regulator": "RBI",
      "document_type": "master_direction",
      "effective_date": "2026-01-01",
      "url": "...",
      "tags": ["KYC", "AML", "UPI"],
      "sections": [
        {
          "section_id": "...",
          "heading": "...",
          "text": "...",
          "clauses": ["...", "..."]
        }
      ]
    }
    """
    document_id = document.get("document_id") or document.get("id") or document.get("doc_id")
    title = document.get("title", "")
    tags = document.get("tags") or []
    regulator = document.get("regulator", "RBI")
    document_type = document.get("document_type", "regulation")
    effective_date = document.get("effective_date")
    url = document.get("url")

    chunked_rows: List[Dict[str, Any]] = []
    sections = document.get("sections") or []
    if not sections and document.get("text"):
        sections = [{"section_id": "full_text", "heading": title, "text": document["text"], "clauses": []}]

    for section in sections:
        section_id = section.get("section_id") or section.get("id") or "section"
        heading = section.get("heading", "")
        clauses = section.get("clauses") or []

        candidate_texts: List[str] = []
        if section.get("text"):
            candidate_texts.extend(_window_words(str(section["text"]), chunk_words, overlap_words))
        for clause in clauses:
            if clause:
                candidate_texts.extend(_window_words(str(clause), chunk_words, overlap_words))

        for index, chunk_text in enumerate(candidate_texts, start=1):
            searchable_text = " ".join(part for part in [title, heading, chunk_text, " ".join(tags)] if part)
            chunked_rows.append(
                {
                    "chunk_id": f"{document_id}:{section_id}:{index}",
                    "document_id": document_id,
                    "title": title,
                    "regulator": regulator,
                    "document_type": document_type,
                    "effective_date": effective_date,
                    "url": url,
                    "tags": tags,
                    "section_id": section_id,
                    "section_heading": heading,
                    "content": chunk_text,
                    "searchable_text": searchable_text,
                }
            )

    return chunked_rows


def load_regulation_corpus(
    corpus_path: Optional[str] = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    path = Path(corpus_path or DEFAULT_LOCAL_CORPUS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Regulation corpus not found at {path}. Add your JSON corpus or set LOCAL_REGULATION_CORPUS_PATH."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        documents = payload.get("documents") or payload.get("regulations") or [payload]
    elif isinstance(payload, list):
        documents = payload
    else:
        raise ValueError("Regulation corpus JSON must be a list or an object with a 'documents' field.")

    chunks: List[Dict[str, Any]] = []
    for document in documents:
        chunks.extend(chunk_regulation_document(document, chunk_words=chunk_words, overlap_words=overlap_words))
    return chunks


def _compute_retrieval_match(top_chunks: Sequence[Dict[str, Any]]) -> float:
    """
    Produces the retrieval-match sub-score expected by L3.

    Current interpretation:
    - high if the top result is strong and the top few results are consistent
    - low if results are weak or sparse
    """
    if not top_chunks:
        return 0.0

    top_score = float(top_chunks[0].get("retrieval_score", 0.0))
    avg_top = sum(float(row.get("retrieval_score", 0.0)) for row in top_chunks[:3]) / min(len(top_chunks), 3)

    # Two genuinely different score scales land here, and treating them the
    # same silently under-reported Azure hybrid results as near-zero
    # confidence:
    #   - Local Chroma cosine similarity: already 0-1, ~0.5+ for a real match.
    #   - Azure hybrid RRF fusion score: Azure's formula is
    #     sum(1/(k+rank)) per contributing query leg with default k=60, so a
    #     document ranked #1 in BOTH the keyword and vector legs scores
    #     ~1/61 + 1/61 = 0.0328 -- that IS the practical ceiling, not a weak
    #     match. Dividing by ~2.0 (as if this were a 0-1ish score) mapped a
    #     best-possible RRF result to ~1.6% instead of ~100%.
    # RRF scores are reliably small (<0.2); cosine scores are not. Detect and
    # rescale accordingly rather than using one linear formula for both.
    RRF_CEILING = 2.0 / 61.0  # best-possible score: rank 1 on both legs, k=60
    if top_score < 0.2:
        match_score = min(1.0, ((0.65 * top_score) + (0.35 * avg_top)) / RRF_CEILING)
    else:
        match_score = min(1.0, (0.65 * top_score) + (0.35 * avg_top))
    return round(match_score, 4)


def search_regulations(
    event: Dict[str, Any],
    corpus_path: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Performs dual-retrieval using Azure AI Search (Hybrid) and local ChromaDB (Vector).
    Returns both sets of chunks for dual LLM evaluation.
    """
    query = build_search_query(event)
    from L3_regulation_interpreter.llm_client import generate_ollama_embedding
    
    azure_chunks = []
    local_chunks = []
    
    try:
        query_vector = generate_ollama_embedding(query.get("query_text", ""))
        if not query_vector:
            return {"query": query, "retrieval_match": 0.0, "chunks": [], "nomic_chunks": [], "backend": "failed"}
            
        print("L3: Searching for relevant regulations via Azure AI Search and Local ChromaDB...")
        
        # 1. Local ChromaDB Search
        try:
            import chromadb
            client = chromadb.PersistentClient(path="chroma_db")
            collection = client.get_collection("compliance_regulations")
            
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            if results and results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - (distance / 2.0)
                    
                    local_chunks.append({
                        "chunk_id": results["ids"][0][i],
                        "document_id": results["metadatas"][0][i].get("document_id", ""),
                        "title": results["metadatas"][0][i].get("title", ""),
                        "content": results["documents"][0][i],
                        "section_heading": results["metadatas"][0][i].get("section_heading", ""),
                        "retrieval_score": round(similarity, 4),
                    })
        except Exception as local_exc:
            print(f"L3: Local ChromaDB vector search failed: {local_exc}")
            
        # 2. Azure AI Search -- genuine hybrid (BM25 keyword + vector, RRF-fused
        # by Azure) against "chargeback-dispute-corpus", a NEW index built
        # from the current regulation_corpus.json (see azure_reindex.py).
        # The original "compliance-regulations" index (1583 docs) was left
        # untouched -- it's the old AML corpus and a shared resource, not
        # something to overwrite without being asked to. This new index uses
        # the same nomic-embed-text vectors as the local Chroma path, so the
        # two retrieval backends are genuinely comparable, not one real one
        # fake.
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            from azure.search.documents.models import VectorizedQuery

            endpoint = os.environ.get("SEARCH_ENDPOINT")
            key = os.environ.get("SEARCH_API_KEY")
            index_name = os.environ.get("AZURE_SEARCH_INDEX", "chargeback-dispute-corpus")

            if endpoint and key:
                credential = AzureKeyCredential(key)
                search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

                keyword_search_string = " ".join(query.get("keywords", [])) or query.get("query_text", "")
                vector_query = VectorizedQuery(
                    vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector"
                )

                results = search_client.search(
                    search_text=keyword_search_string,  # BM25 keyword leg
                    vector_queries=[vector_query],       # vector leg -- Azure fuses both (RRF)
                    select=["chunk_id", "document_id", "title", "content", "section_heading"],
                    top=top_k,
                )

                for result in results:
                    score = result["@search.score"]
                    azure_chunks.append({
                        "chunk_id": result.get("chunk_id"),
                        "document_id": result.get("document_id"),
                        "title": result.get("title"),
                        "content": result.get("content"),
                        "section_heading": result.get("section_heading", ""),
                        "retrieval_score": round(score, 4),
                    })
        except Exception as azure_exc:
            print(f"L3: Azure AI Search failed: {azure_exc}")
            
        retrieval_match = _compute_retrieval_match(azure_chunks if azure_chunks else local_chunks)
        
        return {
            "query": query,
            "retrieval_match": retrieval_match,
            "chunks": azure_chunks,
            "nomic_chunks": local_chunks,
            "backend": "dual_retrieval"
        }
        
    except Exception as exc:
        print(f"L3: Embedding generation failed: {exc}")
        return {"query": query, "retrieval_match": 0.0, "chunks": [], "nomic_chunks": [], "backend": "failed"}
