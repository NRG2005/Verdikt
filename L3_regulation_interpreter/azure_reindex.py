"""
Creates/refreshes a NEW Azure AI Search index (chargeback-dispute-corpus) with
the current regulation_corpus.json, as a genuine hybrid index: BM25 keyword
fields + a vector field (nomic-embed-text, 768-dim, cosine HNSW) so
hybrid_retrieval.py's Azure branch can do real hybrid (keyword + vector)
search, not keyword-only.

Deliberately a NEW index rather than overwriting the old "compliance-regulations"
index (1583 stale AML documents) -- this is a shared Azure resource, not
something to delete from without being asked to.

Run: python3 L3_regulation_interpreter/azure_reindex.py
"""
import os
import sys
import json
import requests
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from L3_regulation_interpreter.hybrid_retrieval import chunk_regulation_document
from L3_regulation_interpreter.llm_client import generate_ollama_embedding

ENDPOINT = os.environ.get("SEARCH_ENDPOINT")
KEY = os.environ.get("SEARCH_API_KEY")
INDEX_NAME = "chargeback-dispute-corpus"
API_VERSION = "2023-11-01"
VECTOR_DIMS = 768  # nomic-embed-text


def create_index():
    schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "chunk_id", "type": "Edm.String", "key": True, "searchable": False, "filterable": True, "retrievable": True},
            {"name": "document_id", "type": "Edm.String", "searchable": True, "filterable": True, "retrievable": True, "facetable": True},
            {"name": "title", "type": "Edm.String", "searchable": True, "filterable": True, "retrievable": True},
            {"name": "content", "type": "Edm.String", "searchable": True, "filterable": False, "retrievable": True},
            {"name": "searchable_text", "type": "Edm.String", "searchable": True, "filterable": False, "retrievable": True},
            {"name": "section_heading", "type": "Edm.String", "searchable": True, "filterable": True, "retrievable": True},
            {"name": "regulator", "type": "Edm.String", "searchable": True, "filterable": True, "retrievable": True, "facetable": True},
            {"name": "document_type", "type": "Edm.String", "searchable": True, "filterable": True, "retrievable": True, "facetable": True},
            {
                "name": "content_vector", "type": "Collection(Edm.Single)",
                "searchable": True, "retrievable": True, "filterable": False,
                "dimensions": VECTOR_DIMS, "vectorSearchProfile": "myHnswProfile",
            },
        ],
        "vectorSearch": {
            "algorithms": [{"name": "myHnsw", "kind": "hnsw",
                             "hnswParameters": {"metric": "cosine", "m": 4, "efConstruction": 400, "efSearch": 500}}],
            "profiles": [{"name": "myHnswProfile", "algorithm": "myHnsw"}],
        },
    }
    r = requests.put(
        f"{ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}",
        headers={"api-key": KEY, "Content-Type": "application/json"},
        json=schema,
    )
    print(f"create/update index: {r.status_code}")
    if r.status_code not in (200, 201):
        print(r.text)
        r.raise_for_status()


def upload_chunks():
    payload = json.loads(open("L3_regulation_interpreter/regulation_corpus.json").read())
    docs = payload.get("documents", [])
    print(f"Found {len(docs)} documents in corpus")

    chunks = []
    for d in docs:
        chunks.extend(chunk_regulation_document(d))
    print(f"Generated {len(chunks)} chunks")

    actions = []
    for c in chunks:
        emb = generate_ollama_embedding(c["searchable_text"])
        if not emb:
            print(f"  WARNING: no embedding for {c['chunk_id']}, skipping vector field")
        actions.append({
            "@search.action": "mergeOrUpload",
            "chunk_id": c["chunk_id"].replace(":", "_"),  # Azure keys disallow ':'
            "document_id": c["document_id"],
            "title": c["title"],
            "content": c["content"],
            "searchable_text": c["searchable_text"],
            "section_heading": c.get("section_heading", ""),
            "regulator": c.get("regulator", ""),
            "document_type": c.get("document_type", ""),
            **({"content_vector": emb} if emb else {}),
        })

    r = requests.post(
        f"{ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version={API_VERSION}",
        headers={"api-key": KEY, "Content-Type": "application/json"},
        json={"value": actions},
    )
    print(f"upload: {r.status_code}")
    if r.status_code not in (200, 201):
        print(r.text)
        r.raise_for_status()
    results = r.json().get("value", [])
    failed = [x for x in results if not x.get("status")]
    print(f"{len(results)} documents indexed, {len(failed)} failed")
    if failed:
        print(failed[:3])


if __name__ == "__main__":
    create_index()
    upload_chunks()
    print("Done. Index:", INDEX_NAME)
