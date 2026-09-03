import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import json
from L3_regulation_interpreter.hybrid_retrieval import chunk_regulation_document
from L3_regulation_interpreter.llm_client import generate_ollama_embedding

def reindex():
    client = chromadb.PersistentClient(path='chroma_db')
    try:
        client.delete_collection('compliance_regulations')
    except:
        pass
    # Explicit cosine space: hybrid_retrieval.py's similarity formula
    # (1 - distance/2) assumes a distance range of [0,2], which only holds
    # for cosine distance. Chroma's default metric is raw L2, which produced
    # nonsensical similarity scores (e.g. -134) for the retrieval_match
    # confidence figure -- a pre-existing bug, not introduced by this swap,
    # fixed here since it directly affects the "explainable confidence"
    # number shown downstream.
    collection = client.create_collection(
        'compliance_regulations', metadata={"hnsw:space": "cosine"}
    )

    payload = json.loads(open('L3_regulation_interpreter/regulation_corpus.json', 'r').read())
    docs = payload.get('documents', [])
    print(f'Found {len(docs)} documents in corpus')

    chunks = []
    for d in docs:
        chunks.extend(chunk_regulation_document(d))

    print(f'Generated {len(chunks)} chunks')

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, c in enumerate(chunks):
        if i % 100 == 0:
            print(f"Processed {i}/{len(chunks)} chunks...")
            if len(ids) > 0:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                ids, embeddings, documents, metadatas = [], [], [], []
                
        emb = generate_ollama_embedding(c['searchable_text'])
        if emb:
            ids.append(c['chunk_id'])
            embeddings.append(emb)
            documents.append(c['content'])
            metadatas.append({
                'document_id': c['document_id'],
                'title': c['title'],
                'section_heading': c['section_heading']
            })
            
    if len(ids) > 0:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    print("Re-indexing complete!")

if __name__ == "__main__":
    reindex()
