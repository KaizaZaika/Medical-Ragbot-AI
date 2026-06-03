from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import ollama

import faiss
import numpy as np

from rank_bm25 import BM25Okapi

import json
import os
import traceback

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "final_medical_qa.jsonl"
)

DB_FAISS_PATH = os.path.join(
    BASE_DIR,
    "vector_db_faiss.index"
)

TEXTS_MAP_PATH = os.path.join(
    BASE_DIR,
    "texts_data.json"
)

# =====================================================
# MODELS
# =====================================================

# LIGHT MODEL
OLLAMA_MODEL = "qwen2.5:0.5b"

# EMBEDDING
EMBED_MODEL = "bge-m3"

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI(
    title="AIMed"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# REQUEST MODEL
# =====================================================
class ChatRequest(BaseModel):
    message: str

# =====================================================
# GLOBAL
# =====================================================
texts = []

index = None

bm25 = None

# =====================================================
# EMBED QUERY
# =====================================================
def embed_query(query):

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )

    embedding = np.array(
        response["embedding"],
        dtype="float32"
    )

    # FIX DIMENSION
    if embedding.shape[0] > 768:

        embedding = embedding[:768]

    elif embedding.shape[0] < 768:

        padding = np.zeros(
            768 - embedding.shape[0],
            dtype="float32"
        )

        embedding = np.concatenate(
            [embedding, padding]
        )

    return embedding.reshape(1, -1)

# =====================================================
# LOAD DB
# =====================================================
print("📂 Loading DB...")

index = faiss.read_index(
    DB_FAISS_PATH
)

with open(
    TEXTS_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:

    texts = json.load(f)

print(f"✅ Loaded {len(texts)} docs")

# =====================================================
# BM25
# =====================================================
tokenized_texts = [

    text.lower().split()

    for text in texts
]

bm25 = BM25Okapi(
    tokenized_texts
)

print("✅ BM25 Ready")

# =====================================================
# VECTOR SEARCH
# =====================================================
def vector_search(
    query,
    k=5
):

    query_embedding = embed_query(
        query
    )

    faiss.normalize_L2(
        query_embedding
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        if idx < len(texts):

            results.append(
                texts[idx]
            )

    return results

# =====================================================
# BM25 SEARCH
# =====================================================
def bm25_search(
    query,
    k=5
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(
        scores
    )[::-1][:k]

    results = []

    for idx in top_indices:

        results.append(
            texts[idx]
        )

    return results

# =====================================================
# RERANK
# =====================================================
def rerank_documents(
    query,
    docs,
    top_k=3
):

    scored_docs = []

    for doc in docs:

        score = 0

        query_words = set(
            query.lower().split()
        )

        doc_words = set(
            doc.lower().split()
        )

        overlap = query_words.intersection(
            doc_words
        )

        score += len(overlap)

        score += doc.lower().count(
            query.lower()
        ) * 5

        scored_docs.append(
            (score, doc)
        )

    scored_docs.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return [

        doc

        for score, doc

        in scored_docs[:top_k]
    ]

# =====================================================
# REWRITE ANSWER
# =====================================================
async def refine_answer(raw_answer):

    prompt = f"""
Bạn là AI y tế.

Hãy viết lại nội dung sau thành câu trả lời:
- tự nhiên
- giống chatbot
- ngắn gọn
- dễ hiểu
- thân thiện
- không lan man
- không copy y nguyên

Nội dung:
{raw_answer}

Câu trả lời:
"""

    try:

        response = await asyncio.to_thread(

            ollama.chat,

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            options={
                "temperature": 0.3,
                "num_predict": 80
            }
        )

        ai_text = response["message"]["content"]

        if not ai_text.strip():

            return raw_answer

        return ai_text.strip()

    except:

        return raw_answer

# =====================================================
# CHAT API
# =====================================================
@app.post("/api/chat")
async def chat_with_aimed(
    request: ChatRequest
):

    try:

        user_msg = request.message.strip()

        print("\n===================")
        print("👤 USER:", user_msg)

        if not user_msg:

            return {
                "error": "Tin nhắn rỗng"
            }

        # ============================================
        # SEARCH
        # ============================================
        vector_docs = vector_search(
            user_msg,
            k=5
        )

        bm25_docs = bm25_search(
            user_msg,
            k=5
        )

        docs = list(
            dict.fromkeys(
                vector_docs + bm25_docs
            )
        )

        docs = rerank_documents(
            user_msg,
            docs,
            top_k=3
        )

        # ============================================
        # EXTRACT ANSWER
        # ============================================
        best_doc = docs[0]

        raw_answer = best_doc.split(
            "Trả lời:"
        )[-1].strip()

        print("\n📄 RAW ANSWER:\n")
        print(raw_answer)

        # ============================================
        # REFINE
        # ============================================
        ai_text = await refine_answer(
            raw_answer
        )

        print("\n🤖 FINAL ANSWER:\n")
        print(ai_text)

        return {

            "user_message": user_msg,

            "ai_response": ai_text,

            "retrieved_docs": docs
        }

    except Exception as e:

        print("\n❌ ERROR")
        traceback.print_exc()

        return {
            "error": str(e)
        }

# =====================================================
# ROOT
# =====================================================
@app.get("/")
async def home():

    return {

        "status": "running",

        "documents": len(texts),

        "model": OLLAMA_MODEL
    }

# =====================================================
# RUN
# =====================================================

# pip install fastapi uvicorn faiss-cpu numpy rank-bm25 ollama
# ollama pull qwen2.5:0.5b
# ollama pull bge-m3
# uvicorn main:app --reload