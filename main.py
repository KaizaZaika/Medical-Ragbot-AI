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
import time

# =====================================================
# BASE DIR
# =====================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =====================================================
# PATHS
# =====================================================
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

# CHAT MODEL
OLLAMA_MODEL = "gemma3:1b"

# EMBEDDING MODEL
EMBED_MODEL = "bge-m3"

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI(
    title="AIMed - Hybrid RAG"
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
# GLOBAL VARIABLES
# =====================================================
texts = []

index = None

bm25 = None

tokenized_texts = []

# =====================================================
# EMBEDDING FUNCTION
# =====================================================
def embed_texts(texts_list):

    embeddings = []

    total = len(texts_list)

    print(f"\n🧠 Embedding {total} docs...\n")

    for i, text in enumerate(texts_list):

        try:

            response = ollama.embeddings(

                model=EMBED_MODEL,

                prompt=text
            )

            embeddings.append(
                response["embedding"]
            )

            print(f"✅ {i+1}/{total}")

        except Exception as e:

            print("❌ EMBED ERROR")
            print(str(e))

    return np.array(
        embeddings,
        dtype="float32"
    )

# =====================================================
# QUERY EMBEDDING
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

    return embedding.reshape(1, -1)

# =====================================================
# BUILD BM25
# =====================================================
def build_bm25():

    global bm25
    global tokenized_texts
    global texts

    print("\n⚡ Building BM25...\n")

    tokenized_texts = [

        text.lower().split()

        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_texts
    )

    print("✅ BM25 Ready!")

# =====================================================
# LOAD VECTOR DB
# =====================================================
if (
    os.path.exists(DB_FAISS_PATH)
    and
    os.path.exists(TEXTS_MAP_PATH)
):

    print("📂 Loading existing vector DB...")

    index = faiss.read_index(
        DB_FAISS_PATH
    )

    with open(
        TEXTS_MAP_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        texts = json.load(f)

    print(
        f"✅ Loaded {len(texts)} docs"
    )

    build_bm25()

# =====================================================
# BUILD VECTOR DB
# =====================================================
else:

    print("🆕 Building vector DB...")

    try:

        print("\n📁 DATA PATH:")
        print(DATA_PATH)

        print(
            "\n📁 FILE EXISTS:",
            os.path.exists(DATA_PATH)
        )

        with open(
            DATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                data = json.loads(line)

                question = data.get(
                    "question",
                    ""
                )

                answer = data.get(
                    "answer",
                    ""
                )

                text = f"""
Câu hỏi:
{question}

Trả lời:
{answer}
"""

                texts.append(text)

        print(
            f"\n📄 Total docs: {len(texts)}"
        )

        # =============================================
        # EMBEDDING
        # =============================================
        embeddings = embed_texts(
            texts
        )

        # =============================================
        # NORMALIZE
        # =============================================
        faiss.normalize_L2(
            embeddings
        )

        dimension = embeddings.shape[1]

        # =============================================
        # FAISS
        # =============================================
        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        # =============================================
        # SAVE
        # =============================================
        faiss.write_index(
            index,
            DB_FAISS_PATH
        )

        with open(
            TEXTS_MAP_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                texts,
                f,
                ensure_ascii=False,
                indent=4
            )

        print("\n💾 Vector DB saved!")

        # =============================================
        # BM25
        # =============================================
        build_bm25()

    except Exception as e:

        print("\n❌ BUILD ERROR")
        print(str(e))

# =====================================================
# VECTOR SEARCH
# =====================================================
def vector_search(
    query,
    k=5
):

    global index
    global texts

    if index is None:

        print("❌ INDEX IS NONE")

        return []

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

    global bm25
    global texts

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
# CHAT API
# =====================================================
@app.post("/api/chat")
async def chat_with_aimed(
    request: ChatRequest
):

    try:

        total_start = time.time()

        user_msg = request.message.strip()

        print("\n==========================")
        print("👤 USER:", user_msg)

        if not user_msg:

            return {
                "error": "Tin nhắn rỗng"
            }

        # =============================================
        # VECTOR SEARCH
        # =============================================
        t1 = time.time()

        vector_docs = vector_search(
            user_msg,
            k=5
        )

        print(
            f"🔎 Vector search: {time.time() - t1:.2f}s"
        )

        # =============================================
        # BM25 SEARCH
        # =============================================
        t2 = time.time()

        bm25_docs = bm25_search(
            user_msg,
            k=5
        )

        print(
            f"📚 BM25 search: {time.time() - t2:.2f}s"
        )

        # =============================================
        # MERGE
        # =============================================
        docs = list(

            dict.fromkeys(

                vector_docs + bm25_docs
            )
        )

        # =============================================
        # RERANK
        # =============================================
        t3 = time.time()

        docs = rerank_documents(
            user_msg,
            docs,
            top_k=3
        )

        print(
            f"🏆 Rerank: {time.time() - t3:.2f}s"
        )

        print("\n📄 FINAL DOCS:\n")

        for i, doc in enumerate(docs):

            print(f"\n========== DOC {i+1} ==========\n")

            print(doc[:400])

        # =============================================
        # CONTEXT
        # =============================================
        context = "\n\n".join(
            docs
        )[:1200]

        # =============================================
        # PROMPT
        # =============================================
        prompt = f"""
Bạn là AI bác sĩ.

Chỉ trả lời dựa trên dữ liệu được cung cấp.

Nếu không thấy thông tin phù hợp:
- hãy nói không tìm thấy thông tin
- không được tự bịa

Trả lời:
- ngắn gọn
- dễ hiểu
- đúng trọng tâm

================ DỮ LIỆU ================
{context}

================ CÂU HỎI ================
{user_msg}
"""

        # =============================================
        # OLLAMA CHAT
        # =============================================
        t4 = time.time()

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
                "temperature": 0.1
            }
        )

        print(
            f"🤖 Ollama: {time.time() - t4:.2f}s"
        )

        print(
            f"⚡ TOTAL: {time.time() - total_start:.2f}s"
        )

        ai_text = response["message"]["content"]

        return {

            "user_message": user_msg,

            "ai_response": ai_text,

            "retrieved_docs": docs
        }

    except Exception as e:

        print("\n❌ CHAT ERROR")
        print(str(e))

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

        "index_loaded": index is not None,

        "llm_model": OLLAMA_MODEL,

        "embedding_model": EMBED_MODEL
    }

# =====================================================
# RUN
# =====================================================

# INSTALL:
# pip install fastapi uvicorn faiss-cpu numpy rank-bm25 ollama

# PULL:
# ollama pull gemma3:1b
# ollama pull bge-m3

# RUN:
# uvicorn main:app --reload