from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import ollama

import faiss
import numpy as np

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

# LLM
OLLAMA_MODEL = "gemma3:1b"

# EMBEDDING MODEL
EMBED_MODEL = "nomic-embed-text"

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI(
    title="AIMed - Full Local RAG"
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

# =====================================================
# EMBEDDING FUNCTION
# =====================================================
def embed_texts(texts_list):

    embeddings = []

    total = len(texts_list)

    print(f"🧠 Embedding {total} docs...")

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

# =====================================================
# BUILD VECTOR DB
# =====================================================
else:

    print("🆕 Building vector DB...")

    try:

        print("📁 DATA PATH:")
        print(DATA_PATH)

        print(
            "📁 FILE EXISTS:",
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
        # SAVE DB
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

        print("💾 Vector DB saved!")

    except Exception as e:

        print("❌ BUILD ERROR")
        print(str(e))

# =====================================================
# SEARCH
# =====================================================
def search_documents(
    query,
    k=2
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
                texts[idx][:700]
            )

    return results

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

        docs = search_documents(
            user_msg,
            k=2
        )

        print(
            f"🔎 Search time: {time.time() - t1:.2f}s"
        )

        context = "\n\n".join(docs[:2])[:700]

        # =============================================
        # PROMPT
        # =============================================
        prompt = f"""
Bạn là AI bác sĩ.

Hãy trả lời:
- ngắn gọn
- dễ hiểu
- đúng trọng tâm

Nếu không đủ dữ liệu:
- nói không đủ thông tin
- khuyên đi khám bác sĩ

================ DỮ LIỆU ================
{context}

================ CÂU HỎI ================
{user_msg}
"""

        # =============================================
        # OLLAMA CHAT
        # =============================================
        t2 = time.time()

        response = await asyncio.to_thread(

            ollama.chat,

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print(
            f"🤖 Ollama time: {time.time() - t2:.2f}s"
        )

        print(
            f"⚡ Total time: {time.time() - total_start:.2f}s"
        )

        ai_text = response["message"]["content"]

        return {

            "user_message": user_msg,

            "ai_response": ai_text,

            "retrieved_docs": docs
        }

    except Exception as e:

        print("❌ CHAT ERROR")
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

# START:
# uvicorn main:app