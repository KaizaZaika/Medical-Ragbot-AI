import os
import json
import faiss
import numpy as np
import time
from google import genai
from google.genai import types

# ==========================================
# SETUP
# ==========================================
GEMINI_API_KEY = "YOUR_API_KEY"
client = genai.Client(api_key="AIzaSyDkQcIA66w_npQVpOOHKVQkLcjjXfh3ecw")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "final_medical_qa.jsonl")
DB_FAISS_PATH = os.path.join(BASE_DIR, "vector_db_faiss.index")
TEXTS_MAP_PATH = os.path.join(BASE_DIR, "texts_data.json")

def main():
    texts = []
    print("[1/3] Reading data from JSONL...")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data.get("question", "")
            answer = data.get("answer", "")
            texts.append(f"Câu hỏi:\n{question}\n\nTrả lời:\n{answer}")

    print(f"[2/3] Loaded {len(texts)} documents. Generating Gemini embeddings (this may take a moment)...")
    
    embeddings = []
    for i, text in enumerate(texts):
        print(f"  -> Embedding doc {i+1}/{len(texts)}...")
        
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        embeddings.append(result.embeddings[0].values)
        
        # Small sleep to prevent hitting free-tier API rate limits
        time.sleep(0.5) 

    # Convert to numpy array and normalize for Cosine Similarity (IndexFlatIP)
    embeddings_np = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings_np)
    
    print("[3/3] Building and saving FAISS index...")
    dimension = embeddings_np.shape[1]  # This will be 768 for Gemini
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    faiss.write_index(index, DB_FAISS_PATH)
    with open(TEXTS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)
        
    print("✅ DONE! New Gemini FAISS database created successfully.")

if __name__ == "__main__":
    main()