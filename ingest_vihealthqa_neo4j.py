import json
import argparse
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# 1. CẤU HÌNH API & DATABASE
os.environ["GEMINI_API_KEY"] = "AIzaSyDpG0x-IQcdUK2vSDcVUEKXQocMTnTYKOE" 
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"

# Kết nối Neo4j
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Dùng Gemini làm thợ xây đồ thị
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Disease", "Symptom", "Medication", "Patient_Group"],
    allowed_relationships=["HAS_SYMPTOM", "TREATS", "CAUSES", "AFFECTS"]
)

def ingest_data(input_file, limit):
    print(f"Đang đọc file {input_file}...")
    documents = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            data = json.loads(line)
            # Gom Q&A thành một text hoàn chỉnh để AI đọc
            text_content = f"Hỏi: {data.get('question', '')}\nĐáp: {data.get('answer', '')}"
            documents.append(Document(page_content=text_content))

    print(f"Bắt đầu bóc tách Thực thể và vẽ Đồ thị cho {len(documents)} câu (sẽ hơi lâu vì phải gọi AI)...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    
    print("Đang nạp dữ liệu vào Neo4j...")
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    print("Nạp thành công! Đã cấu trúc xong Knowledge Graph.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn file JSONL")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng dòng muốn chạy (để 0 chạy hết, cẩn thận tốn token)")
    args = parser.parse_args()
    
    ingest_data(args.input, args.limit)