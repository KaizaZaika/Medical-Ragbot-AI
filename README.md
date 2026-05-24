# AIMed - Hệ thống Hỏi đáp Y tế với Hybrid RAG

Hệ thống hỏi đáp y tế sử dụng công nghệ Hybrid RAG (Retrieval Augmented Generation) kết hợp vector search và BM25 để cung cấp câu trả lời chính xác dựa trên dữ liệu y tế.

## 🏗️ Kiến trúc dự án

- **Backend**: FastAPI (Python)
- **Frontend**: Next.js (React + TypeScript + TailwindCSS)
- **LLM**: Ollama (gemma3:1b)
- **Vector Search**: FAISS
- **Keyword Search**: BM25
- **Embedding Model**: bge-m3

## 📋 Yêu cầu hệ thống

- Python 3.8+
- Node.js 18+
- Ollama (cần cài đặt và chạy dịch vụ)

## 🔧 Cài đặt

### 1. Cài đặt Ollama

Tải và cài đặt Ollama từ: https://ollama.ai/

Sau khi cài đặt, kéo các model cần thiết:

```bash
ollama pull gemma3:1b
ollama pull bge-m3
```

### 2. Cài đặt Backend

Tạo virtual environment (khuyên dùng):

```bash
python -m venv venv
```

Kích hoạt virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

Cài đặt các thư viện Python:

**Cách 1 - Cài đặt từ requirements.txt (khuyên dùng):**
```bash
pip install -r requirements.txt
```

**Cách 2 - Cài đặt thủ công:**
```bash
pip install fastapi uvicorn faiss-cpu numpy rank-bm25 ollama
```

### 3. Cài đặt Frontend

Di chuyển vào thư mục frontend:

```bash
cd frontend
```

Cài đặt các thư viện Node.js:

```bash
npm install
```

Quay lại thư mục gốc:

```bash
cd ..
```

## 🚀 Chạy ứng dụng

### 1. Chạy Backend

Mở terminal, kích hoạt virtual environment và chạy:

```bash
uvicorn main1:app --reload
```

Backend sẽ chạy tại: `http://localhost:8000`

API documentation có sẵn tại: `http://localhost:8000/docs`

### 2. Chạy Frontend

Mở terminal mới, di chuyển vào thư mục frontend và chạy:

```bash
cd frontend
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:3000`

## 📁 Cấu trúc dự án

```
medical_project/
├── main.py                      # Backend FastAPI chính
├── frontend/                    # Frontend Next.js
│   ├── app/                     # Pages và components
│   ├── package.json             # Dependencies frontend
│   └── ...
├── final_medical_qa.jsonl       # Dữ liệu Q&A y tế
├── vector_db_faiss.index        # Vector database (FAISS)
├── texts_data.json              # Mapping văn bản
├── chunking.py                  # Script chunking văn bản
├── ingest_vihealthqa_neo4j.py   # Script ingest dữ liệu
└── venv/                        # Virtual environment Python
```

## 🔄 Quy trình hoạt động

1. **Vector Search**: Sử dụng FAISS để tìm kiếm dựa trên embedding (bge-m3)
2. **BM25 Search**: Tìm kiếm dựa trên từ khóa
3. **Rerank**: Kết hợp và xếp hạng lại kết quả
4. **LLM Generation**: Sử dụng gemma3:1b để tạo câu trả lời dựa trên context

## 📡 API Endpoints

### POST /api/chat

Gửi câu hỏi và nhận câu trả lời từ AI.

**Request Body:**
```json
{
  "message": "Câu hỏi của bạn"
}
```

**Response:**
```json
{
  "user_message": "Câu hỏi của bạn",
  "ai_response": "Câu trả lời từ AI",
  "retrieved_docs": ["Danh sách tài liệu liên quan"]
}
```

### GET /

Kiểm tra trạng thái server.

## 📝 Lưu ý quan trọng

- **Lần chạy đầu tiên**: Hệ thống sẽ tự động xây dựng vector database từ file `final_medical_qa.jsonl`. Quá trình này có thể mất thời gian phụ thuộc vào số lượng tài liệu.
- **Ollama service**: Đảm bảo Ollama đang chạy trước khi khởi động backend.
- **Dữ liệu**: File `final_medical_qa.jsonl` phải tồn tại trong thư mục gốc trước khi chạy.

## 🛠️ Khắc phục sự cố

### Lỗi "Ollama not responding"
- Kiểm tra Ollama có đang chạy không: `ollama list`
- Khởi động lại Ollama nếu cần

### Lỗi "Model not found"
- Đảm bảo đã pull các model: `ollama pull gemma3:1b` và `ollama pull bge-m3`

### Lỗi "vector_db_faiss.index not found"
- Xóa file `vector_db_faiss.index` và `texts_data.json` nếu tồn tại
- Chạy lại backend để hệ thống tự động rebuild vector database

## 📄 License

Dự án này được phát triển cho mục đích nghiên cứu và giáo dục.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón!
