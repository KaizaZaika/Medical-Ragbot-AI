import json
from neo4j import GraphDatabase

# 1. Cấu hình kết nối tới Docker Neo4j của mày
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password123")  # Thay bằng pass mày cấu hình lúc chạy docker

def init_db(tx):
    # Tạo constraint để tránh trùng lặp dữ liệu khi chạy lại script nhiều lần
    tx.run("CREATE CONSTRAINT unique_url IF NOT EXISTS FOR (n:Nguon) REQUIRE n.url IS UNIQUE")
    tx.run("CREATE CONSTRAINT unique_question IF NOT EXISTS FOR (q:CauHoi) REQUIRE q.text IS UNIQUE")

def insert_qa(tx, url, question, answer):
    # Lệnh Cypher tạo Node và kết nối tụi nó lại với nhau
    query = """
    MERGE (n:Nguon {url: $url})
    MERGE (q:CauHoi {text: $question})
    MERGE (a:CauTraLoi {text: $answer})
    
    MERGE (q)-[:TRICH_DAN_TU]->(n)
    MERGE (a)-[:TRICH_DAN_TU]->(n)
    MERGE (q)-[:CO_CAU_TRA_LOI]->(a)
    """
    tx.run(query, url=url, question=question, answer=answer)

# 2. Chạy pipeline bơm dữ liệu
print("Bắt đầu kết nối Neo4j và khởi tạo...")
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        # Khởi tạo constraint trước
        session.execute_write(init_db)
        
        # Đọc file jsonl sạch đã làm
        print("Đang đọc file data và bơm vào đồ thị...")
        count = 0
        with open("final_medical_qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                url = data.get('url', '')
                question = data.get('question', '')
                answer = data.get('answer', '')
                
                if question and answer:
                    session.execute_write(insert_qa, url, question, answer)
                    count += 1
                    if count % 100 == 0:
                        print(f"Đã bơm thành công {count} bản ghi.")

print(f"Ngon lành! Đã nạp xong tổng cộng {count} cặp Q&A vào Neo4j Graph DB.")