import google.generativeai as genai

# Nhét key vào đây
GEMINI_API_KEY = "AIzaSyDpG0x-IQcdUK2vSDcVUEKXQocMTnTYKOE"
genai.configure(api_key=GEMINI_API_KEY)

print(f"{'TÊN MODEL':<40} | {'TÍNH NĂNG':<30}")
print("-" * 80)

for m in genai.list_models():
    # m.supported_generation_methods là list các tính năng: generateContent, embedContent, countTokens...
    methods = ", ".join(m.supported_generation_methods)
    print(f"{m.name:<40} | {methods}")

print("-" * 80)
print("Xong! Nhìn cột 'TÍNH NĂNG' mà chọn con nào hợp với mục đích.")