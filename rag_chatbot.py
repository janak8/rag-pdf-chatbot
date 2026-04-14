import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# 1. LOAD ENV VARIABLES
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found. Please set it in your .env file.")

# 2. LOAD EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 3. LOAD VECTOR DATABASE
FAISS_PATH = "faiss_index"

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError("❌ 'faiss_index' not found. Run embedding step first.")

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

print("✅ PDF Database Loaded Successfully.")

# 4. INITIALIZE LLM CLIENT
client = InferenceClient(api_key=HF_TOKEN)

# 5. CHAT LOOP
print("\n💬 Chatbot ready! Type 'exit' to quit.")

while True:
    query = input("\nAsk a question: ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting chatbot.")
        break

    try:
        # --- Retrieve relevant chunks ---
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # --- Build prompt ---
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer using ONLY the provided context. If the answer is not in the context, say you don't know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]

        # --- Generate response ---
        response = client.chat_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )

        answer = response.choices[0].message.content

        print(f"\nBot: {answer}")

    except Exception as e:
        print(f"\n❌ Error: {e}")