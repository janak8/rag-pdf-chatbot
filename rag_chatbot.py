import os
import streamlit as st
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import InferenceClient

# 1. SETUP UI & SECRETS
st.title("📄 PDF RAG Chatbot")
# Use st.secrets for Streamlit Cloud, os.getenv for local
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found. Set it in Streamlit Secrets or .env file.")
    st.stop()

# 2. INITIALIZE MODELS (Cached to save memory/time)
@st.cache_resource
def load_models():
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm_client = InferenceClient(api_key=HF_TOKEN)
    return embed_model, llm_client

embedding_model, client = load_models()

# 3. PDF UPLOAD AND INDEXING
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Build the index in memory from the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()
    
    # Create vectorstore in memory (replaces FAISS.load_local)
    vectorstore = FAISS.from_documents(pages, embedding_model)
    st.success("✅ PDF Indexed Successfully!")

    # 4. CHAT INTERFACE
    query = st.text_input("Ask a question about your PDF:")

    if query:
        try:
            # Retrieve relevant chunks
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Build prompt
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

            # Generate response
            response = client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=messages,
                max_tokens=500,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            st.write(f"**Bot:** {answer}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a PDF file to begin.")
