from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Convert chunks → vectors
texts = [chunk.page_content for chunk in chunks]
embeddings = embedding_model.embed_documents(texts)

print(f"Number of chunks: {len(texts)}")
print(f"Embedding vector size: {len(embeddings[0])}")