#  RAG PDF Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from PDF documents.


##  Overview

This project implements a PDF-based chatbot using a **RAG pipeline**.
It extracts text from a PDF, chunks it, generates embeddings, stores them in a vector database, and answers user queries using an LLM.

It’s a minimal, easy-to-understand implementation suitable for learning or extending into a full application.


##  Features

* Extract text from PDF files
* Chunk text into overlapping segments
* Generate embeddings using modern embedding models
* Store embeddings in a vector store
* Retrieve relevant chunks based on user queries
* Use an LLM to generate final answers
* Modular, readable Python code



##  Project Structure

```bash
rag-pdf-chatbot/
│
├── load_pdf.py          # Extracts text from PDF
├── chunking.py          # Splits text into chunks
├── embeddings.py        # Generates embeddings
├── vector_store.py      # Stores and retrieves vectors
├── query.py             # Handles user queries
├── rag_chatbot.py       # Main pipeline
│
├── sample.pdf           # Example PDF
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```


##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/janak8/rag-pdf-chatbot
cd rag-pdf-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Add your API keys in a `.env` file (if required by your embedding/LLM provider).


##  Usage

Run the main chatbot script:

```bash
python rag_chatbot.py
```

Replace `sample.pdf` with any PDF you want to query.


##  How It Works

1. **Load PDF** → Extract raw text
2. **Chunk Text** → Split into overlapping segments
3. **Embed Chunks** → Convert text into vector embeddings
4. **Store Vectors** → Save in a vector database
5. **Query** → Retrieve relevant chunks
6. **Generate Answer** → LLM produces final response

This is the standard **RAG pipeline** used in modern AI systems.


##  Future Improvements

* Add a Streamlit or FastAPI web interface
* Support multiple PDFs
* Add caching for embeddings
* Add evaluation metrics
* Integrate more embedding models
* Add Docker support


##  Contributing

Contributions are welcome!
For major changes, please open an issue first to discuss what you’d like to modify.


##  Support

If you find this project useful, consider giving it a star!
