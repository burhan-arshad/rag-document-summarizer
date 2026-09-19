# RAG Assistant

A simple Retrieval-Augmented Generation (RAG) application built with Python and Streamlit.

The application allows users to upload PDF or TXT documents and ask questions about their content. The system retrieves relevant document chunks and provides answers using an LLM.

## Live Demo

Live Demo: [Add your Streamlit Cloud URL here]

## Features

- Upload PDF and TXT documents
- Automatic document loading
- Text splitting into smaller chunks
- Local text embeddings
- Semantic similarity search
- ChromaDB vector storage
- MMR-based document retrieval
- Question answering with Groq
- Retrieved source display
- Simple Streamlit interface

## RAG Pipeline

Document Upload
        ↓
Document Loading
        ↓
Text Splitting
        ↓
Embedding Generation
        ↓
ChromaDB
        ↓
User Question
        ↓
Query Embedding
        ↓
Similarity Search
        ↓
Retriever
        ↓
Relevant Context
        ↓
Groq LLM
        ↓
Final Answer

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face
- Groq
- PyPDF
- MMR Retrieval

## Embedding Model

The project uses:

sentence-transformers/all-MiniLM-L6-v2

The embedding model runs locally, making the project suitable for learning and experimentation without requiring a paid embedding API.

## LLM

The application uses Groq for response generation.

Model:

openai/gpt-oss-120b

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project:

cd RAG

Create and activate a virtual environment:

uv venv

.venv\Scripts\activate

Install dependencies:

uv pip install -r requirements.txt

## Environment Variables

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key

Never commit your `.env` file to GitHub.

## Run the Application

Start Streamlit:

streamlit run app.py

The application will open in your browser.

## How It Works

1. Upload a PDF or TXT document.
2. Click "Process Document".
3. The document is loaded and split into smaller chunks.
4. Each chunk is converted into an embedding using the local Sentence Transformers model.
5. The embeddings are stored in ChromaDB.
6. Enter a question about the document.
7. The retriever searches for relevant chunks.
8. The retrieved context is sent to the Groq LLM.
9. The LLM generates an answer based only on the retrieved context.
10. Relevant source chunks can be viewed from the Sources section.

## Important Note

This project is primarily built for learning and experimentation.

The application may take some time to respond because it uses a local embedding model and free API resources. Performance can vary depending on the system and API availability.

## Project Structure

RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── uploads/
│
└── chroma-db/

The `.env`, `uploads/`, and `chroma-db/` directories should not be committed to GitHub.

## Future Improvements

- Support for DOCX and additional document formats
- Chat history
- Multiple document management
- Document deletion
- Better source citations
- Streaming LLM responses
- Improved UI
- Authentication
- Cloud vector database
- Deployment optimization

## Learning Goals

This project demonstrates the core concepts behind a RAG system:

- Document ingestion
- Chunking
- Embeddings
- Vector databases
- Semantic search
- Retrieval
- Context augmentation
- LLM-based generation

## License

This project is intended for educational and learning purposes.

## Author

Built by Burhan Arshad
Computer Science Student