from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
docs=PyPDFLoader("Document_loader/NLP_with_Classical_ML.pdf").load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks=splitter.split_documents(docs)
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=Chroma.from_documents(
    embedding=embedding_model,
    documents=chunks,
    persist_directory="chroma-db"
)