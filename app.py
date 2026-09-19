import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

CHROMA_PATH = "chroma-db"
UPLOAD_PATH = "uploads"

Path(UPLOAD_PATH).mkdir(exist_ok=True)
Path(CHROMA_PATH).mkdir(exist_ok=True)

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("RAG Assistant")
st.write("Upload a document and ask questions about its content.")

st.info(
    "Note: This project may take a little longer to respond because "
    "it uses a local embedding model and free APIs. This is a learning project."
)


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=load_embedding_model()
    )


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b"
    )


embedding_model = load_embedding_model()
vector_store = load_vector_store()
llm = load_llm()


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful assistant.

Answer the user's question using ONLY the context
provided from the uploaded documents.

Give concise and clear answers.

If the answer is not present in the provided context,
say "I don't know".

Do not make up information.
"""
    ),
    (
        "human",
        """
Context:

{context}

Question:

{question}
"""
    )
])


def process_document(uploaded_file):
    file_extension = Path(uploaded_file.name).suffix.lower()
    file_path = Path(UPLOAD_PATH) / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    if file_extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

    elif file_extension == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )
        docs = loader.load()

    else:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    for chunk in chunks:
        chunk.metadata["filename"] = uploaded_file.name
        chunk.metadata["file_hash"] = file_hash

    ids = [
        f"{file_hash}_{index}"
        for index in range(len(chunks))
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return len(chunks)


st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file:

    st.write(f"Selected: **{uploaded_file.name}**")

    if st.button("Process Document"):

        with st.spinner("Processing document..."):
            result = process_document(uploaded_file)

        if result:
            st.success(
                f"Document processed successfully. {result} chunks created."
            )
        else:
            st.error("Unsupported file type.")


st.divider()

st.subheader("Ask a Question")

query = st.text_input(
    "Your question",
    placeholder="What is this document about?"
)

if st.button("Ask"):

    if not query:
        st.warning("Please enter a question.")

    else:

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        with st.spinner("Searching documents..."):
            docs = retriever.invoke(query)

        if not docs:
            st.warning("No relevant information was found.")

        else:

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            with st.spinner("Generating answer..."):
                response = llm.invoke(final_prompt)

            st.subheader("Answer")
            st.write(response.content)

            with st.expander("View Sources"):

                for index, doc in enumerate(docs):

                    st.markdown(
                        f"**Source {index + 1}**"
                    )

                    st.write(doc.page_content)

                    st.caption(
                        doc.metadata.get(
                            "filename",
                            "Unknown"
                        )
                    )


st.divider()

st.caption(
    "Local Embeddings: all-MiniLM-L6-v2 • "
    "Vector Database: ChromaDB • "
    "LLM: Groq GPT-OSS 120B"
)