from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma-db"
    )

retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
        }
)

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the context provided. and Answers should be in a concise and clear manner and from Document. If it is not in the document, say 'I don't know'"),
    ("human", "{context}\n\nQuestion: {question}")
])

print("------------------ RAG with Groq ------------------")
print("exit to quit")
while True:
    query=input("You: ")
    if query.lower() == "exit":
        break
    docs=retriever.invoke(query)
    context= "\n\n".join([doc.page_content for doc in docs])
    final_prompt=prompt.format_prompt(context=context, question=query)
    response=llm.invoke(final_prompt.to_string())
    print("Assistant: ", response.content)