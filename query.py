import os
import weaviate
from weaviate.auth import AuthApiKey
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert principal software engineer analyzing a codebase.
Answer the developer's question using ONLY the retrieved code snippets provided below.
For each file or logic you cite, explicitly mention the file path (`file_path`).

Retrieved Code Context:
{context}
"""


def create_rag_chain(index_name: str = "CodebaseChunk"):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    if not client.collections.exists(index_name):
        client.close()
        raise ValueError(
            f"Collection '{index_name}' was not found in Weaviate. Click 'Clone & Index Repository' in the sidebar first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vector_store = WeaviateVectorStore(
        client=client,
        index_name=index_name,
        text_key="text",
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
    )

    return client, retriever, llm


def ask_codebase(query: str, index_name: str = "CodebaseChunk"):
    client = None
    try:
        client, retriever, llm = create_rag_chain(index_name)
        retrieved_docs = retriever.invoke(query)

        context_blocks = []
        for doc in retrieved_docs:
            file_path = doc.metadata.get("file_path", "Unknown File")
            lang = doc.metadata.get("language", "code")
            block = f"--- File: {file_path} ({lang}) ---\n{doc.page_content}\n"
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        chain = prompt | llm
        response = chain.invoke({"context": context_str, "question": query})
        return response.content
    finally:
        if client is not None:
            client.close()
