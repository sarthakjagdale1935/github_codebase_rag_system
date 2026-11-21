import os
import time
import shutil
import tempfile
from git import Repo
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from dotenv import load_dotenv

load_dotenv()

EXTENSION_LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
}

IGNORED_DIRS = {"tests", "test", "__pycache__", "node_modules", ".git", "venv", ".venv"}


def clone_repo(repo_url: str, target_dir: str):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    Repo.clone_from(repo_url, to_path=target_dir, depth=1)


def process_and_index_codebase(
    repo_url: str,
    index_name: str = "CodebaseChunk",
    progress_callback=None,
):
    temp_dir = tempfile.mkdtemp(prefix="repo_chat_")
    client = None

    try:
        if progress_callback:
            progress_callback("Cloning repository...")
        clone_repo(repo_url, temp_dir)
        all_documents = []

        for ext, lang in EXTENSION_LANGUAGE_MAP.items():
            loader = GenericLoader.from_filesystem(
                temp_dir,
                glob=f"**/*{ext}",
                suffixes=[ext],
                parser=LanguageParser(language=lang, parser_threshold=500),
            )
            docs = loader.load()
            if not docs:
                continue

            filtered_docs = [
                doc
                for doc in docs
                if not any(
                    part in IGNORED_DIRS
                    for part in os.path.relpath(
                        doc.metadata.get("source", ""), temp_dir
                    ).split(os.sep)
                )
            ]
            if not filtered_docs:
                continue

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=1200,
                chunk_overlap=150,
            )
            split_docs = splitter.split_documents(filtered_docs)

            for doc in split_docs:
                full_path = doc.metadata.get("source", "")
                rel_path = os.path.relpath(full_path, temp_dir)
                doc.metadata["file_path"] = rel_path
                doc.metadata["language"] = lang.value

            all_documents.extend(split_docs)

        if not all_documents:
            raise ValueError("No supported source code files were found in the repository.")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )

        # Recreate collection with an explicit vector index so gRPC vector search works
        if client.collections.exists(index_name):
            client.collections.delete(index_name)

        client.collections.create(
            name=index_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="file_path", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
            ],
        )

        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        vector_store = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embeddings,
        )

        batch_size = 20
        total_batches = (len(all_documents) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            batch = all_documents[start : start + batch_size]

            if progress_callback:
                progress_callback(
                    f"Embedding batch {batch_idx + 1}/{total_batches} ({len(batch)} chunks)..."
                )

            for attempt in range(3):
                try:
                    vector_store.add_documents(batch)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt == 2:
                            raise
                        if progress_callback:
                            progress_callback(
                                "Free-tier quota reached (100 req/min). Pausing 65s before retrying..."
                            )
                        time.sleep(65)
                    else:
                        raise

            if batch_idx + 1 < total_batches:
                time.sleep(15)

    finally:
        if client is not None:
            client.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
