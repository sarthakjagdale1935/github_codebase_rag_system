import os
import shutil
import tempfile
from git import Repo
import weaviate
from weaviate.auth import AuthApiKey
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

def clone_repo(repo_url: str, target_dir: str):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    print(f"Cloning repo: {repo_url}...")
    Repo.clone_from(repo_url, to_path=target_dir)
    print("Repository cloned successfully.")

def process_and_index_codebase(repo_url: str, index_name: str = "CodebaseChunk"):
    temp_dir = tempfile.mkdtemp(prefix="repo_chat_")
    
    try:
        clone_repo(repo_url, temp_dir)
        all_documents = []
        
        for ext, lang in EXTENSION_LANGUAGE_MAP.items():
            loader = GenericLoader.from_filesystem(
                temp_dir,
                glob=f"**/*{ext}",
                suffixes=[ext],
                parser=LanguageParser(language=lang, parser_threshold=500)
            )
            docs = loader.load()
            if not docs:
                continue

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, 
                chunk_size=1200, 
                chunk_overlap=150
            )
            split_docs = splitter.split_documents(docs)
            
            for doc in split_docs:
                full_path = doc.metadata.get("source", "")
                rel_path = os.path.relpath(full_path, temp_dir)
                doc.metadata["file_path"] = rel_path
                doc.metadata["language"] = lang.value
            
            all_documents.extend(split_docs)

        print(f"Total syntax-aware code chunks generated: {len(all_documents)}")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )

        # Removed 'models/' prefix from model name
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

        vector_store = WeaviateVectorStore.from_documents(
            documents=all_documents,
            embedding=embeddings,
            client=client,
            index_name=index_name
        )
        print(f"Successfully indexed codebase into Weaviate collection '{index_name}'.")
        
    finally:
        shutil.rmtree(temp_dir)
        print("Cleaned up temporary workspace.")

if __name__ == "__main__":
    target_repo = "https://github.com/pallets/flask"
    process_and_index_codebase(target_repo)