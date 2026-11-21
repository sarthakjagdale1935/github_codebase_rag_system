#  GitHub Codebase Chatter (Code-Aware RAG)

A code-aware Retrieval-Augmented Generation (RAG) web application that lets you index any public GitHub repository and ask natural-language questions about its architecture, logic, and implementation—complete with exact source file citations.

A Code-Aware Retrieval-Augmented Generation (RAG) system built with LangChain, Weaviate, and Google Gemini API.

🔗 **Live App:** [https://appcodebaseragsystem-kmjxadi5y9dptwhd2xsyru.streamlit.app/](https://appcodebaseragsystem-kmjxadi5y9dptwhd2xsyru.streamlit.app/)

---
![Uploading image.png…]()

##  Key Features

- **Syntax-Aware Code Chunking:** Uses LangChain's `LanguageParser` and language-specific splitters to split code along class and function boundaries across `.py`, `.js`, `.ts`, `.cpp`, `.go`, `.java`, and `.rs` files.
- **Persistent Cloud Vector Storage:** Stores code embeddings and file metadata (`file_path`, `language`) in **Weaviate Cloud** so indexed repositories remain searchable across sessions.
- **Grounded Architectural Q&A:** Powered by **Google Gemini** (`gemini-embedding-001` and `gemini-2.5-flash`) to retrieve relevant code snippets and generate answers that cite exact repository file paths.
- **Rate-Safe Batch Ingestion:** Automatically batches embedding requests and handles API rate limits smoothly during repository indexing.

---

## Project Structure
- `ingest.py`: Handles GitHub repo cloning, AST-aware splitting via LangChain, and vector indexing into Weaviate.
- `query.py`: Implements vector retrieval and Q&A using Google Gemini API with source file path attribution.
- `app.py`: Streamlit frontend for interactive codebase querying.
- `requirements.txt`: Python package dependencies.
- `.env.example`: Environment key template.

