# GitHub Repository Code Chatter

A Code-Aware Retrieval-Augmented Generation (RAG) system built with LangChain, Weaviate, and Anthropic Claude 3.5 Sonnet.

## Project Structure
- `ingest.py`: Handles GitHub repo cloning, AST-aware splitting via LangChain, and vector indexing into Weaviate.
- `query.py`: Implements retrieval and Q&A using Claude 3.5 Sonnet with source file path attribution.
- `app.py`: Streamlit frontend for interactive codebase querying.
- `requirements.txt`: Python package dependencies.
- `.env.example`: Environment key template.

## Execution Instructions
1. Open the project folder in VS Code.
2. Initialize virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install packages: `pip install -r requirements.txt`
5. Configure `.env` with your Weaviate, Anthropic, and OpenAI keys.
6. Run interface: `streamlit run app.py`
