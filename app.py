import streamlit as st
from ingest import process_and_index_codebase
from query import ask_codebase

st.set_page_config(page_title="GitHub Codebase Chatter", layout="wide")
st.title("💻 GitHub Codebase Chatter (Code-Aware RAG)")

with st.sidebar:
    st.header("1. Ingest Repository")
    repo_url = st.text_input("GitHub Repo URL", "https://github.com/pallets/flask")
    collection_name = st.text_input("Weaviate Index Name", "FlaskCodebase")
    
    if st.button("Clone & Index Repository"):
        with st.spinner("Cloning, parsing AST boundaries, and indexing into Weaviate..."):
            process_and_index_codebase(repo_url, index_name=collection_name)
            st.success("Indexing Complete!")

st.header("2. Ask Architectural & Functional Questions")
user_query = st.text_input(
    "Ask a question about the code:", 
    placeholder="Where is the authentication logic handled or where are HTTP exceptions caught?"
)

if st.button("Submit Query") and user_query:
    with st.spinner("Retrieving relevant code chunks & querying Claude 3.5 Sonnet..."):
        answer = ask_codebase(user_query, index_name=collection_name)
        st.markdown("### Claude's Architectural Answer")
        st.write(answer)
