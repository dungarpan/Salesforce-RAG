"""
Simple Snowflake-native RAG query script.

Flow:
1. Read a user question
2. Run similarity search on chunked rows in Snowflake
3. Generate an answer with Snowflake Cortex COMPLETE
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
import snowflake.connector
import streamlit as st

load_dotenv()

DEFAULT_SNOWFLAKE_DATABASE = "RAG_DB"
DEFAULT_SNOWFLAKE_SCHEMA = "RAG_SCHEMA"
DEFAULT_TABLE_NAME = "document_embeddings"
DEFAULT_EMBED_MODEL = "snowflake-arctic-embed-m-v1.5"
DEFAULT_COMPLETE_MODEL = "mistral-large2"


def get_snowflake_connection():
    """Create and return a Snowflake connection."""
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )
    return conn


def set_context(conn):
    """Set database and schema context."""
    database = os.getenv("SNOWFLAKE_DATABASE", DEFAULT_SNOWFLAKE_DATABASE)
    schema = os.getenv("SNOWFLAKE_SCHEMA", DEFAULT_SNOWFLAKE_SCHEMA)
    cursor = conn.cursor()
    try:
        cursor.execute(f"USE DATABASE {database}")
        cursor.execute(f"USE SCHEMA {schema}")
    finally:
        cursor.close()


def retrieve_similar_chunks(
    conn, question: str, top_k: int, table_name: str
) -> List[Tuple[str, int, str, float]]:
    """
    Retrieve top-k most similar chunks from Snowflake.

    Uses stored embedding column for similarity search.
    """
    cursor = conn.cursor()
    embed_model = os.getenv("SNOWFLAKE_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    sql_using_stored_embeddings = f"""
    WITH query_embedding AS (
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, %s) AS q_emb
    )
    SELECT
        filename,
        chunk_id,
        document_text,
        VECTOR_COSINE_SIMILARITY(
            embedding::VECTOR(FLOAT, 768),
            (SELECT q_emb FROM query_embedding)
        ) AS similarity
    FROM {table_name}
    ORDER BY similarity DESC
    LIMIT %s
    """

    try:
        cursor.execute(sql_using_stored_embeddings, (embed_model, question, top_k))
        rows = cursor.fetchall()
        return rows
    finally:
        cursor.close()


def generate_answer(conn, question: str, chunks: List[Tuple[str, int, str, float]]) -> str:
    """Generate answer using Snowflake Cortex COMPLETE."""
    model = os.getenv("SNOWFLAKE_COMPLETE_MODEL", DEFAULT_COMPLETE_MODEL)

    context_lines = []
    for filename, chunk_id, chunk_text, similarity in chunks:
        context_lines.append(
            f"[{filename} | chunk {chunk_id} | score={similarity:.4f}] {chunk_text}"
        )

    context = "\n\n".join(context_lines)
    prompt = (
        "You are a helpful assistant answering from provided context only.\n"
        "If the answer is not in the context, say you do not know.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS response",
            (model, prompt),
        )
        response = cursor.fetchone()[0]
        return str(response)
    finally:
        cursor.close()


def main():
    st.set_page_config(page_title="Snowflake RAG", page_icon=":snowflake:")
    st.title("Snowflake RAG")
    st.caption("Similarity search and answer generation powered by Snowflake Cortex.")

    default_table = os.getenv("SNOWFLAKE_EMBED_TABLE", DEFAULT_TABLE_NAME)
    question = st.text_area("Ask a question", placeholder="Type your question here...")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=2)
    table_name = st.text_input("Embeddings table", value=default_table)

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Connecting to Snowflake..."):
                conn = get_snowflake_connection()
            try:
                set_context(conn)
                with st.spinner("Retrieving relevant chunks..."):
                    chunks = retrieve_similar_chunks(conn, question, top_k, table_name)
                if not chunks:
                    st.info("No chunks found. Check table/database/schema values.")
                    return

                with st.spinner("Generating answer..."):
                    answer = generate_answer(conn, question, chunks)
            finally:
                conn.close()

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved chunks")
            for filename, chunk_id, chunk_text, similarity in chunks:
                with st.expander(f"{filename} | chunk {chunk_id} | score={similarity:.4f}"):
                    st.write(chunk_text)
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

