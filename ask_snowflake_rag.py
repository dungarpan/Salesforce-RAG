"""
Simple Snowflake-native RAG query script.

Flow:
1. Read a user question
2. Run similarity search on chunked rows in Snowflake
3. Generate an answer with Snowflake Cortex COMPLETE
"""

import os
import argparse
from typing import List, Tuple

from dotenv import load_dotenv
import snowflake.connector

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
    parser = argparse.ArgumentParser(description="Ask questions using Snowflake-native RAG.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--table",
        type=str,
        default=os.getenv("SNOWFLAKE_EMBED_TABLE", DEFAULT_TABLE_NAME),
        help="Table name containing chunked documents and embeddings.",
    )
    args = parser.parse_args()

    print("Connecting to Snowflake...")
    conn = get_snowflake_connection()
    try:
        set_context(conn)
        print(f"Searching top {args.top_k} chunks from '{args.table}'...")
        chunks = retrieve_similar_chunks(conn, args.question, args.top_k, args.table)
        if not chunks:
            print("No chunks found. Check table/database/schema values.")
            return

        print("\nTop chunks:")
        for filename, chunk_id, _, similarity in chunks:
            print(f"- {filename} | chunk {chunk_id} | score={similarity:.4f}")

        print("\nGenerating answer with Snowflake Cortex...")
        answer = generate_answer(conn, args.question, chunks)
        print("\nAnswer:\n")
        print(answer)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

