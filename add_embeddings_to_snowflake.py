"""
Script to add document embeddings to Snowflake database.

This script:
1. Reads documents from the documents/ directory
2. Generates embeddings for each document
3. Stores embeddings and document text in Snowflake
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
import snowflake.connector

# Load environment variables
load_dotenv()

DEFAULT_SNOWFLAKE_DATABASE = "RAG_DB"
DEFAULT_SNOWFLAKE_SCHEMA = "RAG_SCHEMA"
DEFAULT_EMBED_MODEL = "snowflake-arctic-embed-m-v1.5"


def get_target_database_and_schema() -> Tuple[str, str]:
    """Return target database and schema using env vars or defaults."""
    database = os.getenv("SNOWFLAKE_DATABASE", DEFAULT_SNOWFLAKE_DATABASE)
    schema = os.getenv("SNOWFLAKE_SCHEMA", DEFAULT_SNOWFLAKE_SCHEMA)
    return database, schema


def get_snowflake_connection():
    """Create and return a Snowflake connection."""
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    account = os.getenv("SNOWFLAKE_ACCOUNT")

    connect_kwargs = {
        "user": user,
        "password": password,
        "account": account,
    }

    # Only set optional objects when explicitly provided.
    for env_key, param_key in [
        ("SNOWFLAKE_WAREHOUSE", "warehouse"),
        ("SNOWFLAKE_ROLE", "role"),
    ]:
        value = os.getenv(env_key)
        if value:
            connect_kwargs[param_key] = value

    conn = snowflake.connector.connect(**connect_kwargs)

    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    if warehouse:
        cursor = conn.cursor()
        try:
            cursor.execute(f"USE WAREHOUSE {warehouse}")
        finally:
            cursor.close()

    return conn


def ensure_database_and_schema(conn, database: str, schema: str):
    """Create database/schema if needed, then set them as current context."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cursor.execute(f"USE DATABASE {database}")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        cursor.execute(f"USE SCHEMA {schema}")
    finally:
        cursor.close()


def read_documents(documents_dir: str = "documents") -> List[Tuple[str, str]]:
    """
    Read all text files from the documents directory.
    
    Returns:
        List of tuples: (filename, content)
    """
    docs = []
    docs_path = Path(documents_dir)
    
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory '{documents_dir}' not found")
    
    for file_path in docs_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                docs.append((file_path.name, content))
    
    return docs


def chunk_document_by_paragraphs(text: str) -> List[str]:
    """Split document into paragraph chunks using blank lines."""
    paragraphs = re.split(r"\n\s*\n+", text)
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def create_embeddings_table_if_not_exists(conn, table_name: str = "document_embeddings"):
    """
    Create the embeddings table if it doesn't exist.
    
    Args:
        conn: Snowflake connection
        table_name: Name of the table to create
        Uses ARRAY(FLOAT) for compatibility.
    """
    cursor = conn.cursor()
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id VARCHAR(255) PRIMARY KEY,
        filename VARCHAR(255),
        chunk_id NUMBER,
        document_text VARCHAR,
        embedding ARRAY,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    """
    
    try:
        cursor.execute(create_table_sql)
        print(f"Table '{table_name}' created or already exists.")
    finally:
        cursor.close()


def create_doc_contents_table_if_not_exists(conn, table_name: str = "doc_contents"):
    """
    Create the source document table if it doesn't exist.

    Args:
        conn: Snowflake connection
        table_name: Name of the table to create
    """
    cursor = conn.cursor()

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        file_id NUMBER AUTOINCREMENT START 1 INCREMENT 1,
        file_name VARCHAR(255) UNIQUE,
        content VARCHAR,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    """

    try:
        cursor.execute(create_table_sql)
        print(f"Table '{table_name}' created or already exists.")
    finally:
        cursor.close()


def insert_embeddings(
    conn,
    documents: List[Tuple[str, str]],
    table_name: str = "document_embeddings",
    doc_contents_table: str = "doc_contents",
):
    """
    Insert documents and embeddings into Snowflake.
    
    Args:
        conn: Snowflake connection
        documents: List of (filename, text) tuples
        table_name: Name of the table
        Embeddings are generated using Snowflake Cortex and stored as ARRAY.
        Each paragraph is stored as a separate row with chunk_id.
    """
    cursor = conn.cursor()
    embed_model = os.getenv("SNOWFLAKE_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    total_chunks = 0
    inserted_docs = 0
    skipped_docs = 0
    
    try:
        for filename, text in documents:
            cursor.execute(
                f"SELECT 1 FROM {doc_contents_table} WHERE file_name = %s LIMIT 1",
                (filename,),
            )
            if cursor.fetchone():
                skipped_docs += 1
                print(f"Skipped (already exists in {doc_contents_table}): {filename}")
                continue

            cursor.execute(
                f"INSERT INTO {doc_contents_table} (file_name, content) VALUES (%s, %s)",
                (filename, text),
            )

            chunks = chunk_document_by_paragraphs(text)
            for chunk_id, chunk_text in enumerate(chunks, start=1):
                row_id = f"{filename}_chunk_{chunk_id}"
                insert_sql = f"""
                INSERT INTO {table_name} (id, filename, chunk_id, document_text, embedding)
                SELECT col1, col2, col3, col4, TO_ARRAY(SNOWFLAKE.CORTEX.EMBED_TEXT_768(col5, col4))
                FROM VALUES (%s, %s, %s, %s, %s) AS v(col1, col2, col3, col4, col5)
                """
                cursor.execute(
                    insert_sql,
                    (row_id, filename, chunk_id, chunk_text, embed_model),
                )

            total_chunks += len(chunks)
            inserted_docs += 1
            print(
                f"Inserted: {filename} "
                f"(doc_contents + {len(chunks)} embedding chunks)"
            )
        
        conn.commit()
        print(
            f"\nDone. Inserted {inserted_docs} new documents, "
            f"skipped {skipped_docs} existing documents, "
            f"inserted {total_chunks} paragraph chunks with embeddings."
        )
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting embeddings: {e}")
        raise
    finally:
        cursor.close()


def main():
    """Main function to orchestrate the embedding process."""
    print("Starting document embedding process...")
    
    # Read documents
    print("\n1. Reading documents...")
    documents = read_documents()
    if not documents:
        print("No documents found!")
        return
    
    print(f"Found {len(documents)} documents")
    
    database, schema = get_target_database_and_schema()

    # Connect to Snowflake
    print("\n2. Connecting to Snowflake...")
    try:
        conn = get_snowflake_connection()
        print("Connected successfully!")
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return
    
    # Create database and schema
    print("\n3. Creating/using database and schema...")
    try:
        ensure_database_and_schema(conn, database, schema)
        print(f"Using database '{database}' and schema '{schema}'.")
    except Exception as e:
        print(f"Error creating/using database/schema: {e}")
        conn.close()
        return

    # Create tables
    print("\n4. Creating tables if needed...")
    try:
        create_doc_contents_table_if_not_exists(conn)
        create_embeddings_table_if_not_exists(conn)
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.close()
        return
    
    # Insert embeddings
    print("\n5. Inserting embeddings into Snowflake...")
    try:
        insert_embeddings(conn, documents)
    except Exception as e:
        print(f"Error inserting embeddings: {e}")
    finally:
        conn.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
