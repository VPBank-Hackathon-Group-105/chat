import os
import time

from .file_loader import load_docs, load_uploaded_docs
from .llm_api import get_embedding

from dotenv import load_dotenv
from langchain.vectorstores.pgvector import PGVector


load_dotenv()

WRITER_CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver = os.environ.get("PGVECTOR_DRIVER"),
    user = os.environ.get("PGVECTOR_USER"),
    password = os.environ.get("PGVECTOR_PASSWORD"),
    host = os.environ.get("PGVECTOR_WRITER_HOST"),
    port = os.environ.get("PGVECTOR_PORT"),
    database = os.environ.get("PGVECTOR_DATABASE")
)

READER_CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver = os.environ.get("PGVECTOR_DRIVER"),
    user = os.environ.get("PGVECTOR_USER"),
    password = os.environ.get("PGVECTOR_PASSWORD"),
    host = os.environ.get("PGVECTOR_READER_HOST"),
    port = os.environ.get("PGVECTOR_PORT"),
    database = os.environ.get("PGVECTOR_DATABASE")
)


def get_index_cv_directory(cv_directory = None): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = get_embedding(model = "openai")
    
    loader = load_docs(root_directory=cv_directory, is_split=True)

    if cv_directory is None:
        PGVector.from_documents(
            documents=loader,
            embedding=embeddings,
            connection_string=WRITER_CONNECTION_STRING,
        )

    return PGVector(
        connection_string=READER_CONNECTION_STRING,
        embedding_function=embeddings
    )

def get_index_cv_upload(uploaded_files: list = []):
    
    embeddings = get_embedding(model = "openai")
    
    if len(uploaded_files) > 0:
        loader = load_uploaded_docs(uploaded_files)
        PGVector.from_documents(
            documents=loader,
            embedding=embeddings,
            connection_string=WRITER_CONNECTION_STRING,
        )

    return PGVector(
        connection_string=READER_CONNECTION_STRING,
        embedding_function=embeddings
    )

def get_similarity_search_results(index: PGVector, question: str, top_k: int = 10):
    results = index.similarity_search_with_score(question, k=top_k)
    flattened_results = [{"content":res[0].page_content, "cv": res[0].metadata["source"]} for res in results] #flatten results for easier display and handling

    return flattened_results

if __name__ == "__main__":
    cv_directory = "test_data/"
    print("Creating index...")
    start = time.time()
    vector_index = get_index_cv_directory(cv_directory = cv_directory)
    end = time.time()
    print(end - start)
    print("Searching for similar documents...")
    response_content = get_similarity_search_results(index=vector_index, question="Python, aws, etl, mlops")
    print(response_content)
