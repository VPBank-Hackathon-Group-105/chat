import time

from .file_loader import load_docs
from .llm_api import get_embedding

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_index(cv_directory): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = get_embedding(model = "openai")
    
    loader = load_docs(root_directory=cv_directory)

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=10)
    )

    index_from_loader = index_creator.from_documents(loader)

    return index_from_loader

def get_similarity_search_results(index, question, top_k =10):
    results = index.vectorstore.similarity_search_with_score(question, k=top_k)
    flattened_results = [{"content":res[0].page_content, "cv": res[0].metadata["source"]} for res in results] #flatten results for easier display and handling
    #"content":res[0].page_content
    return flattened_results

    # def get_embedding(text):
    #     embeddings = OpenAIEmbeddings(api_key=)  #create a Embeddings client
        
    #     return embeddings.embed_query(text)
if __name__ == "__main__":
    cv_directory = "test_data/1"
    print("Creating index...")
    start = time.time()
    vector_index = get_index(cv_directory = cv_directory)
    end = time.time()
    print(end - start)
    print("Searching for similar documents...")
    response_content = get_similarity_search_results(index=vector_index, question="Python, aws, etl, mlops")
    print(response_content)