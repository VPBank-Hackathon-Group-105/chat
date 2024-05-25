import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def load_docs(root_directory: str, is_split: bool = False):

    # Set the batch size (number of files to process in each batch)
    batch_size = 10

    # Initialize an empty list to store loaded documents
    docs = []

    # Function to process a batch of PDF files
    def process_pdf_batch(pdf_files):
        batch_docs = []
        for pdf_file_path in pdf_files:
            pdf_loader = PyPDFLoader(pdf_file_path)
            if is_split:
                batch_docs.extend(pdf_loader.load_and_split(
                    text_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=int(os.environ.get("CHUNK_SIZE",300)),
                        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP",10)),
                    )
                ))
            else:
                batch_docs.extend(pdf_loader.load())
        return batch_docs

    # Get the list of PDF files to process
    pdf_files_to_process = []
    for root, dirs, files in os.walk(root_directory):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        total_files = len(pdf_files_to_process)
        processed_files = 0

        # Iterate through the PDF files in batches
        for i in tqdm(range(0, total_files, batch_size)):
            batch = pdf_files_to_process[i:i+batch_size]
            batch_docs = list(executor.map(process_pdf_batch, [batch]))
            for batch_result in batch_docs:
                docs.extend(batch_result)
                processed_files += len(batch)
    return docs


def load_uploaded_docs(uploaded_files: list):

    # Set the batch size (number of files to process in each batch)
    batch_size = 10

    # Initialize an empty list to store loaded documents
    docs = []

    # Function to process a batch of PDF files
    def process_pdf_batch(pdf_files):
        batch_docs = []
        for pdf_file_path in pdf_files:
            pdf_loader = PyPDFLoader(pdf_file_path)
            loaded_docs = pdf_loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=int(os.environ.get("CHUNK_SIZE", 300)),
                    chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 10)),
                )
            )
            # Remove NUL characters from each loaded document
            for doc in loaded_docs:
                doc.page_content = doc.page_content.replace('\x00', '')
                batch_docs.append(doc)
        print(batch_docs)
                
        return batch_docs

    # Get the list of PDF files to process
    pdf_files_to_process = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            pdf_files_to_process.append(temp_file.name)

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        total_files = len(pdf_files_to_process)
        processed_files = 0

        # Iterate through the PDF files in batches
        for i in tqdm(range(0, total_files, batch_size)):
            batch = pdf_files_to_process[i:i+batch_size]
            batch_docs = list(executor.map(process_pdf_batch, [batch]))
            for batch_result in batch_docs:
                docs.extend(batch_result)
                processed_files += len(batch)
    print(docs[0])
    return docs

if __name__ == "__main__":
    # Test the load_docs function
    docs = load_docs(root_directory="test_data/try", is_split=True)
    print(f"Loaded {len(docs)} documents")
    print(docs[1].page_content)
    print(len(docs[1].page_content))

