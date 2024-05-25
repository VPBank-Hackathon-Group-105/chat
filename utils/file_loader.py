import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
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


def load_uploaded_docs(uploaded_files: list, include_metadata: bool = False):

    # Set the batch size (number of files to process in each batch)
    batch_size = 10

    # Initialize an empty list to store loaded documents
    docs = []

    # Function to process a batch of PDF files
    def process_pdf_batch(pdf_files, is_split = False):
        batch_docs = []
        for pdf_file in pdf_files:
            pdf_loader = PyPDFLoader(pdf_file[0] if include_metadata else pdf_file)
            if is_split:
                loaded_docs = pdf_loader.load_and_split(
                    text_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=int(os.environ.get("CHUNK_SIZE", 300)),
                        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 10)),
                    )
                )
            else:
                loaded_docs = pdf_loader.load()
            # Remove NUL characters from each loaded document
            for doc in loaded_docs:
                doc.page_content = doc.page_content.replace('\x00', '')
                if include_metadata:
                    doc.metadata.update(pdf_file[1])
                batch_docs.append(doc)
                
        return batch_docs
    
    def process_docx_batch(docx_files):
        batch_docs = []
        for docx_file in docx_files:
            docx_loader = Docx2txtLoader(docx_file if include_metadata else docx_file[0])
            loaded_docs = docx_loader.load()

        for doc in loaded_docs:
                doc.page_content = doc.page_content.replace('\x00', '')
                if include_metadata:
                    doc.metadata.update(docx_file[1])
                batch_docs.append(doc)
        return batch_docs

    # Get the list of PDF files to process
    pdf_files_to_process = []
    if not include_metadata:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    pdf_files_to_process.append(temp_file.name)
    else:
        for uploaded_file, metadata in uploaded_files:
            if uploaded_file.lower().endswith(".pdf"):
                pdf_files_to_process.append((uploaded_file, metadata))

    docx_files_to_process = []
    if not include_metadata:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith(".doc") or uploaded_file.name.lower().endswith(".docx") :
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    docx_files_to_process.append(temp_file.name)
    else:
        for uploaded_file, metadata in uploaded_files:
            if uploaded_file.lower().endswith(".doc") or uploaded_file.lower().endswith(".docx"):
                docx_files_to_process.append((uploaded_file, metadata))

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        pdf_total_files = len(pdf_files_to_process)

        # Iterate through the PDF files in batches
        for i in tqdm(range(0, pdf_total_files, batch_size)):
            batch = pdf_files_to_process[i:i+batch_size]
            batch_docs = list(executor.map(process_pdf_batch, [batch]))
            for batch_result in batch_docs:
                docs.extend(batch_result)

        docx_total_files = len(docx_files_to_process)

        # Iterate through the docx files in batches
        for i in tqdm(range(0, docx_total_files, batch_size)):
            batch = docx_files_to_process[i:i+batch_size]
            batch_docs = list(executor.map(process_docx_batch, [batch]))
            for batch_result in batch_docs:
                docs.extend(batch_result)

    return docs

if __name__ == "__main__":
    # Test the load_docs function
    docs = load_docs(root_directory="test_data/try", is_split=True)
    print(f"Loaded {len(docs)} documents")
    print(docs[1])
    print(len(docs[1].page_content))

