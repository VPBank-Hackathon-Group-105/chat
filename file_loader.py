import os
from langchain_community.document_loaders import PyPDFLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def load_docs(root_directory: str):

    # Set the batch size (number of files to process in each batch)
    batch_size = 10

    # Initialize an empty list to store loaded documents
    docs = []

    # Function to process a batch of PDF files
    def process_pdf_batch(pdf_files):
        batch_docs = []
        for pdf_file_path in pdf_files:
            pdf_loader = PyPDFLoader(pdf_file_path)
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

