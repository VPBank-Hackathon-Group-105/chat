import tempfile
import boto3

client = boto3.client('s3')
bucket = "mpqhdemobucket"

def upload_docs(uploaded_files: list):


    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_name = uploaded_file.name
            temp_file.write(uploaded_file.getvalue())
            client.upload_file(temp_file, 'mpqhdemobucket', file_name)
    # for pdf_file in pdf_files_to_process:
    #     file_name = pdf_file.split('\\')[-1]  # Get the name of the file
    #     file_data = open(pdf_file, 'rb')
    #     print(file_data)
    #     client.upload_file(file_data, 'mpqhdemobucket', file_name)  # Use the file name as the Key
    return 
