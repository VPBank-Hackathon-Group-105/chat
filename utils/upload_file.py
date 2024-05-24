import os
import tempfile
import boto3

s3 = boto3.resource('s3')
bucket = os.environ.get('BUCKET_NAME')

def upload_docs(uploaded_files: list):
    files_with_metadata = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_name = uploaded_file.name
            temp_file.write(uploaded_file.getvalue())
            s3_file_path = f"cv/{file_name}"
            s3.Bucket(bucket).upload_file(temp_file.name, s3_file_path)
            files_with_metadata.append({"file": uploaded_file, "cv_file_path": s3_file_path})

    return files_with_metadata
