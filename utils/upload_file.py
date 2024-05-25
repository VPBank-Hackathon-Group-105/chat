import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import tempfile
import boto3
from dotenv import load_dotenv

load_dotenv()

from utils.database import user_cv, get_db, fetch_one
from sqlalchemy import insert

db = get_db(is_writer=True)
s3 = boto3.resource('s3')
bucket = os.environ.get('BUCKET_NAME')

def upload_docs(uploaded_files: list):
    files_with_metadata = []

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            # Upload to S3
            file_name = uploaded_file.name
            temp_file.write(uploaded_file.getvalue())
            s3_file_path = f"cv/{file_name}"
            s3.Bucket(bucket).upload_file(temp_file.name, s3_file_path)

            # Insert metadata into database
            query = (
                insert(user_cv)
                .values(
                    {
                        "cv_file_path": s3_file_path,
                    }
                )
                .returning(user_cv)
            )
            result = fetch_one(db.execute(query))
            db.commit()

            # Append metadata to list
            files_with_metadata.append(
                (
                    temp_file.name, {
                        "source": s3_file_path,
                        "cv_user_id": result["id"],
                    }
                )
            )

    return files_with_metadata
