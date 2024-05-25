import streamlit as st

from sqlalchemy import func, insert, or_, select, update

from utils.database import user_cv, get_db, fetch_one, update_cv_user
from utils.file_loader import load_uploaded_docs
from utils.upload_file import upload_docs
from utils.summarize_cv import get_summarize_documents
from utils.entity_extraction import get_entities
from utils.embedding_search_pg import get_index_summary

st.title("Upload CVs")

st.subheader("Your documents")
docs = st.file_uploader(
    "Select your files here and click on 'Upload'", type=["pdf", "docx","xlsx","xls"], accept_multiple_files=True)

if st.button("Upload"):
    with st.spinner("Processing"):
        #st.session_state.vector_index = get_index_cv_upload(pdf_docs)
        with st.spinner("Uploading CVs..."):
            docs = upload_docs(docs)
            docs = load_uploaded_docs(docs, include_metadata=True)

        with st.spinner("Screening CVs..."):
            summarize_docs = get_summarize_documents(docs=docs)

        with st.spinner("Extracting applicant information ..."):
            # Save embeddings of summarize docs to the database
            get_index_summary(summarize_docs)

            for summary_doc in summarize_docs:

                # Extract entities from the summary
                entities = get_entities(summary_doc.page_content)
                for key, value in entities.items():
                    if isinstance(value, str) and value.lower() == "null":
                        entities[key] = None

                # Update the database with the entities
                db = get_db(is_writer=True)
                update_cv_user(db, summary_doc.metadata["cv_user_id"], entities)

            st.success('CVs uploaded successfully!', icon="✅")
