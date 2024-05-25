import streamlit as st

from sqlalchemy import func, insert, or_, select, update

from utils.database import user_cv, get_db, fetch_one
from utils.file_loader import load_uploaded_docs
from utils.upload_file import upload_docs
from utils.summarize_cv import get_summarize_documents
from utils.entity_extraction import get_entities
from utils.embedding_search_pg import get_index_summary

st.title("Upload CVs")

st.subheader("Your documents")
docs = st.file_uploader(
    "Select your files here and click on 'Upload'", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Upload"):
    with st.spinner("Processing"):
        #st.session_state.vector_index = get_index_cv_upload(pdf_docs)
        docs = upload_docs(docs)
        docs = load_uploaded_docs(docs, include_metadata=True)

        with st.spinner("Screening CVs and extract applicant information..."):
            summarize_docs = get_summarize_documents(docs=docs)

            # Save embeddings of summarize docs to the database
            get_index_summary(summarize_docs)

            for summary_doc in summarize_docs:
                #st.write(summary_doc.page_content)

                # Extract entities from the summary
                entities = get_entities(summary_doc.page_content)

                # Update the database with the entities
                db = get_db(is_writer=True)
                query = (
                    update(user_cv)
                    .values({
                        "name": entities.get("name", None),
                        "summary": summary_doc.page_content,
                        "year_of_birth": entities.get("year_of_birth", None),
                        "skills": entities.get("skills", None),
                        "experiences": entities.get("experiences", None),
                        "year_of_experience": entities.get("year_of_experience", None),
                        "educations": entities.get("educations", None),
                        "awards": entities.get("awards", None),
                        "qualifications": entities.get("qualifications", None),
                    })
                    .where(user_cv.c.id == summary_doc.metadata["cv_user_id"])
                    .returning(user_cv)
                )
                result = fetch_one(db.execute(query))
                db.commit()

            st.success('CVs uploaded successfully!', icon="✅")
