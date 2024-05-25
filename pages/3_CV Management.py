import streamlit as st
import utils.agent as agent
from utils.embedding_search_pg import get_index_cv_upload
from utils.database import get_db, get_cv_users
from langchain_core.callbacks import BaseCallbackHandler

st.set_page_config(page_title="Cv Management")
st.title("CV Management") #page title

per_page = st.number_input("Enter the number of CVs to display", value=10)
page = st.number_input("Enter the page number", value=1)

if st.button("Get data"):
    with st.spinner("Fetching data..."):
        df = get_cv_users(per_page=per_page, page=page)
        st.data_editor(
            df,
            column_config={
                "cv_file_path": st.column_config.LinkColumn("CV URL")
            },
            hide_index=True,
        )
