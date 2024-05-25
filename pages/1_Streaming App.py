import streamlit as st
import utils.agent as agent
from utils.upload_file import upload_docs
from utils.embedding_search_pg import get_index_cv_upload
from langchain_core.callbacks import BaseCallbackHandler
from utils.summarize_cv import get_summary
from utils.entity_extraction import get_entities
from utils.file_loader import load_uploaded_docs

st.set_page_config(page_title="Chatbot")
st.title("Chatbot") #page title


if 'memory' not in st.session_state: 
    st.session_state.memory = agent.get_memory() 

if 'chat_history' not in st.session_state: 
    st.session_state.chat_history = [] 

if 'vector_index' not in st.session_state:
    st.session_state.vector_index = get_index_cv_upload(uploaded_files=[])

for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]):
        st.markdown(message["text"]) 

input_text = st.chat_input("Chat with your bot here") 
        
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Select your files here and click on 'Upload'", accept_multiple_files=True)

    docs = load_uploaded_docs(pdf_docs)

    if st.button("Upload"):
        with st.spinner("Screening CVs..."):
            summaries = get_summary(docs=docs)
            for summary in summaries:
                st.write(summary['cv'])
        with st.spinner("Extracting applicant information..."):
            #to be modify: upload entites to the database
            #this is only test-case. (Anh sửa đoạn này giúp e nhé)
            entities = get_entities(summaries[0]['cv'])

        with st.spinner("Uploading..."):
            #upload summaries to database (anh sửa đoạn này giúp e nhé, các summries của từng CVs đã được lấy ở trên nhưng ở dạng dictionary.)
            st.session_state.vector_index = get_index_cv_upload(summaries)
            upload_docs(pdf_docs)
            st.success('CVs uploaded successfully!', icon="✅")

if input_text: 
    with st.chat_message("user"): 
        st.markdown(input_text) 
    
    st.session_state.chat_history.append({"role":"user", "text":input_text})
    
    #need an Agent here
    callback_handler = StreamHandler(container = st.chat_message("assistant").empty())    


    #print(rerank_results[0].document['content'])
    #st.write(rerank_results)
    with st.spinner("Thinking..."):
        chat_response = agent.execute_response(input_query=input_text, index=st.session_state.vector_index, memory=st.session_state.memory,streaming_callback=callback_handler)
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 
