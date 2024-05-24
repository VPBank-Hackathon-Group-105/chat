import reasoning as glib #reference to local lib script
from langchain_core.callbacks import BaseCallbackHandler
from embedding_search_pg import get_index_cv_upload, get_similarity_search_results
import streamlit as st
from retransforming import retransform
from cohere_aws import Client
from upload_file import upload_docs

co = Client(region_name="us-east-1")
co.connect_to_endpoint(endpoint_name="cohere-rerank-v3-endpoint")

#st.set_page_config(page_title="Chatbot")
st.title("Chatbot") #page title


if 'memory' not in st.session_state: 
    st.session_state.memory = glib.get_memory() 

if 'chat_history' not in st.session_state: 
    st.session_state.chat_history = [] 


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
        "Select your files here and click on 'Upload'", type="pdf", accept_multiple_files=True)


    if st.button("Upload"):
        with st.spinner("Processing"):
            #st.session_state.vector_index = get_index_cv_upload(pdf_docs)
            upload_docs(pdf_docs)
            st.success('PDF uploaded successfully!', icon="✅")

if input_text: 
    with st.chat_message("user"): 
        st.markdown(input_text) 
    
    st.session_state.chat_history.append({"role":"user", "text":input_text})
    
    #need an Agent here
    callback_handler = StreamHandler(container = st.chat_message("assistant").empty())    
    retransformed_query = retransform(input_text)
    search_results = get_similarity_search_results(index=st.session_state.vector_index, question = retransformed_query, top_k = 20)
    rerank_results = co.rerank(documents=search_results, query=retransformed_query, rank_fields=['content'], top_n=5)

    st.write(rerank_results)
    
    #chat_response = glib.get_chat_response(input_text=input_text, memory=st.session_state.memory,streaming_callback=callback_handler)
    
    #st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 
