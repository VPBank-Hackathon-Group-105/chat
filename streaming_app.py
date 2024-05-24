import streamlit as st #all streamlit commands will be available through the "st" alias
import reasoning as glib #reference to local lib script
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from embedding_search_pg import get_index_cv_upload


st.set_page_config(page_title="Chatbot")
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
        "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            st.session_state.vector_index = get_index_cv_upload(pdf_docs)

            st.success('PDF uploaded successfully!', icon="✅")


if input_text: 
    with st.chat_message("user"): 
        st.markdown(input_text) 
    
    st.session_state.chat_history.append({"role":"user", "text":input_text})
    
    callback_handler = StreamHandler(container = st.chat_message("assistant").empty())
    chat_response = glib.get_chat_response(input_text=input_text, memory=st.session_state.memory,streaming_callback=callback_handler)
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 
