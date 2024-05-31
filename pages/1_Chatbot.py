import streamlit as st
import utils.agent as agent
from utils.embedding_search_pg import get_index_cv_upload
from langchain_core.callbacks import BaseCallbackHandler

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
