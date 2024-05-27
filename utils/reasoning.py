import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

from utils.llm_api import get_llm


def get_reason_response(results, query, memory,streaming_callback): #chat client function
    

    llm = get_llm(model= "anthropic.claude-3-sonnet-20240229-v1:0" ,streaming_callback = streaming_callback, temperature=0.4)
    prompt = PromptTemplate(input_variables=['history', 'input'], 
                            template="""The following is a conversation between a HR and an AI. 
                            Some CVs are found, the job of the AI is to draw a table that contain information each CVs.
                            For each CVs the AI should give some anlyze whether the HR should consider the CVs or not.
                            Remember to draw table to summarize CV.
                            \n\nCurrent conversation:\n{history}
                            \nJD and CVs: {input}\nAI:""")

    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, 
        prompt = prompt,
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    input_text = query + " The result after search is: " + str(results) 
    
    chat_response = conversation_with_summary.invoke(input_text) #pass the user message and summary to the model
    return chat_response['response']

if __name__ == "__main__":
    input_query = "Tìm kiếm ứng viên có kinh nghiệm làm việc với các framework xây dựng web backend bằng Java"
