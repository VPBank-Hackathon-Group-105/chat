from .llm_api import get_llm

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

def get_memory(): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    llm = get_llm(model = "anthropic.claude-3-sonnet-20240229-v1:0")
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=512) #Maintains a summary of previous messages
    
    return memory

def get_chat_response(input_text, memory,streaming_callback): #chat client function
    
    llm = get_llm(streaming_callback = streaming_callback)
    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, 
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    
    #input_text_with_prompt = system_prompt + "\n" + input_text

    chat_response = conversation_with_summary.invoke(input_text) #pass the user message and summary to the model
    return chat_response['response']





