from utils.llm_api import get_llm
from utils.retransforming import retransform
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from utils.embedding_search_pg import get_similarity_search_results
from langchain_core.prompts import PromptTemplate


    
def get_reason_response(results, text, memory,streaming_callback): #chat client function
    

    llm = get_llm(model= "anthropic.claude-3-sonnet-20240229-v1:0" ,streaming_callback = streaming_callback)
    prompt = PromptTemplate(input_variables=['history', 'input'], 
                            template="""The following is a conversation between a HR and an AI. 
                            Some CVs are found, the job of the AI is to draw a table that summarize each CVs.
                            For each CVs the AI should give some anlyze whether the HR should consider the CVs or not.
                            \n\nCurrent conversation:\n{history}
                            \nJD and CVs: {input}\nAI:""")

    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, 
        prompt = prompt,
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    input_text = str(results) + "\n" + text
    
    chat_response = conversation_with_summary.invoke(input_text) #pass the user message and summary to the model
    return chat_response['response']

if __name__ == "__main__":
    input_query = "Tìm kiếm ứng viên có kinh nghiệm làm việc với các framework xây dựng web backend bằng Java"
