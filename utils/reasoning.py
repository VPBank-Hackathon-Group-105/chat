from utils.llm_api import get_llm
from utils.retransforming import retransform
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from utils.embedding_search_pg import get_similarity_search_results
from cohere_aws import Client

co = Client(region_name="us-east-1")
co.connect_to_endpoint(endpoint_name="cohere-rerank-v3-endpoint")

def get_memory(): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0")
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=512) #Maintains a summary of previous messages
    
    return memory

def get_search_results(input_text, index): #search client function
    retransformed_query = retransform(input_text)
    search_results = get_similarity_search_results(index=index, question = retransformed_query, top_k = 20)
    rerank_results = co.rerank(documents=search_results, query=retransformed_query, rank_fields=['content'], top_n=5)

    return rerank_results
    
def get_chat_response(input_text, index, memory,streaming_callback): #chat client function
    

    llm = get_llm(model= "anthropic.claude-3-sonnet-20240229-v1:0" ,streaming_callback = streaming_callback)

    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, 
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    
    #input_text_with_prompt = system_prompt + "\n" + input_text

    chat_response = conversation_with_summary.invoke(input_text) #pass the user message and summary to the model
    return chat_response['response']

if __name__ == "__main__":
    input_query = "Tìm kiếm ứng viên có kinh nghiệm làm việc với các framework xây dựng web backend bằng Java"
    print(get_search_results(input_query, []))
    print(get_chat_response(input_query, [], get_memory()))
