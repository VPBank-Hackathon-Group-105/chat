from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils.llm_api import get_llm
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

def decide(input_query):

    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0", temperature=0)

    def _parse(text):
        return text.strip('"').strip("**")

    template = PromptTemplate.from_template(
        """
        You are the decider wether to decide  "Yes" or "No".
        Remember, only say "Yes" or "No" and dont say anything else.
        A HR is asking/saying:{input_query}. 
        If the HR do not require to find CVs or seeking for somebody, simply decide "No".
        In case the HR require to find somebody/CV then decide "Yes".
        HR may require to find anotherone if the former query is not good, in this case decide "Yes".
        Decide:"""
    )

    rewriter = template | llm | StrOutputParser() | _parse


    response = rewriter.invoke(input_query)
    print(response)

    return response

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
    chat_response = conversation_with_summary.invoke(input_text) #pass the user message and summary to the model
    return chat_response['response']

def execute_response(input_query, memory, streaming_callback):
    llm_decision = decide(input_query)
    if "no" in llm_decision.lower():
        response = get_chat_response(input_text = input_query, memory = memory, streaming_callback=streaming_callback)
    return response

if __name__ == "__main__":
    input_query = "CV này tệ quá, bạn tìm những người khác có được không?"
    print(decide(input_query))