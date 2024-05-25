import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from .query import query_cv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

from utils.llm_api import get_llm
from utils.reasoning import get_reason_response
from utils.retransforming import retransform
from utils.embedding_search_pg import get_similarity_search_results
from cohere_aws import Client

co = Client(region_name="us-east-1")
co.connect_to_endpoint(endpoint_name="cohere-rerank-v3-endpoint")

def decide(input_query):

    llm = get_llm(model = "anthropic.claude-3-sonnet-20240229-v1:0", temperature=0)

    def _parse(text):
        return text.strip('"').strip("**")

    template = PromptTemplate.from_template(
        """
        You are the decider whether to decide  "Yes" or "No" to start a tool finding for suitable applicant.
        Remember, only say "Yes" or "No" and dont say anything else.
        A HR is asking/saying:{input_query}. 
        If the HR do not require to find CVs or seeking for somebody new, simply decide "No".
        In case the HR require to find somebody/CV then decide "Yes".
        HR may require to find anotherone if the former query is not good, in this case decide "Yes".
        HR may want to discuss more about the person that assistant chose, now simply decide "No".
        Decide:"""
    )

    rewriter = template | llm | StrOutputParser() | _parse


    response = rewriter.invoke(input_query)

    return response

def get_memory(): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0")
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=512) #Maintains a summary of previous messages
    
    return memory

def get_chat_response(input_text, memory, streaming_callback): #chat client function
    
    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0",streaming_callback = streaming_callback)

    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm,
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    chat_response = conversation_with_summary.invoke(input_text) #pass the user message and summary to the model
    return chat_response['response']

def execute_response(input_query, index, memory, streaming_callback):
    llm_decision = decide(input_query)

    if "no" in llm_decision.lower():
        response = get_chat_response(input_text = input_query, memory = memory, streaming_callback=streaming_callback)
        return response
    else:   
        retransformed_query = retransform(input_query)
        search_results = get_similarity_search_results(index=index, question = retransformed_query, top_k = 20)
        rerank_results = co.rerank(documents=search_results, query=retransformed_query, rank_fields=['content'], top_n=5)
        #after get rerank results, we get the entities from database
        # print(rerank_results[0])
        # print(rerank_results[0].document['cv'])
        cv_ids = [result.document['cv'] for result in rerank_results]
        cv_information = query_cv(cv_ids)
        #final step: get the reasoning from agent
        response = get_reason_response(results = cv_information['data'], query = input_query, memory = memory, streaming_callback=streaming_callback)
        return response

    

if __name__ == "__main__":
    input_query = "CV này tệ quá, bạn tìm những người khác có được không?"
    