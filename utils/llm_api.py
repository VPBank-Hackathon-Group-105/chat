import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from langchain_openai import ChatOpenAI
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings

OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] #hide this

def get_llm(streaming_callback = None, model: str = "gpt-3.5-turbo-0125", temperature = 0.7):
    if "gpt" in model:
        if streaming_callback:
            llm = ChatOpenAI(
                model = model, 
                temperature= temperature, 
                api_key=OPENAI_API_KEY, 
                callbacks=[streaming_callback],
                streaming=True)    
        else:
            llm = ChatOpenAI(
                model = model, 
                temperature= 0, 
                api_key=OPENAI_API_KEY)
    
    # Bedrock model, configure for each models, here use llama and anthropic models only        
    else:
        #llama model
        if "meta" in model:
            model_kwargs = {
            "max_gen_len": 1024,
            "temperature": temperature,
            "top_p": 0.01,
            }

            if streaming_callback:
                llm = BedrockLLM(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=True,
                    callbacks=[streaming_callback],
                    region_name = "us-east-1"
                )
            else:
                llm = BedrockLLM(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=False,
                    region_name = "us-east-1"
                )

        #anthropic model
        if "anthropic" in model:
            model_kwargs = {
            "max_tokens": 1024,
            "temperature": temperature, 
            "top_k": 250, 
            "top_p": 1, 
            }

            if streaming_callback:
                llm = ChatBedrock(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=True,
                    callbacks=[streaming_callback],
                    region_name = "us-east-1"
                )
            else:
                llm = ChatBedrock(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=False,
                    region_name = "us-east-1"
                )
    return llm

def get_embedding(text= None, model: str = "openai"):
    if "openai" in model:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")  #create a Embeddings client

    else:
        embeddings = BedrockEmbeddings(region_name = "us-east-1") 
    
    if text:
        return embeddings.embed_query(text)
    else:
        return embeddings

if __name__ == "__main__":
    vector = get_embedding(text="Hello world")
    print(len(vector))