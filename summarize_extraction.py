from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import llm_api
import file_loader

def get_summary(return_intermediate_steps=False,docs=None):
    map_prompt_template = "{text}\n\nWrite a few sentences summarizing the above:"
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    
    combine_prompt_template = "{text}\n\nWrite a detailed analysis of the above:"
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
    
    
    llm = llm_api.get_llm(model="meta.llama3-8b-instruct-v1:0")
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, return_intermediate_steps=return_intermediate_steps)
    
    if return_intermediate_steps:
        return chain.invoke({"input_documents": docs}, return_only_outputs=True)
    else:
        return chain.invoke(docs, return_only_outputs=True)
    
if __name__ == "__main__":
    docs = file_loader.load_docs(root_directory="test_data/1")
    print(get_summary(docs=docs))