from llm_api import get_llm
from file_loader import load_docs
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from io import StringIO
import pandas as pd

def summary_llm(input_query):
    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0", temperature=0)

    template = PromptTemplate.from_template(
        """Given this as a part of CV:{text}\n\n 
        Write 1-5 sentences to summarize the CV, and it better including (if any) the following information: 
        Name, year of birth, skills, experiences and years of experience, education, award and qualifications. 
        If any of those information is missing then do not include it, do not say anything about it in the summary.
        Importance: Be as short as possible but at most specific.
        Summarization:"""
    )

    summarizer = template | llm | StrOutputParser() 

    response = summarizer.invoke(input_query)

    return response    

def get_summary(docs):
    temp = docs[0].metadata['source']
    concat_summarize = []
    temp_summarize = ""
    for doc in docs:
        if doc.metadata['source'] == temp:
            temp_summarize = temp_summarize + summary_llm(doc.page_content)
            temp_summarize += " "
        else:
            concat_summarize.append({"cv": temp_summarize, "source": temp})
            temp = doc.metadata['source']
            temp_summarize = summary_llm(doc.page_content)
            temp_summarize += " "
    concat_summarize.append({"cv": temp_summarize, "source": temp})
    return concat_summarize

if __name__ == "__main__":
    docs = load_docs(root_directory="test_data/", is_split=False)    
    print(get_summary(docs=docs))
    
            