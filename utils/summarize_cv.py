import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from io import StringIO

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm_api import get_llm
from utils.file_loader import load_docs

import pandas as pd

def summary_llm(input_query):
    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0", temperature=0)

    template = PromptTemplate.from_template(
        """Given this as a part of CV:{text}\n\n 
        Write 1-4 sentences to summarize the CV, and it better including (if any) the following information: 
        Name, year of birth, skills, experiences and years of experience, education, award and qualifications. 
        If any of those information is missing then do not include it, do not say anything about it in the summary.
        Importance: Be as short as possible but at most specific, if the CVs use Vietnamese then summarize in English.
        Summarization:"""
    )

    summarizer = template | llm | StrOutputParser() 

    response = summarizer.invoke(input_query)

    return response    

def get_summarize_documents(docs):
    if len(docs) == 0:
        return []

    temp_metadata = docs[0].metadata
    concat_summarize = []
    temp_summarize = ""
    for doc in docs:
        if doc.metadata['source'] == temp_metadata['source']:
            temp_summarize = temp_summarize + summary_llm(doc.page_content)
            temp_summarize += " "
        else:
            # Set result as a document, not text
            concat_summarize.append(
                Document(page_content=temp_summarize, metadata=temp_metadata)
            )
            temp_metadata = doc.metadata
            temp_summarize = summary_llm(doc.page_content)
            temp_summarize += " "

    # Set result as a document, not text
    concat_summarize.append(
        Document(page_content=temp_summarize, metadata=temp_metadata)
    )
    return concat_summarize

if __name__ == "__main__":
    docs = load_docs(root_directory="test_data/", is_split=False)    
    print(get_summarize_documents(docs=docs))
