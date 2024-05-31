import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm_api import get_llm
from utils.file_loader import load_docs



def summary_llm(input_query, last_summary=False):
    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0", temperature=0)

    template_1 = PromptTemplate.from_template(
        """Given this as a part of CV:{text}\n\n 
        Write 1-4 sentences to summarize the CV, and it better including (if any) the following information: 
        Name, year of birth, skills, experiences and years of experience, education, award and qualifications. 
        If any of those information is missing then do not include it, do not say anything about it in the summary.
        Importance: Be as short as possible but at most specific, if the CVs use Vietnamese then summarize in English. Do not make a new line
        Summarization:"""
    )

    template_2 = PromptTemplate.from_template(
        """Given this as many summarize parts of a CV:{text}\n\n 
        Write 3-9 sentences to summarize the CV, and it better including (if any) the following information: 
        Name, year of birth, skills, experiences and years of experience, education, award and qualifications. 
        If any of those information is missing then do not include it, do not say anything about it in the summary.
        Importances: only give summarization, do not give any other explain. Do not begin a new line. Do not repeat the word "Summarization".
        Summarization:"""
    )
    if not last_summary:
        summarizer = template_1 | llm | StrOutputParser() 
    if last_summary:
        summarizer = template_2 | llm | StrOutputParser()
    
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

    final_concat_summarize = []
    for summarize in concat_summarize:
        if len(summarize.page_content) < 1500:
            final_concat_summarize.append(
                Document(page_content=summarize.page_content, metadata=summarize.metadata))
        else:
            final_summary = summary_llm(summarize.page_content, last_summary=True)
            final_concat_summarize.append(
                Document(page_content= final_summary, metadata=summarize.metadata)
            )

    return final_concat_summarize

if __name__ == "__main__":
    docs = load_docs(root_directory="test_data/", is_split=False)    
    print(get_summarize_documents(docs=docs))
