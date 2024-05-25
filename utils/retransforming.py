import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.llm_api import get_llm

from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

def retransform(input_query, memory):
    llm = get_llm(model = "anthropic.claude-3-haiku-20240307-v1:0", temperature=0)

    def _parse(text):
        return text.strip('"').strip("**")

    template = PromptTemplate.from_template(
        """
        In most case, HR will ask you to find somebody with specific skills and experiences.
        Retransform the query from HR. List 8-17 requirements focus in skills and experiences that the applicant need to have, remember to generate shortly but at most specific requirement as possible. 
        List 0-5 more requirement about education, certification, soft skills only if needed.
        In some case, HR will ask you to find somebody specific candidate given their name, if so just retransform the query to the name of the candidate and some more information if needed.
        If the query written in Vietnamese, please translate it to English.
        \nQuery: {input_query}. Retransformed query shortly:"""
    )
    
    rewriter =  template | llm | StrOutputParser() | _parse

    response = rewriter.invoke(input_query)

    return response

if __name__ == "__main__":
    input_query = "Tìm kiếm ứng viên có kinh nghiệm làm việc với các framework xây dựng web backend bằng Java"
    print(retransform(input_query))