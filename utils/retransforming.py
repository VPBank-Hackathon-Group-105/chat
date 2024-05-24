from .llm_api import get_llm

from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

def retransform(input_query):
    llm = get_llm(model = "anthropic.claude-3-sonnet-20240229-v1:0", temperature=0)

    def _parse(text):
        return text.strip('"').strip("**")

    template = PromptTemplate.from_template(
        """
        Retransform the query from HR. List 8-17 requirements focus in skills and experiences that the applicant need to have, remember to generate shortly but at most specific requirement as possible. 
        List 0-5 more requirement about education, certification, soft skills only if needed.
        If the query written in Vietnamese, please translate it to English.
        End generating after 10 requirements.
        Query: {input_query}. Retransformed query shortly:"""
    )

    rewriter = template | llm | StrOutputParser() | _parse


    response = rewriter.invoke(input_query)

    print(response)
    return response

if __name__ == "__main__":
    input_query = "Tìm kiếm ứng viên có kinh nghiệm làm việc với các framework xây dựng web backend bằng Java"
    print(retransform(input_query))