import os
import json
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd

from io import StringIO
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from utils.llm_api import get_llm
from utils.file_loader import load_docs
from utils.summarize_cv import get_summarize_documents

def get_entities(input_query):
    llm = get_llm(model = "anthropic.claude-3-sonnet-20240229-v1:0", temperature=0)

    def _parse(text):
        return text.strip('"').strip("**")

    template = PromptTemplate.from_template(
        """Given a CV document,
        Extract entity from CV including: 
        Name, year of birth (if any), skills, experiences and years of experience, education, award/qualifications. 
        Return string in a json format, if any entity can not be extracted, simply set value as null.
        Skills, experiences, educations, awards/qualifications are list of strings. If there is no entity to be extracted, set it as null.
        Result is just json format, do not print out anything else.
        
        For example: a json response like this:
        {{
            "name": "John Doe",
            "year_of_birth": null,
            "skills": ["Java", "Python", "SQL"],
            "experiences": [
                "Built and maintain a URL categorization system.",
                "Collaborate with the data science team on system requirements and architecture design",
                "Develop ETL pipelines processing 20~30 GB of data per hour",
                "Work with data science team to implement deep learning models for URL category prediction"
            ],
            "year_of_experience": 5,
            "educations": ["Bachelor of Science in Computer Science"],
            "awards": null,
            "qualifications": null
        }}
        
        Given this CV: {text}\n\n 
        Extract entity from CV to json format:
        """
    )

    summarizer = template | llm | StrOutputParser() | _parse

    response = summarizer.invoke({"text": input_query})

    return validate_and_return_json(response)

def validate_and_return_json(response_text):
    try:
        return json.loads(response_text)

    except Exception as err:
        return None

if __name__ == "__main__":
    docs = load_docs(root_directory="test_data/", is_split=False)
    summaries = get_summarize_documents(docs=docs)
    summary = summaries[0].page_content
    print(get_entities(summary))
