from llm_api import get_llm
from file_loader import load_docs
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from io import StringIO
import pandas as pd

def get_summary(input_query):
    llm = get_llm(model = "anthropic.claude-3-sonnet-20240229-v1:0", temperature=0)

    def _parse(text):
        return text.strip('"').strip("**")

    template = PromptTemplate.from_template(
        """Given this CV:{text}\n\n 
        Extract entity from CV including: 
        Name, year of birth (if available), skills, experiences and years of experience, education, award and qualifications. 
        Return in CSV format only return CSV do not give anymore explain, 
        use ";" to seperate each column so dont use any ";" in a field value:"""
    )

    summarizer = template | llm | StrOutputParser() | _parse

    response = summarizer.invoke(input_query)

    print(response)
    return validate_and_return_csv(response)    

def validate_and_return_csv(response_text):
    #returns has_error, response_content, err 
    try:
        csv_io = StringIO(response_text)
        csv_file = pd.read_csv(csv_io, sep=";")
        csv_file.to_csv("test.csv")
        return False, csv_file, None #attempt to load response CSV into a dataframe

    except Exception as err:
        return True, response_text, err
    

if __name__ == "__main__":
    docs = load_docs(root_directory="test_data/try")    
    doc = docs[1].page_content
    print(get_summary(input_query=doc))
    