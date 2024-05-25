from .llm_api import get_llm
from .file_loader import load_docs
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from io import StringIO
import pandas as pd
from .summarize_cv import get_summary

def get_entities(input_query):
    llm = get_llm(model = "anthropic.claude-3-sonnet-20240229-v1:0", temperature=0)

    def _parse(text):
        return text.strip('"').strip("**")

    template = PromptTemplate.from_template(
        """Given this CV:{text}\n\n 
        Extract entity from CV including: 
        Name, year of birth (if any), skills, experiences and years of experience, education, award/qualifications. 
        Return in CSV format only return CSV do not give anymore explain,  
        use ";" to seperate each column so dont use any ";" in a field value:"""
    )

    summarizer = template | llm | StrOutputParser() | _parse

    response = summarizer.invoke(input_query)

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
    docs = load_docs(root_directory="test_data/try", is_split=False)    
    summaries = get_summary(docs=docs)
    summary = summaries[0]['cv']
    print(get_entities(summary))
    