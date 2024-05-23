import embedding_search
from cohere_aws import Client
co = Client(region_name="us-east-1")
co.connect_to_endpoint(endpoint_name="cohere-rerank-v3-endpoint")

vector_index = embedding_search.get_index(cv_directory = "test_data/1")
response_content = embedding_search.get_similarity_search_results(index=vector_index, question="Python, aws, etl, mlops", top_k=20)

response = co.rerank(documents=response_content, query='who can responsible for a position as a top intern data engineer in Ha Noi?', rank_fields=['content'], top_n=5)
print(f'Documents: {response}') 


