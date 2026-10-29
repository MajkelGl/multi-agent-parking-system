from src.utils.llm_config import load_llm
from src.bot.sql_chain import run_sql_query
from src.bot.retriever import run_chain as run_milvus_query

prompt_template = """
You are a well-trained classification model. Your task is to classify the user query into one of the following categories: SQL, Milvus.
Your job is to analyze the query and determine which category it belongs to based on its content.

If the query is about policies, rules, or regulations related to parking, it should be classified as RAG.
If the query is about searching for parking spots, it should be classified as SQL.

As an output you should return only the category name, either "SQL" or "RAG", without any additional text or explanations.
User query: {query}
"""

def classify_query(query: str) -> str:
    llm = load_llm()
    prompt = prompt_template.format(query=query)
    response = llm.invoke(prompt)
    category = response.content
    if category not in ["SQL", "RAG"]:
        raise ValueError(f"Invalid category returned by the model: {category}")
    return category

def route_query(query: str) -> str:
    category = classify_query(query)
    if category == "SQL":
        print(f"Routing query to SQL chain: {query}")
        return run_sql_query(query)
    elif category == "RAG":
        print(f"Routing query to RAG chain: {query}")
        return run_milvus_query(query)
    else:
        raise ValueError(f"Unknown category: {category}")

def run():
    while True:
        user_query = input("Enter your query: ")
        try:
            print(route_query(user_query))
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")

if __name__ == "__main__":
    run()
