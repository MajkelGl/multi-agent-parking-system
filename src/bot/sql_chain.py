import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import QuerySQLDatabaseTool

from src.utils.llm_config import load_llm

load_dotenv()

def load_models_and_database():
    llm = load_llm()
    db = SQLDatabase.from_uri(os.getenv("POSTGRES_URL"), include_tables=["parking_spots"])
    return llm, db

def clean_sql(sql_query: str) -> str:
    """Fallback: removes markdown blocks if the model generates them."""
    return sql_query.replace("```sql", "").replace("```", "").strip()

def run_sql_query(query: str):
    llm, db = load_models_and_database()
    
    template = """You are a PostgreSQL expert. Your task is to write a correct SQL query based on the database schema.
    Do not use markdown formatting. Return ONLY the raw SQL query text, without any introductory words or explanations.
    
    Table schema:
    {schema}
    
    User question: {question}
    
    SQL query:"""
    
    prompt = PromptTemplate.from_template(template)
    
    execute_sql_tool = QuerySQLDatabaseTool(db=db)
    
    generate_query_chain = (
        {"schema": lambda _: db.get_table_info(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | clean_sql
    )
    
    sql_pipeline = generate_query_chain | execute_sql_tool
    
    raw_result = sql_pipeline.invoke(query)
    
    formatting_template = """You are a helpful parking assistant. 
    The user asked: '{question}'. 
    The database returned this raw data: '{result}'. 
    Formulate a polite, natural language response answering the question based ONLY on the database result.
    If the result is empty or indicates an error, say you don't have this information right now.
    
    Answer:"""
    
    formatting_prompt = PromptTemplate.from_template(formatting_template)
    
    formatting_chain = formatting_prompt | llm | StrOutputParser()
    
    final_human_answer = formatting_chain.invoke({
        "question": query,
        "result": raw_result
    })
    
    return final_human_answer

if __name__ == "__main__":
    print("Running SQL query...")
    query = "How many free places are available in sector A?"
    result = run_sql_query(query)
    print("SQL query result:")
    print(result)