import os
import re
from datetime import datetime
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
    db = SQLDatabase.from_uri(os.getenv("POSTGRES_URL"), include_tables=["parking_spots", "reservations"])
    return llm, db

def clean_sql(sql_query: str) -> str:
    """Fallback: removes markdown blocks if the model generates them."""
    return sql_query.replace("```sql", "").replace("```", "").strip()

def run_sql_query(query: str) -> str:
    llm, db = load_models_and_database()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    template = """You are a PostgreSQL expert. Your task is to write a correct SQL query based on the database schema.
    Do not use markdown formatting. Return ONLY the raw SQL query text, without any introductory words or explanations.
    
    Table schema:
    {schema}
    
    CRITICAL RULES FOR AVAILABILITY:
    1. The current system time is: {current_time}
    2. To find available spots, you MUST check the `reservations` table for time overlaps.
    3. An overlap exists if: (reservations.start_time < 'Requested End Time') AND (reservations.end_time > 'Requested Start Time').
    4. If a spot has NO overlapping records for the requested period, it is available.
    5. If the user does not specify a time, assume they mean "right now" ({current_time}).
    
    User question: {question}
    
    SQL query:"""
    
    prompt = PromptTemplate.from_template(template)
    execute_sql_tool = QuerySQLDatabaseTool(db=db)
    
    generate_query_chain = (
        {
            "schema": lambda _: db.get_table_info(), 
            "question": RunnablePassthrough(),
            "current_time": lambda _: current_time
        }
        | prompt
        | llm
        | StrOutputParser()
        | clean_sql
    )
    
    raw_sql = generate_query_chain.invoke(query)
    
    def execute_safe_query(sql: str) -> str:
        query_upper = sql.upper().strip()
        forbidden_keywords = [r'\bDROP\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bINSERT\b', r'\bALTER\b', r'\bTRUNCATE\b', r'\bCREATE\b']
        
        for keyword in forbidden_keywords:
            if re.search(keyword, query_upper):
                return "ERROR: Security violation. Prohibited SQL operation detected. Only read-only queries are allowed."
                
        if not query_upper.startswith("SELECT") and not query_upper.startswith("WITH"):
            return "ERROR: Security violation. Query must start with SELECT."
            
        try:
            return execute_sql_tool.invoke(sql)
        except Exception as e:
            return f"Database error: {str(e)}"
            
    return execute_safe_query(raw_sql)

if __name__ == "__main__":
    print("Running SQL query...")
    query = "How many free places are available in sector A right now?"
    result = run_sql_query(query)
    print("SQL query result (raw data for agent):")
    print(result)