import os
from dotenv import load_dotenv

from langchain_milvus import Milvus

from src.utils.llm_config import load_llm, load_embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MILVUS_FILE_PATH = os.getenv("MILVUS_FILE_PATH", "./milvus_parking.db")
os.environ["GRPC_ARG_KEEPALIVE_TIME_MS"] = "60000"
os.environ["GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "0"
os.environ["GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def load_models_and_milvus():
    llm = load_llm()
    embeddings = load_embeddings()

    milvus_db = Milvus.from_documents(
        documents=[],
        embedding=embeddings,
        connection_args={
            "uri": MILVUS_FILE_PATH,
            },
        collection_name="parking_rules"
    )

    retriever = milvus_db.as_retriever(search_kwargs={"k": 3})

    return llm, retriever

def run_chain(query: str):

    llm, retriever = load_models_and_milvus()

    prompt_template = PromptTemplate.from_template(
        """
    You are an AI assistant that helps users find information about parking spots in a database. 
    The database contains information about parking spots, including their status (available or occupied), 
    location, and other relevant details.
    When a user asks a question, you should provide a clear and concise answer based on the information available 
    in the database. If the information is not available, you should inform the user that you cannot provide an answer.
    Don't tell the user about SQL queries or the database structure. Just provide the information in a user-friendly manner.
    
    Knowledge from the database:
    {context}
    User question: {question}
    Your answer:
    """
    )

    rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()} 
                | prompt_template 
                | llm 
                | StrOutputParser()
            )
    return rag_chain.invoke(query)

def return_output(query: str) -> str:
    return run_chain(query)

