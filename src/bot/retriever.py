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


def load_models_and_milvus(k: int = 3):
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

    retriever = milvus_db.as_retriever(search_kwargs={"k": k})

    return llm, retriever

def run_chain(query: str, k: int = 3, return_context: bool = False) -> str:

    llm, retriever = load_models_and_milvus()

    search_kwargs = {"k": k}

    prompt_template = PromptTemplate.from_template(
        """
        You are an AI assistant that helps users find information about parking rules, pricing, and policies. 
        When a user asks a question, you should provide a clear and concise answer based ONLY on the information available 
        in the provided context. If the information is not available, inform the user that you cannot provide an answer.
        
        Knowledge from the database:
        {context}
        
        User question: {question}
        
        Your answer:
        """
    )

    if return_context:
        docs = retriever.invoke(query)
        context_string = format_docs(docs)
        
        eval_chain = prompt_template | llm | StrOutputParser()
        answer = eval_chain.invoke({
            "context": context_string,
            "question": query
        })
        
        return answer, context_string
        
    else:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()} 
            | prompt_template 
            | llm 
            | StrOutputParser()
        )
        return rag_chain.invoke(query)

def return_output(query: str, k: int = 3) -> str:
    return run_chain(query, k=k)

