import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
LLM_DEPLOYMENT = os.getenv("LLM_MODEL_NAME", "gpt-5-mini")
EMBEDDINGS_DEPLOYMENT = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-small")

API_VERSION = "2024-02-15-preview"

if AZURE_ENDPOINT:
    AZURE_ENDPOINT = AZURE_ENDPOINT.replace("/openai/v1", "").replace("/openai/v1/", "")

if not AZURE_API_KEY:
    raise ValueError("Brak AZURE_OPENAI_API_KEY w pliku .env!")

def load_llm():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=API_VERSION,
        azure_deployment=LLM_DEPLOYMENT,
        max_tokens=2048,
    )

def load_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=API_VERSION,
        azure_deployment=EMBEDDINGS_DEPLOYMENT
    )