from llm_config import llm, embeddings

print("Testing LLM connection...")

try:
    response = llm.invoke("Hello, how are you?")
    print("LLM connection successful. Response:")
    print(response.content[:])
except Exception as e:
    print("LLM connection failed. Error:")
    print(e)

try:
    print("Testing Embeddings connection...")
    test_text = "This is a test."
    embedding_vector = embeddings.embed_query(test_text)
    print("Embeddings connection successful. Embedding vector:")
    print(embedding_vector)
except Exception as e:
    print("Embeddings connection failed. Error:")
    print(e)