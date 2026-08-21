def get_groq_provider(api_key: str):
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
