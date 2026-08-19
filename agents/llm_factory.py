import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

class DummyLLM:
    """A fallback LLM that returns a generic response when no API keys are provided."""
    def invoke(self, messages):
        # Extract system prompt or last message to guess the context
        context = messages[0].content if messages else ""
        return AIMessage(content=f"[Simulated LLM Response] Evaluated state based on logic matrix. Please provide a GEMINI_API_KEY or GROQ_API_KEY in .env for true reasoning.")

def get_llm():
    """
    Returns an initialized LLM based on available API keys.
    Prioritizes Gemini, then Groq. Falls back to DummyLLM if none found.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_key, temperature=0.2)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Using gemini-1.5-flash caused a 404 in the current SDK version
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_key, temperature=0.2)
        
    return DummyLLM()
