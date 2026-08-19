import os
import sys
import json
import httpx
import asyncio

# Add core_model to sys path so we can import our native classes
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "core_model"))

from prompt_engine import PromptEngine

class GenixService:
    """
    The main backend service. 
    Currently hooked up to local Llama 3 via Ollama for high-quality baseline reasoning.
    """
    def __init__(self):
        self.prompt_engine = PromptEngine()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "qwen2.5:3b"

    async def stream_inference(self, user_prompt: str):
        """
        Takes a prompt, formats it via our custom ReAct prompt engine,
        sends it to the local Llama 3 API, and streams the decoded tokens back 
        asynchronously via SSE format.
        """
        # Inject the user's prompt into our advanced ReAct structure
        formatted_prompt = self.prompt_engine.react_loop(
            task=user_prompt, 
            available_tools=["DataCollector", "Preprocessor", "ModelBuilder"]
        )
        
        # We simulate thinking delay just so it looks smooth on the UI while it calculates
        thinking_payload = {"agent": "Genix Engine (Qwen 2.5)", "message": "\n\n[Thinking...]\n"}
        yield f"data: {json.dumps(thinking_payload)}\n\n"
        await asyncio.sleep(0.5)

        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "stream": True,
            "options": {
                "temperature": 0.5,
                "top_k": 40
            }
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream("POST", self.ollama_url, json=payload) as response:
                    if response.status_code != 200:
                        error_msg = f"\n\n[Error: Ollama returned status {response.status_code}. Is it running?]\n"
                        err_payload = {"agent": "Genix Engine (Qwen 2.5)", "message": error_msg}
                        yield f"data: {json.dumps(err_payload)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                new_token_str = data["response"]
                                # Send to UI via Server-Sent Events format
                                token_payload = {"agent": "Genix Engine (Qwen 2.5)", "message": new_token_str}
                                yield f"data: {json.dumps(token_payload)}\n\n"
                        except json.JSONDecodeError:
                            continue
                            
            except httpx.RequestError as e:
                error_msg = f"\n\n[Error connecting to Ollama: {str(e)}. Make sure 'ollama run llama3' is running!]\n"
                err_payload = {"agent": "Genix Engine (Qwen 2.5)", "message": error_msg}
                yield f"data: {json.dumps(err_payload)}\n\n"

        yield "data: [DONE]\n\n"

# Singleton instance
genix_app = GenixService()
