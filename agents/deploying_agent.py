from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

DEPLOYMENT_LOGIC_MATRIX = """
You are the Deploying Agent for a student/hackathon ML project.
Your task is to determine the best deployment strategy based on the current state and requirements.

DEPLOYMENT LOGIC MATRIX:
1. Demo happens on your own laptop (viva, hackathon judging in person):
   -> Recommendation: FastAPI, run locally, no container.
   -> Reason: Containerization is pure overhead when there's one machine. You will burn hours debugging Docker GPU passthrough for local LLMs instead of finishing agent logic.
2. Demo needs a public link (evaluators access remotely, resume/portfolio live):
   -> Recommendation: Cloud (Render, Railway) — but only the orchestration layer, not the local LLM.
   -> Reason: Free-tier clouds do not have VRAM to run 7B+ models. If you promise a "fully local free stack" demo, it will OOM remotely.
3. You need reproducibility for submission / multiple teammates:
   -> Recommendation: Docker Compose (not bare FastAPI).
   -> Reason: You have multiple services (LLM, Orchestrator) that need consistent versions. "Works on my machine" is a grading-day failure.
4. Agent needs to run unattended (background workers, Telegram bot pattern):
   -> Recommendation: FastAPI + background worker process, containerized.
5. Edge device:
   -> Recommendation: N/A. Don't do this for the current FYP scope.

Analyze the current state and recommend the deployment strategy. Keep it concise.
"""

def deploying_agent_node(state: AgentState, config: RunnableConfig):
    """
    Deploying Agent.
    Evaluates how to package and deploy the model based on the hackathon logic matrix.
    """
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "Deploy the model for local hackathon demo."

    print(f"[Deploying Agent] Evaluating deployment constraints...")
    
    # Construct LLM prompt
    sys_msg = SystemMessage(content=DEPLOYMENT_LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    
    # Invoke LLM
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "deployment_completed": True,
        "next_agent": "END"
    }
    
    return new_state
