from mcp.server.fastmcp import FastMCP
from main import build_graph
from langchain_core.messages import HumanMessage

# Initialize FastMCP Server
mcp = FastMCP("Master ML Engineer Agentic AI")

@mcp.tool()
def run_ml_pipeline(dataset_description: str, task: str) -> str:
    """
    Run the Master ML Engineer Agentic Pipeline on a given dataset and task.
    
    Args:
        dataset_description: A description of the data available (or lack thereof).
        task: The ML problem to solve (e.g., 'Predict Customer Churn').
    """
    app = build_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=dataset_description)],
        "current_task": task,
        "dataset_path": "",
        "dataset_info": {},
        "data_engineering_completed": False,
        "data_collected": False,
        "data_preprocessed": False,
        "eda_completed": False,
        "feature_selection_completed": False,
        "model_built": False,
        "model_tested": False,
        "model_validated": False,
        "deployment_completed": False,
        "next_agent": None
    }
    
    output_log = [f"Starting ML Pipeline for task: {task}"]
    
    # Run the graph
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            if "messages" in state_update and state_update["messages"]:
                msg = state_update["messages"][-1].content
                output_log.append(f"[{node_name.upper()}]:\n{msg}")
                
    output_log.append("Pipeline Execution Completed Successfully.")
    
    # Return the full aggregated log
    return "\n\n------------------------\n\n".join(output_log)

if __name__ == "__main__":
    print("Starting MCP Server: Master ML Engineer Agentic AI on stdio...")
    mcp.run(transport="stdio")
