from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from agents import (
    AgentState, 
    data_engineering_agent_node,
    data_collector_node, 
    data_preprocessor_node,
    eda_agent_node,
    feature_selection_agent_node,
    model_building_agent_node,
    testing_agent_node,
    validating_agent_node,
    deploying_agent_node
)

def master_agent_router(state: AgentState):
    """
    Master Agent Routing Logic.
    Decides which agent to call next based on the state flags.
    """
    next_agent = state.get("next_agent")
    
    if next_agent == "deploying_agent":
        return "deploying_agent"
    if next_agent == "validating_agent":
        return "validating_agent"
    if next_agent == "testing_agent":
        return "testing_agent"
    if next_agent == "model_building":
        return "model_building"
    if next_agent == "feature_selection":
        return "feature_selection"
    if next_agent == "eda_agent":
        return "eda_agent"
    if next_agent == "data_preprocessor":
        return "data_preprocessor"
    if next_agent == "data_collector":
        return "data_collector"
    
    # Sequence fallbacks if next_agent wasn't explicitly set
    if not state.get("data_engineering_completed"):
        return "data_engineering_agent"
    if not state.get("data_collected"):
        return "data_collector"
    if not state.get("data_preprocessed"):
        return "data_preprocessor"
    if not state.get("eda_completed"):
        return "eda_agent"
    if not state.get("feature_selection_completed"):
        return "feature_selection"
    if not state.get("model_built"):
        return "model_building"
    if not state.get("model_tested"):
        return "testing_agent"
    if not state.get("model_validated"):
        return "validating_agent"
    if not state.get("deployment_completed"):
        return "deploying_agent"
        
    return END

def build_graph():
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)

    # Add the nodes (Sub-Agents)
    workflow.add_node("data_engineering_agent", data_engineering_agent_node)
    workflow.add_node("data_collector", data_collector_node)
    workflow.add_node("data_preprocessor", data_preprocessor_node)
    workflow.add_node("eda_agent", eda_agent_node)
    workflow.add_node("feature_selection", feature_selection_agent_node)
    workflow.add_node("model_building", model_building_agent_node)
    workflow.add_node("testing_agent", testing_agent_node)
    workflow.add_node("validating_agent", validating_agent_node)
    workflow.add_node("deploying_agent", deploying_agent_node)

    # Add conditional edges
    agents = [
        "data_engineering_agent", "data_collector", "data_preprocessor", "eda_agent", 
        "feature_selection", "model_building", "testing_agent", "validating_agent", "deploying_agent"
    ]
    
    workflow.add_conditional_edges(
        START,
        master_agent_router,
        {a: a for a in agents} | {END: END}
    )
    
    for agent in agents:
        workflow.add_conditional_edges(
            agent,
            master_agent_router,
            {a: a for a in agents} | {END: END}
        )

    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    print("Initializing Master ML Engineer Agentic AI...")
    app = build_graph()
    
    # Example starting state
    initial_state = {
        "messages": [HumanMessage(content="I need to build an ML model to predict customer churn. I don't have any data yet.")],
        "current_task": "Predict Customer Churn",
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
    
    print("\n--- Starting Pipeline execution ---")
    # Run the graph
    for output in app.stream(initial_state):
        # Stream yields dictionaries where keys are node names and values are their output state
        for node_name, state_update in output.items():
            print(f"\nOutput from {node_name}:")
            if "messages" in state_update and state_update["messages"]:
                print(state_update["messages"][-1].content)
            print("-" * 50)
            
    print("\nPipeline Execution Completed.")
