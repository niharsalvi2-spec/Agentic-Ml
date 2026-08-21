"""
EDA Agent Controller Module.
Follows the Agent -> Skill -> Service -> Tool architecture.
"""

import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.eda.statistics import EDAEngine

logger = logging.getLogger("agentic_ml.agents.eda")

EDA_SYSTEM_PROMPT = """You are the Senior EDA & Statistical Profiling Agent.
Analyze univariate distributions, skewness, kurtosis, outlier boundaries (1.5*IQR), and multi-collinearity (|r| >= 0.75).
Provide an actionable technical synthesis for downstream Feature Engineering and Selection agents.
"""

def eda_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes automated exploratory data analysis, extracts statistical distributions,
    detects multi-collinear pairs, and logs insights into the global state.
    """
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    dataset_path = state.get("dataset_path", "")
    
    # 1. Load Data
    df, target_col = DataLoader.load_or_synthesize(task_type, dataset_path)
    
    # 2. Execute Deterministic EDA Engine
    stats_data = EDAEngine.analyze(df)
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    skewness_map = {col: round(float(df[col].skew()), 4) for col in numeric_cols if len(df[col].dropna()) > 2}
    
    summary_report = {
        "dimensions": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "numeric_features": len(numeric_cols),
        "skewness": skewness_map,
        "summary_statistics": stats_data.get("summary_stats", {})
    }
    
    # 3. LLM Reasoning Call
    human_prompt = (
        f"Task: {state.get('current_task', 'ML Modeling')}\n"
        f"Target Column: {target_col}\n"
        f"Computed Statistical Data:\n"
        f"```json\n{json.dumps(summary_report, indent=2)}\n```\n\n"
        f"Synthesize the exploratory findings and highlight any feature distribution anomalies."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=EDA_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt)
        ])
    except Exception as e:
        logger.error("LLM reasoning fallback in EDA node: %s", e)
        response = AIMessage(
            content=f"[EDA Agent Technical Summary]\n"
                    f"- Analyzed {df.shape[0]} rows across {df.shape[1]} attributes.\n"
                    f"- Skewed attributes (|skew| > 1.0): {[k for k, v in skewness_map.items() if abs(v) > 1.0]}\n"
                    f"- Statistical profiling completed. Proceeding to Feature Engineering."
        )
    
    return {
        "messages": [response],
        "data_summary": summary_report,
        "eda_completed": True,
        "next_agent": "feature_engineering"
    }
