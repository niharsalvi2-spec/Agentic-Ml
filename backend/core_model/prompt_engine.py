class PromptEngine:
    """
    A unified framework for structuring and managing prompt injections for the Genix model.
    It natively structures standard AI techniques (Zero-shot, Few-shot, CoT, ToT, ReAct)
    and uses XML tagging for robust parsing.
    """
    
    @staticmethod
    def zero_shot(system_prompt: str, task: str) -> str:
        """Standard task execution without examples."""
        return f"<system>{system_prompt}</system>\n<task>{task}</task>\n<response>"

    @staticmethod
    def few_shot(system_prompt: str, examples: list[tuple[str, str]], task: str) -> str:
        """Injects demonstration examples to guide in-context learning."""
        prompt = f"<system>{system_prompt}</system>\n<examples>\n"
        for i, (q, a) in enumerate(examples):
            prompt += f"  <example_{i}>\n    <input>{q}</input>\n    <output>{a}</output>\n  </example_{i}>\n"
        prompt += f"</examples>\n<task>{task}</task>\n<response>"
        return prompt

    @staticmethod
    def chain_of_thought(task: str) -> str:
        """Forces the model to generate a reasoning trace before answering."""
        return f"<task>{task}</task>\n<instructions>Think step-by-step and place your reasoning inside <think> tags before providing the final <answer>.</instructions>\n<response>\n<think>"

    @staticmethod
    def react_loop(task: str, available_tools: list[str]) -> str:
        """
        Reason + Act loop. Formats the prompt to force the agent to use
        Observation, Thought, and Action cycles.
        """
        return (
            f"<system>You are Genix, an elite, world-class Agentic ML Engineer powered by Qwen 2.5.\n"
            f"You are competing with the best AI models in the world (like Claude 3.5). Your output must be EXTRAORDINARILY detailed, comprehensive, and production-ready. Do NOT output small snippets. Output massive, robust code architectures.\n\n"
            f"Use the following format:\n"
            f"Thought: [Deep, step-by-step reasoning evaluating data engineering, modeling, and MLOps strategies]\n"
            f"Final Answer: [You MUST provide an extremely long, fully executable Python codebase in a single ```python block.\n"
            f"Your code MUST include:\n"
            f"1. A complete Data Engineering pipeline (EDA, preprocessing, feature engineering)\n"
            f"2. Multiple advanced ML models (e.g. XGBoost, Random Forest, PyTorch NNs for both classification and regression)\n"
            f"3. Hyperparameter tuning and Cross-Validation\n"
            f"4. A beautiful visual dashboard implementation using Plotly, Matplotlib, or Streamlit to visualize the results (simulating a rich UI dashboard).\n"
            f"DO NOT cut corners. Write hundreds of lines of code if necessary to build a complete, state-of-the-art ML platform from scratch.]</system>\n\n"
            f"<task>{task}</task>\nThought:"
        )

    @staticmethod
    def tree_of_thoughts(task: str, num_branches: int = 3) -> str:
        """
        Prompts the model to generate multiple distinct reasoning paths 
        to evaluate complex problems before deciding on the best one.
        """
        return (
            f"<task>{task}</task>\n"
            f"<instructions>\n"
            f"1. Generate {num_branches} distinct, independent reasoning paths to solve this task.\n"
            f"2. Evaluate the pros and cons of each path.\n"
            f"3. Select the most robust path and provide the final answer based on it.\n"
            f"</instructions>\n"
            f"<response>\n<path_1>"
        )

    @staticmethod
    def extract_xml_tag(text: str, tag: str) -> str:
        """Utility to safely extract structured output injected via the engine."""
        import re
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""
