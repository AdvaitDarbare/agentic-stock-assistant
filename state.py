from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The shape of our LangGraph state:
      - input: the user's latest question
      - chat_history: sequence of HumanMessage/AIMessage
      - output: last agent response text
      - current_date: last parsed date (YYYY-MM-DD)
    """
    input: str
    chat_history: List[BaseMessage]
    output: str
    current_date: str
    next_node: None