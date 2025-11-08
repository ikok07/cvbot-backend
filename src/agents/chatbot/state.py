
from langgraph.graph import MessagesState

from src.models.agent.agent_source import AgentSource

class State(MessagesState):
    sources: list[AgentSource]
    question_suggestions: list[str]
    pass