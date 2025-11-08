import os
from dataclasses import asdict

import requests
from langchain_core.tools import tool

from src.models.agent.agent_source import AgentSource
from src.models.agent.custom_tool_response import CustomToolResponse
from src.models.app_state import app_state
from src.models.db import Project, ProjectSchema
from src.models.services.vector_store import VectorStore

@tool
def send_notification_tool(text: str) -> CustomToolResponse:
    """
    Use this tool to send a notification in bulgarian
    :param text: Message informing me about the specific question that couldn't be answered
    """
    try:
        # app_state.mailer.send_email(
        #     from_addr=os.getenv("CHATBOT_NOTIFICATION_SENDER_EMAIL"),
        #     to_addr=os.getenv("CHATBOT_NOTIFICATION_RECIPIENT_EMAIL"),
        #     subject="Чатботът не разполага с някаква информация",
        #     msg=text
        # )
        return CustomToolResponse(data={"status": "success"}, sources=None)
    except Exception as e:
        return CustomToolResponse(data={"status": "fail", "error": e}, sources=None)

@tool
def semantic_search(queries: list[str]) -> CustomToolResponse:
    """
    Search for relevant information in the knowledge base using semantic similarity.

    Use this tool FIRST when the user asks questions that might be answered by stored documents.
    Always try this tool before using send_notification_tool.

    The search queries SHOULD be only in ENGLISH.

    :param queries: List of search queries to find relevant information.
                   Use multiple related queries to get comprehensive results.
    """

    results = VectorStore.semantic_search(os.getenv("VECTOR_STORE_COLLECTION"), queries)

    return CustomToolResponse(
        data=[{
            "query": queries[index],
            "results": [asdict(doc) for doc in result]
        } for index, result in enumerate(results)],
        sources=[AgentSource(name=f"Vector Store", url=None)]
    )

@tool
async def get_all_projects() -> CustomToolResponse:
    """
    Get all projects that I have developed.

    Always use tool before providing information about some project
    :return: A list of projects in JSON format
    """

    return CustomToolResponse(
        data=[(await ProjectSchema.from_tortoise_orm(project)).model_dump() for project in await Project.all()],
        sources=[AgentSource(name="Database", url=None)]
    )

tools = [send_notification_tool, semantic_search, get_all_projects]