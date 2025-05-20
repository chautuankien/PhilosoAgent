from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import RemoveMessage

from src.philosoagent.config import settings
from src.philosoagent.chains import (
    get_philosopher_conversation_chain,
    get_philosopher_summarize_chain
)

from typing_extensions import Literal

class PhilosopherState(MessagesState):
    """State class for the LangGraph workflow. It keeps track of the information necessary to maintain a coherent
    conversation between the Philosopher and the user.

    Attributes:
        philosopher_context (str): The historical and philosophical context of the philosopher.
        philosopher_name (str): The name of the philosopher.
        philosopher_perspective (str): The perspective of the philosopher about AI.
        philosopher_style (str): The style of the philosopher.
        summary (str): A summary of the conversation. This is used to reduce the token usage of the model.
    """
    
    philosopher_name: str
    summary: str

def create_workflow_graph() -> StateGraph:
    graph_builder = StateGraph(PhilosopherState)

    # Add essential nodes to the graph
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

    # Define simple workflow
    graph_builder.add_edge(START, "conversation_node")
    graph_builder.add_conditional_edges(
        "conversation_node",
        should_summarize_conversation,
    )
    graph_builder.add_edge("conversation_node", END)

    return graph_builder

async def conversation_node(state: PhilosopherState, config: RunnableConfig):
    summary = state.get("summary", "")
    conversation_chain = get_philosopher_conversation_chain()

    response = await conversation_chain.ainvoke(
        {
            "messages": state["messages"],
            "summary": summary,
        },
        config,
    )

    return {"messages": response}

async def summarize_conversation_node(state: PhilosopherState, config: RunnableConfig):
    summary = state.get("summary", "")
    summarize_chain = get_philosopher_summarize_chain(summary)

    response = await summarize_chain.ainvoke(
        {
            "messages": state["messages"],
            "philosopher_name": state["philosopher_name"],
            "summary": summary,
        },
        config,
    )

    delete_message = [
        RemoveMessage(id=m.id or "")
        for m in state["messages"][:-settings.TOTAL_MESSAGES_AFTER_SUMMARY]
    ]

    return {"summary": response.content, "messages": delete_message}

def should_summarize_conversation(state: PhilosopherState) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"
    return END

graph = create_workflow_graph().compile()


