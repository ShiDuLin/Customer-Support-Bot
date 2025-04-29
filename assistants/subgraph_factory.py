from typing import List, Callable
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from tools.utilities_tools import create_tool_node_with_fallback
from langchain_core.messages import ToolMessage
from assistants.common import State
from assistants.base import Assistant, CompleteOrEscalate

def create_entry_node(assistant_name_des: str, new_dialog_state: str) -> Callable:
    """创建入口节点函数"""
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name_des}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name_des},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def create_specialized_subgraph(
    builder: StateGraph,
    assistant_name: str,
    assistant_name_des: str,
    prompt: ChatPromptTemplate,
    safe_tools: List[Runnable],
    sensitive_tools: List[Runnable],
    llm: Runnable,
    route_function: Callable,
) -> None:
    """创建专门的子图（如航班预订、酒店预订等）"""
    # 创建runnable
    runnable = prompt | llm.bind_tools(
        safe_tools + sensitive_tools + [CompleteOrEscalate]
    )

    # 添加入口节点
    builder.add_node(
        f"enter_{assistant_name}", create_entry_node(assistant_name_des, assistant_name)
    )

    # 添加助手节点
    builder.add_node(assistant_name, Assistant(runnable))

    # 添加工具节点
    builder.add_node(
        f"{assistant_name}_safe_tools", create_tool_node_with_fallback(safe_tools)
    )
    builder.add_node(
        f"{assistant_name}_sensitive_tools",
        create_tool_node_with_fallback(sensitive_tools),
    )
    
    # 添加边
    builder.add_edge(f"enter_{assistant_name}", assistant_name)

    builder.add_edge(f"{assistant_name}_safe_tools", assistant_name)
    builder.add_edge(f"{assistant_name}_sensitive_tools", assistant_name)
    builder.add_conditional_edges(
        assistant_name,
        route_function,
        [
            f"{assistant_name}_safe_tools",
            f"{assistant_name}_sensitive_tools",
            "leave_skill",
            END,
        ],
    )
    builder.add_edge("leave_skill", "primary_assistant")
    
