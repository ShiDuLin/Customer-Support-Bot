import os
import uuid
from typing import Literal
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from assistants.common import State
from assistants.subgraph_factory import create_specialized_subgraph
from assistants.primary import (
    primary_assistant_tools,
    route_primary_assistant,
    create_primary_assistant,
)
from assistants.flight import (
    flight_booking_prompt,
    update_flight_safe_tools,
    update_flight_sensitive_tools,
    route_update_flight,
)
from assistants.hotel import (
    book_hotel_prompt,
    book_hotel_safe_tools,
    book_hotel_sensitive_tools,
    route_book_hotel,
)
from assistants.car_rental import (
    book_car_rental_prompt,
    book_car_rental_safe_tools,
    book_car_rental_sensitive_tools,
    route_book_car_rental,
)
from assistants.excursion import (
    book_excursion_prompt,
    book_excursion_safe_tools,
    book_excursion_sensitive_tools,
    route_book_excursion,
)
from tools.flight_tools import fetch_user_flight_information
from tools.utilities_tools import create_tool_node_with_fallback


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


def get_user_id():
    """获取测试用户ID，实际应用中可以通过登录系统获取"""
    # 这里使用一个示例ID
    return "0000 000001"


def pop_dialog_state(state: State) -> dict:
    """弹出对话状态栈并返回主助手"""
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


def create_agent(passenger_id: str) -> StateGraph:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=os.environ.get("MODEL_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=1,
        streaming=True,
    )

    builder = StateGraph(State)

    # 添加基础节点
    builder.add_node("fetch_user_info", user_info)

    # Primary assistant
    builder.add_node("primary_assistant", create_primary_assistant(llm))
    builder.add_node(
        "primary_assistant_tools",
        create_tool_node_with_fallback(primary_assistant_tools),
    )
    # 添加退出节点
    builder.add_node("leave_skill", pop_dialog_state)

    # Flight booking assistant
    create_specialized_subgraph(
        builder=builder,
        assistant_name="update_flight",
        assistant_name_des="Flight Updates & Booking Assistant",
        prompt=flight_booking_prompt,
        safe_tools=update_flight_safe_tools,
        sensitive_tools=update_flight_sensitive_tools,
        llm=llm,
        route_function=route_update_flight,
    )

    # Car rental assistant
    create_specialized_subgraph(
        builder=builder,
        assistant_name="book_car_rental",
        assistant_name_des="Car Rental Assistant",
        prompt=book_car_rental_prompt,
        safe_tools=book_car_rental_safe_tools,
        sensitive_tools=book_car_rental_sensitive_tools,
        llm=llm,
        route_function=route_book_car_rental,
    )

    # Hotel booking assistant
    create_specialized_subgraph(
        builder=builder,
        assistant_name="book_hotel",
        assistant_name_des="Hotel Booking Assistant",
        prompt=book_hotel_prompt,
        safe_tools=book_hotel_safe_tools,
        sensitive_tools=book_hotel_sensitive_tools,
        llm=llm,
        route_function=route_book_hotel,
    )

    # Excursion assistant
    create_specialized_subgraph(
        builder=builder,
        assistant_name="book_excursion",
        assistant_name_des="Trip Recommendation Assistant",
        prompt=book_excursion_prompt,
        safe_tools=book_excursion_safe_tools,
        sensitive_tools=book_excursion_sensitive_tools,
        llm=llm,
        route_function=route_book_excursion,
    )

    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges("fetch_user_info", route_to_workflow)
    builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant,
        [
            "enter_update_flight",
            "enter_book_car_rental",
            "enter_book_hotel",
            "enter_book_excursion",
            "primary_assistant_tools",
            END,
        ],
    )
    builder.add_edge("primary_assistant_tools", "primary_assistant")

    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=[
            "update_flight_sensitive_tools",
            "book_car_rental_sensitive_tools",
            "book_hotel_sensitive_tools",
            "book_excursion_sensitive_tools",
        ],
    )
    graph = graph.with_config(
        configurable={
            "passenger_id": passenger_id,
            "thread_id": str(uuid.uuid4()),
        }
    )
    return graph


def _print_event(event, printed):
    """打印事件，避免重复打印"""
    for key, value in event.items():
        if key not in printed:
            if key == "messages" and value:
                message = value[-1]
                if hasattr(message, "content") and message.content:
                    print(f"{key}: {message.content}")
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"Tool Calls: {message.tool_calls}")
            elif key == "assistant" and "messages" in value:
                message = value["messages"]
                if hasattr(message, "content") and message.content:
                    print(f"{key}: {message.content}")
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"Tool Calls: {message.tool_calls}")
            printed.add(key)


def process_message(
    agent, message: str, chat_history: list[BaseMessage] = None
) -> dict[str, any]:
    """处理用户消息并返回机器人回复"""
    if chat_history is None:
        chat_history = []
    messages = []
    for msg in chat_history:
        if isinstance(msg, (HumanMessage, AIMessage)):
            messages.append(msg)
    messages.append(HumanMessage(content=message))

    print("\n=== 开始处理新消息 ===")
    print(f"用户输入: {message}")

    # 用于跟踪已打印的事件
    _printed = set()
    response_chunks = []

    print("\n=== 开始流式处理 ===")
    events = list(agent.stream({"messages": messages}))
    print(f"获取到 {len(events)} 个事件")

    config = agent.config
    snapshot = agent.get_state(config)
    while snapshot.next:
        # 我们有一个中断！智能体正在尝试使用工具，用户可以批准或拒绝
        print("\n=== 检测到工具调用中断点 ===")
        try:
            user_input = input(
                "您是否批准上述操作？输入'y'继续；否则，请解释您要求的更改。\n\n"
            )
        except:
            user_input = "y"

        print(f"\n用户决定: {'批准' if user_input.strip() == 'y' else '拒绝'}")

        if user_input.strip() == "y":
            print("继续执行工具调用...")
            result = list(agent.stream(None))
        else:
            print(f"拒绝原因: {user_input}")
            print("重新规划执行...")
            result = list(
                agent.stream(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=events["messages"][-1].tool_calls[0]["id"],
                                content=f"API调用被用户拒绝。原因: '{user_input}'。请继续协助，考虑用户的输入。",
                            )
                        ]
                    },
                )
            )

        # 更新事件列表
        if result:
            print("\n=== 工具调用结果 ===")
            events.extend(result)

        snapshot = agent.get_state(config)

    print("\n=== 处理所有事件 ===")
    for event in events:
        if isinstance(event, dict) and "assistant" in event:
            assistant_message = event["assistant"].get("messages")
            if isinstance(assistant_message, AIMessage) and assistant_message.content:
                chunk = assistant_message.content
                response_chunks.append(chunk)
                yield chunk

        # 打印事件
        _print_event(event, _printed)

    print("\n=== 完成消息处理 ===")
    # 合并完整响应用于历史记录
    full_response = "".join(response_chunks)

    # 更新对话历史
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=full_response))

    return None


def chat_loop():
    """简单的命令行聊天界面"""
    passenger_id = get_user_id()
    print(f"当前用户ID: {passenger_id}")

    # 创建智能体
    agent = create_agent(passenger_id)

    chat_history = []
    print("客户支持机器人已启动（LangGraph版本）。输入'退出'结束对话。")

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("感谢使用客户支持系统，再见！")
            break

        # 处理用户消息并实时显示流式响应
        print("\n机器人: ", end="", flush=True)
        response_chunks = []

        # 收集流式响应
        for chunk in process_message(agent, user_input, chat_history):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)
        print()  # 换行


def main():
    """主程序入口点"""
    print("航空公司客户支持机器人")
    print("-" * 40)

    # 启动聊天循环
    chat_loop()


if __name__ == "__main__":
    # main()

    agent = create_agent(get_user_id())
    # 生成图表并保存为文件
    graph_png = agent.get_graph(xray=True).draw_mermaid_png()

    # 保存到文件
    with open("agent_graph.png", "wb") as f:
        f.write(graph_png)
    print("图表已保存为 agent_graph.png")
