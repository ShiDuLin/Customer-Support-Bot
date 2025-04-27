import os
import uuid
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt.tool_node import tools_condition

from car_rental_tools import (
    book_car_rental,
    cancel_car_rental,
    search_car_rentals,
    update_car_rental,
)
from excursions_tools import (
    book_excursion,
    cancel_excursion,
    search_trip_recommendations,
    update_excursion,
)
from flight_tools import (
    cancel_ticket,
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
)
from hotel_tool import book_hotel, cancel_hotel, search_hotels, update_hotel

# 导入所有工具
from retriever import lookup_policy
from utilities_tools import create_tool_node_with_fallback


class State(MessagesState):
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


llm = ChatOpenAI(
    base_url=os.environ.get("MODEL_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=1,
    streaming=True,
)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

tools = [
    # 航班工具
    lookup_policy,
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
    # 酒店工具
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    # 租车工具
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    # 旅游工具
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


def get_user_id():
    """获取测试用户ID，实际应用中可以通过登录系统获取"""
    # 这里使用一个示例ID
    return "0000 000001"


def create_agent(passenger_id: str) -> StateGraph:
    assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

    builder = StateGraph(State)

    builder.add_node("fetch_user_info", user_info)
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(tools))

    builder.add_edge(START, "fetch_user_info")
    builder.add_edge("fetch_user_info", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
    graph = graph.with_config(
        configurable={
            "passenger_id": passenger_id,
            "thread_id": str(uuid.uuid4()),
        }
    )
    return graph


# def _print_event(event, printed):
#     """打印事件，避免重复打印"""
#     for key, value in event.items():
#         if key not in printed:
#             print(f"\n=== {key} 事件 ===")
#             if key == "assistant":
#                 if "messages" in value:
#                     message = value["messages"]
#                     # 处理 AIMessage 对象
#                     if isinstance(message, AIMessage):
#                         if message.content:
#                             print(f"AI回复: {message.content}")
                        
#                         # 处理工具调用
#                         if hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
#                             tool_calls = message.additional_kwargs["tool_calls"]
#                             print("\n工具调用详情:")
#                             for tool_call in tool_calls:
#                                 print(f"- 工具名称: {tool_call['function']['name']}")
#                                 print(f"- 调用ID: {tool_call['id']}")
#                                 print(f"- 参数: {tool_call['function']['arguments']}")
                        
#                         # 处理响应元数据
#                         if hasattr(message, "response_metadata"):
#                             metadata = message.response_metadata
#                             print("\n响应元数据:")
#                             print(f"- 完成原因: {metadata.get('finish_reason', 'unknown')}")
#                             print(f"- 模型名称: {metadata.get('model_name', 'unknown')}")
            
#             elif key == "messages" and value:
#                 if isinstance(value, list) and value:
#                     last_message = value[-1]
#                     if hasattr(last_message, "content"):
#                         print(f"消息内容: {last_message.content}")
#                     if hasattr(last_message, "tool_calls"):
#                         print("工具调用:")
#                         for tool_call in last_message.tool_calls:
#                             print(f"- {tool_call}")
            
#             printed.add(key)


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
            result = list(agent.stream(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=events["messages"][-1].tool_calls[0]["id"],
                            content=f"API调用被用户拒绝。原因: '{user_input}'。请继续协助，考虑用户的输入。",
                        )
                    ]
                },
            ))

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
    main()
