from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt.tool_node import tools_condition
from langgraph.graph import MessagesState
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
import os
import uuid
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
# 导入所有工具
from retriever import lookup_policy
from hotel_tool import search_hotels, book_hotel, update_hotel, cancel_hotel
from car_rental_tools import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from excursions_tools import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
from flight_tools import (
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight, 
    cancel_ticket
)
from utilities_tools import create_tool_node_with_fallback

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: MessagesState, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
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
    base_url=os.environ.get('MODEL_BASE_URL'),
    api_key=os.environ.get('OPENAI_API_KEY'),
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


def get_user_id():
    """获取测试用户ID，实际应用中可以通过登录系统获取"""
    # 这里使用一个示例ID
    return "0000 000001"

def create_agent(passenger_id: str) -> StateGraph:
    
    assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools) 

    builder = StateGraph(MessagesState)

    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools",create_tool_node_with_fallback(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant", tools_condition
    )
    builder.add_edge("tools", "assistant")
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    graph = graph.with_config(
        configurable={
            "passenger_id": passenger_id,
             "thread_id": str(uuid.uuid4()),
        }
    )
    return graph

def process_message(agent, message: str, chat_history: list[BaseMessage] = None) -> dict[str, any]:
    """处理用户消息并返回机器人回复"""
    if chat_history is None:
        chat_history = []
    messages = []
    for msg in chat_history:
        if isinstance(msg, (HumanMessage, AIMessage)):
            messages.append(msg)
    messages.append(HumanMessage(content=message))   

    # 调用图并获取流式响应
    response_chunks = []
    for chunk in agent.stream({"messages": messages}):
        if isinstance(chunk, dict) and "assistant" in chunk:
            assistant_message = chunk["assistant"].get("messages")
            if isinstance(assistant_message, AIMessage) and assistant_message.content:
                response_chunks.append(assistant_message.content)
                yield assistant_message.content

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
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用客户支持系统，再见！")
            break

        # 处理用户消息并实时显示流式响应
        print("\n机器人: ", end='', flush=True)
        response_chunks = []
        
        # 收集流式响应
        for chunk in process_message(agent, user_input, chat_history):
            print(chunk, end='', flush=True)
            response_chunks.append(chunk)
        print()  # 换行
        
        # 更新对话历史
        full_response = "".join(response_chunks)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=full_response))
        

def main():
    """主程序入口点"""
    print("航空公司客户支持机器人")
    print("-" * 40)
    
    # 启动聊天循环
    chat_loop()

if __name__ == "__main__":
    main()