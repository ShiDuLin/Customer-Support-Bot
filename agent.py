import os
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# LangGraph 导入
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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

# 构建工具列表
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

# 创建语言模型
model = ChatOpenAI(
    base_url="https://api.chatanywhere.tech/v1",  # 或者使用环境变量
    api_key=os.environ.get('OPENAI_API_KEY'),
    model="gpt-3.5-turbo",
    temperature=0
)

# 使用LangGraph创建智能体
def create_agent(passenger_id: str):
    """使用LangGraph创建客户支持智能体"""
    
    # 系统提示模板
    system_prompt = f"""
    你是一个航空公司的客户支持智能体。你的任务是帮助客户解决他们的问题，包括：
    - 查询航班信息和政策
    - 管理航班预订、改签和取消
    - 处理酒店、租车和旅游活动的预订
    - 提供其他旅行相关服务

    在回答问题前，请先查询相关政策确保你的回答符合公司规定。
    使用工具帮助你解决客户问题。

    遵循以下原则：
    1. 始终保持专业、礼貌和有帮助的态度
    2. 回答应该简洁明了
    3. 如果不确定，先查询政策
    4. 在做任何更改前获取客户的确认
    5. 提供全面的解决方案

    当前用户ID: {passenger_id}
    """
    
    # 绑定工具到LLM
    llm_with_tools = model.bind_tools(tools)
    
    # 创建助手节点
    def assistant(state: MessagesState) -> Dict[str, Any]:
        """助手节点 - 分析用户消息并决定使用工具或回复"""
        system_message = SystemMessage(content=system_prompt)
        messages = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # 创建工具节点（带错误处理）
    tools_node = create_tool_node_with_fallback(tools)
    
    # 构建图
    builder = StateGraph(MessagesState)
    
    # 添加节点
    builder.add_node("assistant", assistant)
    builder.add_node("tools", tools_node)
    
    # 添加边
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", "assistant")
    
    # 创建内存持久化
    memory = MemorySaver()
    
    # 编译图
    graph = builder.compile(checkpointer=memory)
    
    # 配置工具的用户ID
    graph = graph.with_config(
        configurable={
            "passenger_id": passenger_id,
        }
    )
    
    return graph

# 处理用户消息
def process_message(agent_graph, message: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
    """处理用户消息并返回智能体回复"""
    if chat_history is None:
        chat_history = []
    
    thread_id = f"thread_{hash(str(chat_history))}"
    config = {"configurable": {"thread_id": thread_id}}
    # 转换现有聊天历史为消息列表
    messages = []
    for msg in chat_history:
        if isinstance(msg, (HumanMessage, AIMessage)):
            messages.append(msg)
    
    # 添加当前用户消息
    messages.append(HumanMessage(content=message))
    
    # 调用图
    result = agent_graph.invoke({"messages": messages}, config)
    
    # 获取最后的助手消息
    response = result["messages"][-1].content
    
    # 更新对话历史
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=response))
    
    return {
        "response": response,
        "chat_history": chat_history
    }