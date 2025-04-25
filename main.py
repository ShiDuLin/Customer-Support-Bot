# main_langgraph.py
from agent import create_agent, process_message

def get_user_id():
    """获取测试用户ID，实际应用中可以通过登录系统获取"""
    # 这里使用一个示例ID
    return "0000 000001"

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
            
        # 处理用户消息
        result = process_message(agent, user_input, chat_history)
        
        # 更新对话历史
        chat_history = result["chat_history"]
        
        # 显示回复
        print(f"\n机器人: {result['response']}")

def main():
    """主程序入口点"""
    print("航空公司客户支持机器人 - LangGraph版本")
    print("-" * 40)
    
    # 启动聊天循环
    chat_loop()

if __name__ == "__main__":
    main()