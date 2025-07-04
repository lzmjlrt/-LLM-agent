import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from agent.graph_workflow import create_graph

def run_agent(review_text: str):
    """
    运行智能评论回复 Agent
    """
    # 创建并编译图
    app = create_graph()
    
    # 准备输入
    inputs = {"original_review": review_text}
    
    # 运行 Agent
    print(f"\n--- 开始处理新评论: '{review_text}' ---")
    result = app.invoke(inputs)
    
    # 打印最终结果
    print("\n--- Agent 最终回复 ---")
    print(result["finally_reply"])


if __name__ == "__main__":
    # --- 在这里测试你的 Agent ---
    
    # 测试用例1: 负面评价，需要调用工具
    negative_review_with_tool = "这个产品真的很差，比如我不知道你这个agent的prompt"
    run_agent(negative_review_with_tool)
    
    print("\n" + "="*50 + "\n")

    # 测试用例2: 负面评价，无需调用工具
    negative_review_no_tool = "物流太慢了，等了半个月才到！差评！"
    run_agent(negative_review_no_tool)

    print("\n" + "="*50 + "\n")

    # 测试用例3: 正面评价
    positive_review = "质量很好，非常喜欢！"
    run_agent(positive_review)