# tabs/scenario_tab.py

import gradio as gr
from agents.scenario_agent import ScenarioAgent
from utils.logger import LOG

# 初始化场景代理
agents = {
    "job_interview": ScenarioAgent("job_interview"),
    "hotel_checkin": ScenarioAgent("hotel_checkin"),
    "renting": ScenarioAgent("renting"),
    "salary_negotiation": ScenarioAgent("salary_negotiation"),
    # 可以根据需要添加更多场景代理
}

def get_page_desc(scenario):
    try:
        with open(f"content/page/{scenario}.md", "r", encoding="utf-8") as file:
            scenario_intro = file.read().strip()
        return scenario_intro
    except FileNotFoundError:
        LOG.error(f"场景介绍文件 content/page/{scenario}.md 未找到！")
        return "场景介绍文件未找到。"
    
# 获取场景介绍并启动新会话的函数
def start_new_scenario_chatbot(scenario):
    """
    切换场景时启动新的聊天会话，清除之前的聊天历史。
    
    参数:
        scenario: 场景名称
        
    返回:
        list: 包含初始AI消息的消息列表，用于重置聊天界面
    """
    LOG.info(f"[Scenario] Switching to scenario: {scenario}")
    
    # 启动新场景的会话并清除历史
    # 这会清除当前场景的会话历史，确保每次切换场景都是全新的会话
    initial_ai_message = agents[scenario].start_new_session()  # 启动新会话并清除历史
    
    # Gradio 6.0.0 使用字典格式的消息
    # 返回新的消息列表会替换聊天机器人中的所有历史消息
    # 这应该清除 ChatInterface 中显示的旧消息
    LOG.debug(f"[Scenario] Returning new initial message for chatbot reset")
    return [{"role": "assistant", "content": initial_ai_message}]


def create_scenario_tab():
    with gr.Tab("场景"):  # 场景标签
        gr.Markdown("## 选择一个场景完成目标和挑战")  # 场景选择说明

        # 创建单选框组件
        scenario_radio = gr.Radio(
            choices=[
                ("求职面试", "job_interview"),  # 求职面试选项
                ("酒店入住", "hotel_checkin"),  # 酒店入住选项
                ("租房", "renting"),  # 租房选项
                ("薪资谈判", "salary_negotiation"),  # 薪资谈判选项
            ], 
            label="场景"  # 单选框标签
        )

        scenario_intro = gr.Markdown()  # 场景介绍文本组件
        
        # 使用 State 来跟踪当前场景
        current_scenario_state = gr.State(value=None)
        
        scenario_chatbot = gr.Chatbot(
            placeholder="<strong>你的英语私教 LangCoach</strong><br><br>选择场景后开始对话吧！",  # 聊天机器人的占位符
            height=600,  # 聊天窗口高度
            value=None,  # 初始值为空
        )
        
        # 手动创建聊天输入框和发送按钮，放在同一行
        with gr.Row():
            scenario_input = gr.Textbox(
                placeholder="输入你的消息...",
                label="消息",
                scale=9,
                container=False,
            )
            scenario_submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=100)

        # 更新场景介绍并在场景变化时启动新会话
        def on_scenario_change(scenario, current_state):
            """处理场景切换，重置聊天界面"""
            LOG.info(f"[Scenario] Scenario changed from {current_state} to: {scenario}")
            
            # 如果场景发生变化，清除之前场景的会话历史
            if current_state and current_state != scenario and current_state in agents:
                LOG.debug(f"[Scenario] Clearing previous scenario history: {current_state}")
                agents[current_state].start_new_session()  # 清除之前场景的历史
            
            # 启动新场景的会话
            intro = get_page_desc(scenario)
            new_chat_history = start_new_scenario_chatbot(scenario)
            
            LOG.debug(f"[Scenario] Returning intro and new chat history for scenario: {scenario}, history: {new_chat_history}")
            # 直接返回新消息列表，这会替换 chatbot 中的所有旧消息
            return intro, new_chat_history, scenario
        
        scenario_radio.change(
            fn=on_scenario_change,  # 更新场景介绍和聊天机器人
            inputs=[scenario_radio, current_scenario_state],  # 输入为选择的场景和当前状态
            outputs=[scenario_intro, scenario_chatbot, current_scenario_state],  # 输出为场景介绍、聊天机器人和新状态
        )

        # 处理用户消息的函数
        def on_message_submit(user_input, chat_history, scenario):
            """处理用户提交的消息"""
            if not user_input or not user_input.strip():
                return chat_history or [], ""
            
            LOG.debug(f"[Scenario] User message submitted for scenario: {scenario}")
            
            # 确保 chat_history 不为 None
            if chat_history is None:
                chat_history = []
            
            # 创建新的聊天历史列表（避免直接修改原列表）
            new_chat_history = list(chat_history) if chat_history else []
            
            # 添加用户消息到聊天历史
            new_chat_history.append({"role": "user", "content": user_input.strip()})
            
            # 获取AI回复
            bot_message = agents[scenario].chat_with_history(user_input.strip())
            
            # 添加AI回复到聊天历史
            new_chat_history.append({"role": "assistant", "content": bot_message})
            
            LOG.debug(f"[Scenario] Chat history updated, length: {len(new_chat_history)}")
            return new_chat_history, ""  # 返回更新后的聊天历史和清空的输入框
        
        # 绑定提交事件
        scenario_submit_btn.click(
            fn=lambda msg, hist, scen: on_message_submit(msg, hist, scen),
            inputs=[scenario_input, scenario_chatbot, scenario_radio],
            outputs=[scenario_chatbot, scenario_input],
        )
        
        # 也支持回车键提交
        scenario_input.submit(
            fn=lambda msg, hist, scen: on_message_submit(msg, hist, scen),
            inputs=[scenario_input, scenario_chatbot, scenario_radio],
            outputs=[scenario_chatbot, scenario_input],
        )
