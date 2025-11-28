# tabs/vocab_tab.py

import gradio as gr
from agents.vocab_agent import VocabAgent
from utils.logger import LOG

# 初始化词汇代理，负责管理词汇学习会话
vocab_agent = VocabAgent()

# 定义功能名称为“vocab_study”，表示词汇学习模块
feature = "vocab_study"

# 获取页面描述，从指定的 markdown 文件中读取介绍内容
def get_page_desc(feature):
    try:
        # 打开指定的 markdown 文件来读取词汇学习介绍
        with open(f"content/page/{feature}.md", "r", encoding="utf-8") as file:
            scenario_intro = file.read().strip()  # 去除多余空白
        return scenario_intro
    except FileNotFoundError:
        # 如果找不到文件，记录错误并返回默认消息
        LOG.error(f"词汇学习介绍文件 content/page/{feature}.md 未找到！")
        return "词汇学习介绍文件未找到。"

# 重新启动词汇学习聊天机器人会话
def restart_vocab_study_chatbot():
    """
    重置词汇学习会话：
    1. 清除会话历史
    2. 开启新话题，让 LLM 生成新的 5 个单词
    3. 返回新的初始消息列表
    
    返回:
        list: 包含初始 AI 消息（新单词介绍）的消息列表
    """
    LOG.info("[Vocab] 重置会话，开始新的一关")
    
    # 清除会话历史，确保每次重启都是全新的会话
    vocab_agent.restart_session()
    
    # 根据 prompt 文件，LLM 会在收到 "Let's do it" 时开始生成新的单词
    # 但为了更明确，我们直接让 LLM 生成新单词介绍
    # 使用 prompt 中定义的初始触发词
    initial_message = "Let's do it"
    bot_message = vocab_agent.chat_with_history(initial_message)
    
    LOG.info(f"[Vocab] 新话题开始，AI回复长度: {len(bot_message)} 字符")
    LOG.debug(f"[Vocab] AI回复预览: {bot_message[:200]}...")

    # Gradio 6.0.0 使用字典格式的消息
    # 返回新的初始消息列表，这会完全替换聊天机器人中的所有历史消息
    # 包括用户消息和 AI 回复（新单词介绍）
    new_chat_history = [
        {"role": "user", "content": initial_message},
        {"role": "assistant", "content": bot_message}
    ]
    
    return new_chat_history

# 创建词汇学习的 Tab 界面
def create_vocab_tab():
    # 创建一个 Tab，标题为“单词”
    with gr.Tab("单词"):
        gr.Markdown("## 闯关背单词")  # 添加 Markdown 标题

        # 显示从文件中获取的页面描述
        gr.Markdown(get_page_desc(feature))

        # 初始化一个聊天机器人组件，设置占位符文本和高度
        vocab_study_chatbot = gr.Chatbot(
            placeholder="<strong>你的英语私教 LangCoach</strong><br><br>点击「下一关」开始学习新单词！",
            height=600,
            value=None,  # 初始值为空
        )

        # 创建一个按钮，用于重置词汇学习状态，值为“下一关”
        restart_btn = gr.Button(value="下一关", variant="primary", size="lg")

        # 手动创建聊天输入框和发送按钮
        with gr.Row():
            vocab_input = gr.Textbox(
                placeholder="输入你的消息...",
                label="消息",
                scale=9,
                container=False,
            )
            vocab_submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=100)

        # 当用户点击「下一关」按钮时，重置会话并生成新的单词
        restart_btn.click(
            fn=restart_vocab_study_chatbot,
            inputs=[],  # 空列表表示没有输入
            outputs=vocab_study_chatbot,  # 输出到聊天机器人组件
        )

        # 处理用户消息的函数
        def on_message_submit(user_input, chat_history):
            """处理用户提交的消息"""
            if not user_input or not user_input.strip():
                return chat_history or [], ""
            
            LOG.debug("[Vocab] User message submitted")
            
            # 确保 chat_history 不为 None
            if chat_history is None:
                chat_history = []
            
            # 创建新的聊天历史列表（避免直接修改原列表）
            new_chat_history = list(chat_history) if chat_history else []
            
            # 添加用户消息到聊天历史
            new_chat_history.append({"role": "user", "content": user_input.strip()})
            
            # 获取AI回复
            bot_message = vocab_agent.chat_with_history(user_input.strip())
            
            # 添加AI回复到聊天历史
            new_chat_history.append({"role": "assistant", "content": bot_message})
            
            LOG.debug(f"[Vocab] Chat history updated, length: {len(new_chat_history)}")
            return new_chat_history, ""  # 返回更新后的聊天历史和清空的输入框
        
        # 绑定提交事件
        vocab_submit_btn.click(
            fn=on_message_submit,
            inputs=[vocab_input, vocab_study_chatbot],
            outputs=[vocab_study_chatbot, vocab_input],
        )
        
        # 也支持回车键提交
        vocab_input.submit(
            fn=on_message_submit,
            inputs=[vocab_input, vocab_study_chatbot],
            outputs=[vocab_study_chatbot, vocab_input],
        )
