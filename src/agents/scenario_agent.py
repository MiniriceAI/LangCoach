import random
from typing import Optional

from langchain_core.messages import AIMessage  # 导入消息类

from .session_history import get_session_history  # 导入会话历史相关方法
from .agent_base import AgentBase
from .conversation_config import ConversationConfig
from utils.logger import LOG


class ScenarioAgent(AgentBase):
    """
    场景代理类，负责处理特定场景下的对话。
    支持会话配置（对话轮数、难度级别）。
    """
    def __init__(
        self,
        scenario_name,
        session_id=None,
        config: Optional[ConversationConfig] = None
    ):
        prompt_file = f"prompts/{scenario_name}_prompt.txt"
        intro_file = f"content/intro/{scenario_name}.json"
        super().__init__(
            name=scenario_name,
            prompt_file=prompt_file,
            intro_file=intro_file,
            session_id=session_id,
            config=config
        )

    def start_new_session(self, session_id=None, config: Optional[ConversationConfig] = None):
        """
        开始一个新的场景会话，清除之前的会话历史并发送随机的初始 AI 消息。

        参数:
            session_id (str, optional): 会话的唯一标识符
            config (ConversationConfig, optional): 新的会话配置

        返回:
            str: 初始 AI 消息
        """
        if session_id is None:
            session_id = self.session_id

        # 如果提供了新配置，更新代理配置
        if config is not None:
            self.update_config(config)

        history = get_session_history(session_id)
        LOG.debug(f"[history][{session_id}] before clear: {len(history.messages)} messages")

        # 清除之前的会话历史，确保每次切换场景时都开始全新的会话
        history.clear()

        # 随机选择初始AI消息并添加到历史记录
        initial_ai_message = random.choice(self.intro_messages)
        history.add_message(AIMessage(content=initial_ai_message))

        LOG.debug(f"[history][{session_id}] after clear and add initial message")
        return initial_ai_message
