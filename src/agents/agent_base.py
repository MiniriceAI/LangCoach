import json
from abc import ABC, abstractmethod
from typing import Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage  # 导入消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类

from .session_history import get_session_history  # 导入会话历史相关方法
from .llm_factory import create_llm  # 导入 LLM 工厂函数
from .conversation_config import ConversationConfig, get_default_config
from utils.logger import LOG  # 导入日志工具


class AgentBase(ABC):
    """
    抽象基类，提供代理的共有功能。
    支持 Jinja2 模板和会话配置。
    """
    def __init__(
        self,
        name,
        prompt_file,
        intro_file=None,
        session_id=None,
        config: Optional[ConversationConfig] = None,
        template_dir: str = "prompts/templates"
    ):
        self.name = name
        self.prompt_file = prompt_file
        self.intro_file = intro_file
        self.session_id = session_id if session_id else self.name
        self.template_dir = template_dir
        self._config = config if config else get_default_config()

        # 初始化 Jinja2 环境
        self._jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False
        )

        self.prompt = self.load_prompt()
        self.intro_messages = self.load_intro() if self.intro_file else []
        self.create_chatbot()

    @property
    def config(self) -> ConversationConfig:
        """获取当前会话配置。"""
        return self._config

    def update_config(self, config: ConversationConfig):
        """
        更新会话配置并重新创建聊天机器人。

        参数:
            config (ConversationConfig): 新的会话配置
        """
        self._config = config
        self.prompt = self.load_prompt()
        self.create_chatbot()
        LOG.info(f"[{self.name}] Config updated: turns={config.turns}, difficulty={config.difficulty.value}")

    def load_prompt(self):
        """
        从 Jinja2 模板文件加载并渲染系统提示语。
        如果模板不存在，回退到原始 txt 文件。
        """
        # 尝试加载 Jinja2 模板
        template_name = self.prompt_file.replace("prompts/", "").replace(".txt", ".j2")
        try:
            template = self._jinja_env.get_template(template_name)
            template_vars = self._config.to_template_vars()
            rendered_prompt = template.render(**template_vars)
            LOG.debug(f"[{self.name}] Loaded Jinja2 template: {template_name}")
            return rendered_prompt.strip()
        except TemplateNotFound:
            LOG.debug(f"[{self.name}] Jinja2 template not found, falling back to txt: {self.prompt_file}")
            # 回退到原始 txt 文件
            try:
                with open(self.prompt_file, "r", encoding="utf-8") as file:
                    return file.read().strip()
            except FileNotFoundError:
                raise FileNotFoundError(f"找不到提示文件 {self.prompt_file}!")

    def load_intro(self):
        """
        从 JSON 文件加载初始消息。
        """
        try:
            with open(self.intro_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到初始消息文件 {self.intro_file}!")
        except json.JSONDecodeError:
            raise ValueError(f"初始消息文件 {self.intro_file} 包含无效的 JSON!")

    def create_chatbot(self):
        """
        初始化聊天机器人，包括系统提示和消息历史记录。
        """
        # 创建聊天提示模板，包括系统提示和消息占位符
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        # 根据环境变量选择合适的 LLM 提供者
        llm = create_llm()

        # 组合提示模板和 LLM
        self.chatbot = system_prompt | llm

        # 将聊天机器人与消息历史记录关联
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)

    def chat_with_history(self, user_input, session_id=None):
        """
        处理用户输入，生成包含聊天历史的回复。

        参数:
            user_input (str): 用户输入的消息
            session_id (str, optional): 会话的唯一标识符

        返回:
            str: AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id

        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )

        LOG.debug(f"[ChatBot][{self.name}] {response.content}")  # 记录调试日志
        return response.content  # 返回生成的回复内容
