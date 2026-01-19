import json
import os
from abc import ABC, abstractmethod
from typing import Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage, SystemMessage  # 导入消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类

from .session_history import get_session_history  # 导入会话历史相关方法
from .llm_factory import create_llm  # 导入 LLM 工厂函数
from .conversation_config import ConversationConfig, get_default_config
from utils.logger import LOG  # 导入日志工具

# 延迟导入长期记忆模块（可选依赖）
try:
    from .long_term_memory import get_memory_instance
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    LOG.warning("[AgentBase] 长期记忆模块不可用（Milvus 未配置）")


class AgentBase(ABC):
    """
    抽象基类，提供代理的共有功能。
    支持 Jinja2 模板、会话配置和长期记忆（Phase 2 新增）。
    """
    def __init__(
        self,
        name,
        prompt_file,
        intro_file=None,
        session_id=None,
        config: Optional[ConversationConfig] = None,
        template_dir: str = "prompts/templates",
        enable_long_term_memory: bool = None,
        user_id: str = "default_user",
    ):
        self.name = name
        self.prompt_file = prompt_file
        self.intro_file = intro_file
        self.session_id = session_id if session_id else self.name
        self.template_dir = template_dir
        self._config = config if config else get_default_config()
        self.user_id = user_id

        # 长期记忆配置（Phase 2）
        # 默认：如果 MILVUS_HOST 配置了，且模块可用，则启用
        if enable_long_term_memory is None:
            enable_long_term_memory = (
                MEMORY_AVAILABLE and
                os.getenv("MILVUS_HOST") is not None
            )

        self.enable_long_term_memory = enable_long_term_memory
        self.memory = None

        if self.enable_long_term_memory:
            if MEMORY_AVAILABLE:
                try:
                    self.memory = get_memory_instance()
                    # 检查是否真正连接成功
                    if self.memory and self.memory.is_connected:
                        LOG.info(f"[{self.name}] 长期记忆已启用")
                    else:
                        LOG.info(f"[{self.name}] Milvus 不可用，长期记忆已禁用")
                        self.enable_long_term_memory = False
                except Exception as e:
                    LOG.warning(f"[{self.name}] 初始化长期记忆失败: {e}")
                    self.enable_long_term_memory = False
            else:
                LOG.warning(f"[{self.name}] 长期记忆模块不可用，已禁用")
                self.enable_long_term_memory = False

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

    def _retrieve_relevant_memories(self, user_input: str) -> str:
        """
        从 Milvus 检索相关的历史记忆（Phase 2）

        参数:
            user_input: 用户输入

        返回:
            str: 格式化的记忆文本
        """
        if not self.enable_long_term_memory or not self.memory:
            return ""

        try:
            # 检索相关记忆
            memories = self.memory.retrieve_relevant_memories(
                user_id=self.user_id,
                query=user_input,
                scenario=self.name,
                top_k=3,
                check_context_limit=True,
            )

            if memories:
                formatted = self.memory.format_memories_for_prompt(memories)
                LOG.debug(f"[{self.name}] 注入 {len(memories)} 条历史记忆到对话上下文")
                return formatted

        except Exception as e:
            LOG.error(f"[{self.name}] 检索记忆失败: {e}")

        return ""

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
        Phase 2 增强：集成长期记忆检索。

        参数:
            user_input (str): 用户输入的消息
            session_id (str, optional): 会话的唯一标识符

        返回:
            str: AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id

        # Phase 2: 检索相关的历史记忆
        memory_context = self._retrieve_relevant_memories(user_input)

        # 如果有记忆上下文，将其作为系统消息注入
        messages = []
        if memory_context:
            messages.append(SystemMessage(content=memory_context))

        messages.append(HumanMessage(content=user_input))

        response = self.chatbot_with_history.invoke(
            messages,  # 传入消息列表（可能包含记忆上下文）
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )

        LOG.debug(f"[ChatBot][{self.name}] {response.content}")  # 记录调试日志
        return response.content  # 返回生成的回复内容

    def chat_with_history_stream(self, user_input, session_id=None):
        """
        处理用户输入，流式生成包含聊天历史的回复。

        参数:
            user_input (str): 用户输入的消息
            session_id (str, optional): 会话的唯一标识符

        Yields:
            str: 逐步生成的回复内容
        """
        if session_id is None:
            session_id = self.session_id

        # Phase 2: 检索相关的历史记忆
        memory_context = self._retrieve_relevant_memories(user_input)

        # 如果有记忆上下文，将其作为系统消息注入
        messages = []
        if memory_context:
            messages.append(SystemMessage(content=memory_context))

        messages.append(HumanMessage(content=user_input))

        # 使用 stream 方法进行流式输出
        full_response = ""
        for chunk in self.chatbot_with_history.stream(
            messages,
            {"configurable": {"session_id": session_id}},
        ):
            # chunk 可能是 AIMessageChunk 或字符串
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)

            if content:
                full_response += content
                yield full_response

        LOG.debug(f"[ChatBot][{self.name}] {full_response}")

    def save_conversation_summary(
        self,
        summary: str,
        session_id: str = None,
        metadata: dict = None
    ) -> bool:
        """
        保存对话摘要到长期记忆（Phase 2）

        参数:
            summary: 对话摘要文本
            session_id: 会话 ID
            metadata: 额外的元数据

        返回:
            bool: 是否保存成功
        """
        if not self.enable_long_term_memory or not self.memory:
            LOG.debug(f"[{self.name}] 长期记忆未启用，跳过保存")
            return False

        if session_id is None:
            session_id = self.session_id

        try:
            # 添加配置信息到元数据
            if metadata is None:
                metadata = {}

            metadata.update({
                "difficulty": self.config.difficulty.value,
                "turns": self.config.turns,
            })

            success = self.memory.store_conversation_summary(
                user_id=self.user_id,
                session_id=session_id,
                scenario=self.name,
                summary=summary,
                metadata=metadata,
            )

            if success:
                LOG.info(f"[{self.name}] 对话摘要已保存到长期记忆")

            return success

        except Exception as e:
            LOG.error(f"[{self.name}] 保存对话摘要失败: {e}")
            return False
