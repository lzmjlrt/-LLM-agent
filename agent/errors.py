class AgentAppError(Exception):
    """应用层可预期错误。"""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class AgentInitializationError(AgentAppError):
    """初始化阶段错误。"""


class AgentRuntimeError(AgentAppError):
    """运行阶段错误。"""


class ConfigValidationError(AgentAppError):
    """配置校验错误。"""
