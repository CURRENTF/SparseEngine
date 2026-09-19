import os
import sys
from loguru import logger


def quick_debug_print(something):
    return f'{something}'


# 移除默认的 handler
logger.remove()

# 添加自定义格式的 handler
# 格式包含：时间 | 级别 | 文件名:函数名:行号 - 消息
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level
)

# 导出 logger 供其他模块使用
__all__ = ["logger", "log_once"]

_seen_messages = set()


def log_once(msg: str, level: str = 'INFO'):
    """
    仅记录一次日志消息。使用 set 记录已发送的消息。
    """
    if msg not in _seen_messages:
        # 使用 depth=1 让 loguru 报告调用者的文件和行号
        logger.opt(depth=1).log(level.upper(), msg)
        _seen_messages.add(msg)
