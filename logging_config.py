from loguru import logger
import sys
from pathlib import Path

# Remove default handler and create logs directory
logger.remove()
Path("logs").mkdir(exist_ok=True)

# Configure format to include module name
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[module]:<15}</cyan> | "
    "<level>{message}</level>"
)

# Add handlers for both console and single log file
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level="INFO",
    enqueue=True,
    # colorize=True,
    catch=True
)

logger.add(
    "logs/main.log",
    format=LOG_FORMAT,
    level="INFO",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    enqueue=True,
    # colorize=True,
    catch=True
)

logger.add(
    "logs/errors.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    enqueue=True,
    # colorize=True,
    catch=True
)

logger.add(
    "logs/debug.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    enqueue=True,
    # colorize=True,
    catch=True
)
def get_logger(module_name: str):
    """Get a logger instance with module context"""
    return logger.bind(module=module_name)

# Initialize logging with a default module name
logger = logger.bind(module="system")
logger.info("Logging system initialized")