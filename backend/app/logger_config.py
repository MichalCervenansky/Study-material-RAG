import logging
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name):
    return logging.getLogger(name)

def log_time(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Finished {func.__name__} in {duration:.2f} seconds")
            return result
        return wrapper
    return decorator
