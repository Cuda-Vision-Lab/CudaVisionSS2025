''' Wrapper fpr logging functionality'''

import logging
import traceback

def log_function(func):
    """
    Decorator for logging a method in case of raising an exception
    """
    def try_call_log(*args, **kwargs):
        """
        Calling the function but calling the logger in case an exception is raised
        """
        try:
            message = f"Calling: {func.__name__}..."
            logging.log_info(message=message, message_type="info")
            return func(*args, **kwargs)
        except Exception as e:
            message = traceback.format_exc()
            print(message, message_type="error")
            exit()
    return try_call_log