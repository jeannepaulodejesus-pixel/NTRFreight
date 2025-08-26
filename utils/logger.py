# utils.py

import logging

def get_logger(name="ntr_to_standard", level=logging.INFO):
    """
    Returns a configured logger.
    
    Parameters:
        name (str): Name of the logger.
        level (int): Logging level (default: logging.INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid adding multiple handlers if called multiple times
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger