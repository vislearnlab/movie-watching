import logging
import sys
from pathlib import Path
from datetime import datetime
import traceback

def setup_logging(subject_id, log_dir="../data/logs", debug=False):
    """
    Setup logging configuration for the experiment.
    
    Args:
        subject_id: Subject ID for log file naming
        log_dir: Directory to store log files
        debug: If True, set logging level to DEBUG, otherwise INFO
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{subject_id}_{timestamp}_session.log"
    
    # Create logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create file handler for detailed logging
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    
    # Create console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO and above in console
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Set formatters
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for subject {subject_id}")
    logger.info(f"Session log file: {log_file}")
    
    return logger

def log_exception(logger, exception, context=""):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exception: Exception object
        context: Additional context about where the exception occurred
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    if context:
        logger.error(f"Exception in {context}: {str(exception)}")
    else:
        logger.error(f"Exception occurred: {str(exception)}")
    
    # Log full traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in tb_lines:
        logger.error(line.rstrip())

def safe_execute(logger, func, context="", *args, **kwargs):
    """
    Execute a function with automatic exception logging.
    
    Args:
        logger: Logger instance
        func: Function to execute
        context: Description of what's being executed
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result of func or None if exception occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_exception(logger, e, context)
        return None