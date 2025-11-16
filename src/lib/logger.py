"""
Logging utility untuk aplikasi CCTV Vehicle Detection
"""
import logging
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter dengan warna untuk terminal"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format timestamp
        record.asctime = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        
        # Format dengan warna
        log_fmt = f"{log_color}[%(asctime)s] [%(levelname)-8s]{reset_color} %(message)s"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name='vehicle_detection', log_level=logging.INFO, log_file=None):
    """
    Setup logger dengan colored output dan optional file logging
    
    Args:
        name: Logger name
        log_level: Logging level (default: INFO)
        log_file: Optional path ke log file
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler dengan warna
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler (jika diperlukan)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)-8s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()
