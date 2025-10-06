"""Logging utility for BloomBotanics system"""

import logging
import os
from datetime import datetime
from config import LOG_DIR, LOG_LEVEL, MAX_LOG_SIZE

def get_logger(name):
    """Get configured logger instance with rotation"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set log level
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)
        
        # Create log file path
        log_file = os.path.join(LOG_DIR, 'bloom_botanics.log')
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=MAX_LOG_SIZE, 
            backupCount=5
        )
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatter with emoji support
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
