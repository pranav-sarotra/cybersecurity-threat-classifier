"""
Logging module.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class SecurityLogger:
    def __init__(self, name: str = "SecurityScanner", log_file: Optional[str] = None, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(console_handler)

        if log_file:
            try:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(file_handler)
            except:
                pass

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def scan_started(self, input_type: str, input_preview: str) -> None:
        preview = input_preview[:50] + "..." if len(input_preview) > 50 else input_preview
        self.info(f"SCAN STARTED - Type: {input_type} - Input: {preview}")

    def scan_completed(self, classification: str, confidence: float, threat_score: float) -> None:
        self.info(f"SCAN COMPLETED - Classification: {classification} - Confidence: {confidence}% - Threat Score: {threat_score}%")


_logger: Optional[SecurityLogger] = None


def get_logger() -> SecurityLogger:
    global _logger
    if _logger is None:
        _logger = SecurityLogger()
    return _logger


def setup_logger(log_file: str = None, level: str = "INFO") -> SecurityLogger:
    global _logger
    _logger = SecurityLogger(log_file=log_file, level=level)
    return _logger