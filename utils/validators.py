"""
Input validation utilities.
"""

import re
import ipaddress
from typing import Tuple, Optional
from urllib.parse import urlparse


class InputValidator:
    MAX_TEXT_LENGTH = 50000
    MAX_URL_LENGTH = 2048

    @staticmethod
    def validate_text_input(text: str) -> Tuple[bool, str, Optional[str]]:
        if not text:
            return False, "", "Input text is empty"
        if len(text) > InputValidator.MAX_TEXT_LENGTH:
            return False, "", f"Input exceeds maximum length of {InputValidator.MAX_TEXT_LENGTH}"
        sanitized = text.replace('\x00', '')
        return True, sanitized, None

    @staticmethod
    def validate_ip_address(ip: str) -> Tuple[bool, str, Optional[str]]:
        if not ip:
            return False, "", "IP address is empty"
        ip = ip.strip()
        try:
            ip_obj = ipaddress.ip_address(ip)
            return True, str(ip_obj), None
        except ValueError:
            return False, "", f"Invalid IP address format: {ip}"

    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str, Optional[str]]:
        if not url:
            return False, "", "URL is empty"
        url = url.strip()
        if len(url) > InputValidator.MAX_URL_LENGTH:
            return False, "", f"URL exceeds maximum length"
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False, "", "Invalid URL: no domain"
            return True, url, None
        except Exception as e:
            return False, "", f"URL parsing error: {str(e)}"

    @staticmethod
    def validate_file_hash(hash_str: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
        if not hash_str:
            return False, "", None, "Hash is empty"
        hash_str = hash_str.strip().lower()
        if not re.match(r'^[a-f0-9]+$', hash_str):
            return False, "", None, "Hash contains invalid characters"
        hash_types = {32: 'MD5', 40: 'SHA-1', 64: 'SHA-256', 128: 'SHA-512'}
        hash_type = hash_types.get(len(hash_str))
        if not hash_type:
            return False, "", None, f"Unrecognized hash length: {len(hash_str)}"
        return True, hash_str, hash_type, None