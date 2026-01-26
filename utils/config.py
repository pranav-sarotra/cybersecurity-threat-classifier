"""
Configuration management module.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    ml_weight: float = 0.6
    rule_weight: float = 0.4
    confidence_threshold: float = 0.7
    max_features: int = 5000
    ngram_range: tuple = (1, 3)
    n_estimators: int = 100
    max_depth: int = 10


@dataclass
class APIConfig:
    virustotal: str = ""
    abuseipdb: str = ""
    shodan: str = ""


@dataclass
class RateLimitConfig:
    ip_checks_per_minute: int = 30
    url_checks_per_minute: int = 20


@dataclass
class AppConfig:
    name: str = "XYZ Company Security Portal"
    version: str = "3.0.0"
    author: str = "Pranav Sarotra"
    debug: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    api_keys: APIConfig = field(default_factory=APIConfig)
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    database_path: str = "data/scan_history.db"
    log_file: str = "logs/security_scanner.log"
    log_level: str = "INFO"


class ConfigManager:
    DEFAULT_CONFIG_PATH = "config.yaml"
    LOCAL_CONFIG_PATH = "config_local.yaml"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        config = AppConfig()
        config_path_to_use = self.config_path

        if os.path.exists(self.LOCAL_CONFIG_PATH):
            config_path_to_use = self.LOCAL_CONFIG_PATH

        if os.path.exists(config_path_to_use):
            try:
                with open(config_path_to_use, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config = self._parse_yaml_config(yaml_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

        config = self._apply_env_overrides(config)
        return config

    def _parse_yaml_config(self, yaml_config: Dict) -> AppConfig:
        config = AppConfig()
        app_config = yaml_config.get('app', {})
        config.name = app_config.get('name', config.name)
        config.version = app_config.get('version', config.version)
        config.author = app_config.get('author', config.author)
        config.debug = app_config.get('debug', config.debug)

        model_config = yaml_config.get('model', {})
        config.model = ModelConfig(
            ml_weight=model_config.get('ml_weight', 0.6),
            rule_weight=model_config.get('rule_weight', 0.4),
            confidence_threshold=model_config.get('confidence_threshold', 0.7),
            max_features=model_config.get('vectorizer', {}).get('max_features', 5000),
            ngram_range=tuple(model_config.get('vectorizer', {}).get('ngram_range', [1, 3])),
            n_estimators=model_config.get('random_forest', {}).get('n_estimators', 100),
            max_depth=model_config.get('random_forest', {}).get('max_depth', 10)
        )

        api_config = yaml_config.get('api_keys', {})
        config.api_keys = APIConfig(
            virustotal=api_config.get('virustotal', ''),
            abuseipdb=api_config.get('abuseipdb', ''),
            shodan=api_config.get('shodan', '')
        )

        db_config = yaml_config.get('database', {})
        config.database_path = db_config.get('path', config.database_path)

        log_config = yaml_config.get('logging', {})
        config.log_level = log_config.get('level', config.log_level)
        config.log_file = log_config.get('file', config.log_file)

        return config

    def _apply_env_overrides(self, config: AppConfig) -> AppConfig:
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                try:
                    if 'VIRUSTOTAL_API_KEY' in st.secrets:
                        config.api_keys.virustotal = st.secrets['VIRUSTOTAL_API_KEY']
                except:
                    pass
                try:
                    if 'ABUSEIPDB_API_KEY' in st.secrets:
                        config.api_keys.abuseipdb = st.secrets['ABUSEIPDB_API_KEY']
                except:
                    pass
                try:
                    if 'SHODAN_API_KEY' in st.secrets:
                        config.api_keys.shodan = st.secrets['SHODAN_API_KEY']
                except:
                    pass
        except:
            pass

        # Environment variables
        if os.environ.get('VIRUSTOTAL_API_KEY'):
            config.api_keys.virustotal = os.environ['VIRUSTOTAL_API_KEY']
        if os.environ.get('ABUSEIPDB_API_KEY'):
            config.api_keys.abuseipdb = os.environ['ABUSEIPDB_API_KEY']
        if os.environ.get('SHODAN_API_KEY'):
            config.api_keys.shodan = os.environ['SHODAN_API_KEY']
        if os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes'):
            config.debug = True

        return config

    def has_api_key(self, service: str) -> bool:
        key = getattr(self.config.api_keys, service, '')
        return bool(key and key.strip())

    def get_api_key(self, service: str) -> str:
        return getattr(self.config.api_keys, service, '')


_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager.config


def has_api_key(service: str) -> bool:
    return get_config_manager().has_api_key(service)


def get_api_key(service: str) -> str:
    return get_config_manager().get_api_key(service)