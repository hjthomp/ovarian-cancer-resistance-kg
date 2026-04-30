"""
Utility functions for loading and saving various file formats (JSON, YAML, TXT).
Ensures consistent encoding (UTF-8) and error handling across the project.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Loads the YAML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.error(f"Error loading file: {e}")
        return None

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Loads a JSON file and returns a dictionary."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading file: {e}")
        return None

def save_json(data: Union[Dict, list], output_path: Path, indent: int = 2) -> None:
    """Saves data to a JSON file with specified indentation."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return 
    except IOError as e:
        logging.error(f"Failed to save data: {e}")
        raise 

def load_txt(path: Path) -> Optional[str]:
    """Reads a text file and returns its content as a single string."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as e:
        logging.error(f"Error loading file: {e}")
        return None
