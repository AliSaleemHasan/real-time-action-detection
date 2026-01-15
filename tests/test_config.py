import os
import sys
import yaml
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

def test_config_exists():
    """Test that the config.yaml file exists."""
    assert os.path.exists(CONFIG_PATH), "config.yaml not found in project root"

def test_config_structure():
    """Test that config.yaml contains all required keys."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = [
        'data_directory',
        'test_set_path',
        'classes',
        'sequence_length',
        'no_sequences',
        'epochs',
        'optimizer',
        'loss',
        'model'
    ]
    
    for key in required_keys:
        assert key in config, f"Missing key '{key}' in config.yaml"
    
    assert isinstance(config['classes'], list), "'classes' must be a list"
    assert len(config['classes']) > 0, "'classes' list cannot be empty"
