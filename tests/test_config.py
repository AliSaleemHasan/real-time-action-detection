import os
import sys
import yaml
import pytest
from pydantic import ValidationError

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config_schema import Config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

@pytest.fixture
def valid_config_data():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def test_config_exists():
    """Test that the config.yaml file exists."""
    assert os.path.exists(CONFIG_PATH), "config.yaml not found in project root"

def test_config_validation(valid_config_data):
    """Test that the current config.yaml is valid according to the schema."""
    try:
        config = Config(**valid_config_data)
        assert config.no_classes == len(config.classes)
        assert config.threshold >= 0.0 and config.threshold <= 1.0
    except ValidationError as e:
        pytest.fail(f"Config validation failed: {e}")

def test_invalid_threshold(valid_config_data):
    """Test that invalid threshold raises ValidationError."""
    data = valid_config_data.copy()
    data['threshold'] = 1.5 # Invalid: > 1.0
    with pytest.raises(ValidationError) as excinfo:
        Config(**data)
    assert 'threshold' in str(excinfo.value)

def test_class_mismatch(valid_config_data):
    """Test that mismatch between no_classes and classes length raises ValidationError."""
    data = valid_config_data.copy()
    data['classes'] = ['A', 'B']
    data['no_classes'] = 5 # Mismatch
    with pytest.raises(ValidationError) as excinfo:
        Config(**data)
    # The error comes from model_validator, check message
    assert 'Number of classes' in str(excinfo.value)

def test_missing_field(valid_config_data):
    """Test that missing required field raises ValidationError."""
    data = valid_config_data.copy()
    del data['data_directory']
    with pytest.raises(ValidationError) as excinfo:
        Config(**data)
    assert 'data_directory' in str(excinfo.value)

def test_alias_support(valid_config_data):
    """Test that 'theshold' alias still works if we use it."""
    data = valid_config_data.copy()
    # If we use 'theshold' instead of 'threshold'
    if 'threshold' in data:
        del data['threshold']
    data['theshold'] = 0.5
    config = Config(**data)
    assert config.threshold == 0.5
