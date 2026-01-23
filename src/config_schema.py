import os
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

class LayerConfig(BaseModel):
    layer: str
    filters: Optional[int] = None
    kernel_size: Optional[int] = None
    activation: Optional[str] = None
    pool_size: Optional[int] = None
    units: Optional[int] = Field(None, gt=0)
    drop_perc: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_layer_params(self):
        layer_type = self.layer
        if layer_type == 'Conv1D':
            if self.filters is None or self.kernel_size is None:
                raise ValueError('Conv1D layer requires "filters" and "kernel_size"')
        elif layer_type == 'MaxPooling1D':
            if self.pool_size is None:
                raise ValueError('MaxPooling1D layer requires "pool_size"')
        elif layer_type == 'Dense':
            if self.units is None:
                raise ValueError('Dense layer requires "units"')
        elif layer_type == 'Dropout':
            if self.drop_perc is None:
                raise ValueError('Dropout layer requires "drop_perc"')
        return self

class Config(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # General Settings
    data_directory: str = Field(..., description="Directory containing the dataset")
    test_set_path: str = Field(default="", description="Path to test set, can be empty")
    classes: List[str] = Field(..., description="List of action classes")
    no_classes: int = Field(..., gt=0, description="Number of classes")
    test_size: float = Field(..., ge=0.0, le=1.0, description="Fraction of dataset to use for testing")
    threshold: float = Field(..., ge=0.0, le=1.0, alias="theshold", description="Threshold for keypoints and bounding boxes")
    
    # Model Configuration
    model_directory: str
    sequence_length: int = Field(..., gt=0)
    no_sequences: int = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    optimizer: str
    loss: str
    log_path: str
    saved_weights_path: str
    frame_distance: int = Field(..., gt=0)
    
    # Architecture
    architecture: Literal['LSTM', 'CNN1D']
    model: List[LayerConfig]

    @field_validator('no_classes')
    @classmethod
    def validate_class_count(cls, v, info):
        # We can't easily validate against len(classes) here without accessing 'values' logic which depends on verification order
        # But using model_validator is better for cross-field validation
        return v

    @model_validator(mode='after')
    def check_class_consistency(self):
        if len(self.classes) != self.no_classes:
            raise ValueError(f"Number of classes ({self.no_classes}) does not match the length of the classes list ({len(self.classes)})")
        return self
