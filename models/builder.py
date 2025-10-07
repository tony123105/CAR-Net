import torch.nn as nn
import config
from models.CarNet import CarNet

def build_model(config) -> nn.Module:
    """
    Model factory. Selects and builds the model based on the config file.
    """
    model_name = config.MODEL_NAME.lower()
    print(f"--- Building model: {model_name} ---")

    if model_name == "car-net":
        model_config = config.RMD_MODEL_CONFIG
        return CarNet(**model_config)

    else:
        raise ValueError(f"Unknown model name: {model_name}. Options are 'car-net'.")