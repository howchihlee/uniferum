from typing import List

import torch
from safetensors import safe_open
from torch.nn import Module


def load_checkpoint(model, ckpt_paths):
    for path in ckpt_paths:
        checkpoint = torch.load(path, map_location=model.device)
        model.load_state_dict(checkpoint, strict=False)
    return model


def load_safetensors(
    model: Module,
    model_path: str,
    filter_prefixs: List[str] = None,
    strict: bool = True,
) -> Module:
    """
    Loads a model's state_dict from a safetensors file and applies optional filtering to exclude specific keys.

    Args:
        model (Module): The PyTorch model to load the weights into.
        model_path (str): Path to the safetensors file.
        filter_prefixst (List[str], optional): A set of keys to filter out from the state_dict. Default is None.
        strict (bool): Whether to strictly enforce that the keys in the model's state_dict match the file. Default is True.

    Returns:
        model (Module): The model with the loaded weights.
    """
    try:
        with safe_open(model_path, framework="pt") as f:
            state_dict = {}
            for key in f.keys():
                if filter_prefixs is not None and any(f in key for f in filter_prefixs):
                    continue
                state_dict[key] = f.get_tensor(key)
        # TODO: log how many variables are loaded
        model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors from {model_path}: {e}")

    return model
