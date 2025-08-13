import os
from collections import defaultdict

import monai.transforms
import yaml
from tap import Tap

from models.utils import load_safetensors


def load_checkpoint_from_config_if_available(model, config):
    """
    Loads a checkpoint into the model if specified in the configuration.

    Args:
        model (nn.Module): The model to initialize or load weights into.
        config (dict): Configuration dictionary that may contain checkpoint information.

    Returns:
        nn.Module: Model with potentially loaded weights from a checkpoint.
    """
    init_checkpoint = config.get("init_checkpoint")

    if init_checkpoint:
        ckpt_file = init_checkpoint.get("file")
        if not ckpt_file:
            raise ValueError(
                "Checkpoint file path is missing in the config['init_checkpoint']['file']"
            )

        print(f"Initializing weights from checkpoint: {ckpt_file}")

        # Get optional filter and strict loading settings
        vars_filter = init_checkpoint.get("filter_key", [])
        strict = init_checkpoint.get("strict", True)

        try:
            model = load_safetensors(model, ckpt_file, vars_filter, strict)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_file}: {e}")

    return model


def create_transform(aug_config):
    transform_list = []
    for aug in aug_config:
        aug_class = getattr(monai.transforms, aug["name"], None)
        if aug_class:
            aug_params = {k: v for k, v in aug.items() if k != "name"}
            transform_list.append(aug_class(**aug_params))
        else:
            print(
                f"Warning: skipping {aug['name']} as it is not a valid MONAI transform"
            )
    aug_transform = monai.transforms.Compose(transform_list)
    return aug_transform


class ConfigFileArgs(Tap):
    config_file: str


def dict_to_defaultdict(d, default=None):
    """
    Recursively converts a dictionary into a defaultdict with a default value of `None`.
    """
    if isinstance(d, dict):
        return defaultdict(
            lambda: default, {k: dict_to_defaultdict(v, default) for k, v in d.items()}
        )
    return d


def defaultdict_to_dict(d):
    """Convert a defaultdict to a dict recursively."""
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
