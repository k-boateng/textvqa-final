import os
import json
import random
import yaml
import numpy as np
import torch


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_answer(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text