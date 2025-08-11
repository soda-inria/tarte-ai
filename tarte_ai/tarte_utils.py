"""Common functions used for TARTE package."""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download


def reshape_pred_output(y_pred):
    """Reshape the predictive output accordingly."""

    num_pred = len(y_pred)
    if y_pred.shape == (num_pred, 2):
        y_pred = y_pred[:, 1]
    elif y_pred.shape == (num_pred, 1):
        y_pred = y_pred.ravel()
    else:
        pass
    return y_pred


def load_tarte_pretrain_model(device='cpu'):
    """Function to load the TARTE pretrained weights."""

    import torch

    # Get the base path relative to this file's location
    base_path = Path(__file__).resolve().parent.parent

    # Obtain the weights. It downloads from huggingface if not downloaded.
    cache_dir = str(base_path / "data/pretrained_weights")
    repo_id = 'inria-soda/tarte'
    filename = 'tarte_pretrained_weights.pt'
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    pretrain_weights = torch.load(model_path, map_location=device, weights_only=True)

    # Obtain the pretrain model configurations
    filename = 'tarte_pretrained_configs.json'
    config_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    filename = open(config_path)
    pretrain_model_configs = json.load(filename)
    filename.close()

    return pretrain_weights, pretrain_model_configs


def load_fasttext():
    """Function to load the TARTE pretrained weights."""

    import fasttext

    # Get the base path relative to this file's location
    base_path = Path(__file__).resolve().parent.parent

    # Obtain the fasttext model. It downloads from huggingface if not downloaded.
    cache_dir = str(base_path / "data/fasttext")
    repo_id = 'hi-paris/fastText'
    filename = 'cc.en.300.bin'
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    return fasttext.load_model(model_path)


def load_data(data_name):
    """Function to load datasets in carte-benchmark."""

    import pandas as pd

    # Basic settings
    base_path = Path(__file__).resolve().parent.parent
    cache_dir = str(base_path / f"data/data_carte/{data_name}")
    repo_id = 'inria-soda/carte-benchmark'

    # Dataset
    filename = f'data_carte/{data_name}/raw.parquet'
    data_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, repo_type="dataset")
    # Dataset configs
    filename = f'data_carte/{data_name}/config_data.json'
    config_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, repo_type="dataset")
    filename = open(config_path)
    config_data = json.load(filename)
    filename.close()

    return pd.read_parquet(data_path), config_data 
