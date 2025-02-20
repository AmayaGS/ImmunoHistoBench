# Misc
import os
import logging
import argparse
import ast
import yaml

import random
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset


def setup_results_and_logging(args, log_type):
    current_directory = args.directory

    run_results_folder = (
        f"{args.model_name}_{args.embedding_net}_{args.dataset_name}_{args.seed}_{args.learning_rate}")

    results_dir = os.path.join(current_directory, "results", run_results_folder)
    os.makedirs(results_dir, exist_ok=True)

    log_file_path = results_dir + "/" + f"{run_results_folder}_{log_type}.log"

    # Create a new logger with a unique name
    logger = logging.getLogger(f'MUSTANG_{run_results_folder}_{log_type}')

    # Reset handlers to avoid duplicate logging
    if logger.handlers:
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M')

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return results_dir, logger


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_dict(string):
    try:
        return ast.literal_eval(string)
    except:
        raise argparse.ArgumentTypeError("Invalid dictionary format")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
