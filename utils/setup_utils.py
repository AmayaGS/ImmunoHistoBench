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
        f"{args.model_name}_{args.embedding_net}_{args.dataset_name}_"
        f"{args.seed}_{args.learning_rate}")

    results_dir = os.path.join(current_directory, "results", run_results_folder)
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging
    log_file_path = os.path.join(results_dir, run_results_folder + log_type + ".log")

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%m',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])

    logger = logging.getLogger('ImmunoHistoBench')

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
    torch.backends.cudnn.benchmark = True


def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
