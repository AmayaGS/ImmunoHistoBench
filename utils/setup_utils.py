# Misc
import os
import logging
import argparse
import ast

import random
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset

from models.ABMIL_model import GatedAttention as ABMIL


MODEL_CONFIGS = {
    'ABMIL': {
        'graph_mode': 'embedding',
        'convolution': 'Linear',
        'encoding_size': 0,
        'heads': 0,
        'pooling_ratio': 0,
        'model_class': ABMIL,
        'use_args': False
    }
}


def get_model_config(args):
    config = MODEL_CONFIGS[args.model_name].copy()
    if config['use_args']:
        config['graph_mode'] = args.graph_mode
        config['convolution'] = args.convolution
        config['encoding_size'] = args.encoding_size
        config['heads'] = args.heads
        config['pooling_ratio'] = args.pooling_ratio
    return config


def setup_results_and_logging(args, log_type):
    current_directory = args.directory
    config = get_model_config(args)

    run_results_folder = (
        f"{args.model_name}_{args.embedding_net}_{args.dataset_name}_"
        f"{args.seed}_{config['heads']}_{args.learning_rate}_{args.scheduler}")

    results_dir = os.path.join(current_directory, "results", run_results_folder)
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging
    log_file_path = os.path.join(results_dir, run_results_folder + log_type + ".log")

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])

    logger = logging.getLogger('KRAG')

    return results_dir, logger


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
