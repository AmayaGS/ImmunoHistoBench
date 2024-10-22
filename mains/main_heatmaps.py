# -*- coding: utf-8 -*-

# Misc
import os.path
import pickle

# PyTorch
import torch

from utils.model_utils import load_data, prepare_data_loaders
from utils.heatmap_utils import generate_heatmap

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()


def heatmap_generation(args, results_dir, logger):
    _, _, _, sss_folds = load_data(args, results_dir)
    heatmap_dir = os.path.join(results_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    try:
        os.path.exists(args.path_to_patches)
    except FileNotFoundError:
        logger.error(f"Path to patches not found. Check the path to the extracted patches. Exiting.")
        exit(1)

    for fold_idx, fold_name in enumerate(sss_folds):
        try:
            with open(f"{results_dir}/attention_scores_fold_{fold_idx}.pkl", 'rb') as test_results:
                attention_scores_dict = pickle.load(test_results)
            generate_heatmap(args, attention_scores_dict, heatmap_dir, fold_idx, args.patch_size, args.path_to_patches, sigma=10)
        except FileNotFoundError:
            logger.error(f"Attention scores not found for fold {fold_idx}. Check you've generated them. Skipping.")
            continue
