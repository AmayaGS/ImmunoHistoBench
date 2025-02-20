# Misc
import os
import pandas as pd
import pickle
import time
import importlib

import torch
from torchvision import transforms

from utils.setup_utils import collate_fn_none
from utils.profiling_utils import embedding_profiler
from utils.embedding_utils import create_embedding
from utils.dataloaders_utils import Loaders

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check for GPU availability
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = "cuda"



def patch_embedding(args, config, logger):

    # ImageNet transforms - good for UNI, GigaPath, Phikon
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ]
    )

    if args.embedding_net == 'BiOptimus':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.707223, 0.578729, 0.703617],
                                    std=[0.211883, 0.230117, 0.177517])
            ]
        )

    # Load df with patient_id and corresponding labels here, to merge with extracted patches.
    patient_labels = pd.read_csv(args.directory + "/patient_labels.csv")
    # Load file with all extracted patches metadata and locations.
    extracted_patches = pd.read_csv(args.directory + "/extracted_patches_" + str(args.slide_level) + "/extracted_patches.csv")

    df = pd.merge(extracted_patches, patient_labels, on= args.patient_id)
    # Drop duplicates to obtain the actual patient IDs that have a label assigned by the pathologist
    df_labels = df.drop_duplicates(subset= args.patient_id)
    ids = list(df_labels[args.patient_id])

    # Create dictionary with patient ID as key and Dataloaders containing the corresponding patches as values.
    slides = Loaders().slides_dataloader(df=df,
                                         ids=ids,
                                         transform=transform,
                                         slide_batch=args.slide_batch,
                                         num_workers=args.num_workers,
                                         shuffle=False, collate=collate_fn_none,
                                         label=args.label,
                                         patient_id=args.patient_id)

    embedding_profiler.set_logger(logger)
    embedding_profiler.reset_gpu_memory()
    embedding_profiler.update_peak_memory()

    embedding_net = get_embedding_net(args, config)

    if use_gpu:
         embedding_net.cuda()

    embedding_profiler.update_peak_memory()

    logger.info(f"Start creating {args.dataset_name} embeddings for {args.embedding_net}")
    start_time = time.time()
    embedding_dict = create_embedding(embedding_net, slides,stain_types=args.stain_types)
    embedding_time = time.time() - start_time
    logger.info(f"Done creating {args.dataset_name} embeddings for {args.embedding_net}")

    dictionaries = os.path.join(args.directory, "dictionaries")
    os.makedirs(dictionaries, exist_ok = True)

    with open(dictionaries + f"/embedding_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_used}.pkl", "wb") as file:
        pickle.dump(embedding_dict, file)  # encode dict into Pickle
        logger.info("Done writing embedding_dict into pickle file")

    logger.info("Embedding done in {:.0f}m {:.0f}s"
                .format(embedding_time // 60, embedding_time % 60))
    logger.info("Embedding Profiling Results:")
    embedding_profiler.report(is_training=False, is_testing=False)


def get_embedding_net(args, config):
    if args.embedding_net not in config['embedding_nets']:
        raise ValueError(f"Unknown embedding network: {args.embedding_net}")

    net_config = config['embedding_nets'][args.embedding_net]

    embedding_class = get_embedding_class(net_config['class'], config)

    kwargs = {}

    if net_config['weight_path'] and args.embedding_weights:
        if not os.path.exists(args.embedding_weights):
            raise FileNotFoundError(f"Embedding weights directory not found: {args.embedding_weights}")

        weight_path = os.path.join(args.embedding_weights, net_config['weight_path'])
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        kwargs['weight_path'] = weight_path

    try:
        return embedding_class(**kwargs)
    except TypeError as e:
        raise TypeError(f"Error creating {args.embedding_net} embedding: {str(e)}")


def get_embedding_class(class_name, config):
    if class_name not in config['embedding_classes']:
        raise ValueError(f"Embedding class {class_name} is not defined in the config")

    module_path, class_name = config['embedding_classes'][class_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)