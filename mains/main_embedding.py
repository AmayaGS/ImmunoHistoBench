# Misc
import os
import pandas as pd
import pickle
import time

import torch
from torchvision import transforms

from utils.setup_utils import seed_everything, collate_fn_none
from utils.profiling_utils import embedding_profiler
from utils.embedding_utils import create_embedding
from utils.dataloaders_utils import Loaders

from models.embedding_models import VGG_embedding, resnet18_embedding, resnet50_embedding, convNext_embedding, ViT_embedding
from models.embedding_models import ssl_resnet18_embedding, ssl_resnet50_embedding, CTransPath_embedding, Lunit_embedding
from models.embedding_models import GigaPath_embedding, UNI_embedding, BiOptimus_embedding, Phikon_embedding

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check for GPU availability
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = "cuda"


def patch_embedding(args, logger):

    # Set seed
    seed_everything(args.seed)

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

    if args.embedding_net == 'vgg16':
        # Load weights for vgg16
        embedding_net = VGG_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'resnet18':
        # Load weights for resnet18
        embedding_net = resnet18_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'resnet50':
        # Load weights for resnet 50
        embedding_net = resnet50_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'convnext':
        # Load weights for convnext
        embedding_net = convNext_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'ViT':
        # Load weights for Vision Transformer
        embedding_net = ViT_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'ssl_resnet18':
        # Load weights for pretrained resnet18
        weight_path = os.path.join(args.embedding_weights, "Ciga", "tenpercent_resnet18.pt")
        embedding_net = ssl_resnet18_embedding(weight_path, embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'ssl_resnet50':
        # Load weights for resnet 50
        embedding_net = ssl_resnet50_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'CTransPath':
        # Load weights for CTransPath
        weight_path = os.path.join(args.embedding_weights, "CTransPath", "ctranspath.pth")
        embedding_net = CTransPath_embedding(weight_path, embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'Lunit':
        # Load weights for Lunit
        embedding_net = Lunit_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'GigaPath':
        # Load weights for GigaPath
        embedding_net = GigaPath_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'Phikon':
        # Load weights for Phikon
        embedding_net = Phikon_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'BiOptimus':
        # Load weights for BioOptimus
        embedding_net = BiOptimus_embedding(embedding_vector_size=args.embedding_vector_size)
    elif args.embedding_net == 'UNI':
        # Load weights for UNI
        embedding_net = UNI_embedding(embedding_vector_size=args.embedding_vector_size)

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

    with open(dictionaries + f"/embedding_dict_{args.dataset_name}_{args.embedding_net}_{args.stain_type}.pkl", "wb") as file:
        pickle.dump(embedding_dict, file)  # encode dict into Pickle
        logger.info("Done writing embedding_dict into pickle file")

    logger.info("Embedding done in {:.0f}m {:.0f}s"
                .format(embedding_time // 60, embedding_time % 60))
    logger.info("Embedding Profiling Results:")
    embedding_profiler.report(is_training=False, is_testing=False)