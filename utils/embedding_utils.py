# Misc
import os
import numpy as np
import random

# PyTorch
import torch
from utils.profiling_utils import embedding_profiler

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
use_gpu = torch.cuda.is_available()


def create_embedding(embedding_net, loader, stain_types):

    embedding_dict = dict()

    embedding_net.eval()
    with torch.no_grad():
        for patient_ID, slide_loader in loader.items():
            patient_embedding = []
            patient_ids = []
            folder_ids = []
            filenames = []
            coordinates = []
            stains = []

            for patch in slide_loader:
                embedding_profiler.update_peak_memory()
                inputs, label, patient_id, folder_id, file_name, coordinate, stain = patch
                stain = stain[0]

                stains.append(stain_types[stain])

                label = label[0].unsqueeze(0)
                patient_ID = patient_id[0]
                folder_ID = folder_id[0]
                coordinate = coordinate[0]

                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()

                embedding = embedding_net(inputs)
                embedding = embedding.to('cpu')
                embedding = embedding.squeeze(0).squeeze(0)

                patient_embedding.append(embedding)
                patient_ids.append(patient_ID)
                folder_ids.append(folder_ID)
                filenames.append(file_name)
                coordinates.append(coordinate)

            patient_embedding = torch.stack(patient_embedding)

            # Create a dictionary to store patch metadata
            patch_metadata = {
                'folder_ids': list(np.unique(folder_ids)),
                'filenames': filenames,
                'coordinates': coordinates,
                'stains': stains
            }

            # Embedding dictionary
            embedding_dict[patient_ID] = [patient_embedding.to('cpu'), label.to('cpu'), patch_metadata]

    return embedding_dict