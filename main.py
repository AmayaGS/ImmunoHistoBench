import os
import argparse

from mains.main_tissue_segmentation import tissue_segmentation
from mains.main_embedding import patch_embedding
from mains.main_train_test import train_model, test_model
from mains.main_heatmaps import heatmap_generation
from utils.setup_utils import setup_results_and_logging
from utils.model_utils import create_cross_validation_splits
from utils.setup_utils import load_config


def parse_arguments():

    # Step 1: Parse only the config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='RA_config.yaml', help='Path to the config file')
    args, remaining_argv = parser.parse_known_args()

    # Load the config file
    config = load_config(args.config)

    # Step 2: Parse all arguments, using config for defaults
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--input_directory', type=str, default=config['paths']['input_directory'], help='Input data directory')
    parser.add_argument('--directory', type=str, default=config['paths']['output_directory'], help='Location of patient labels/extracted patches. Embeddings dictionaries will be kept here')
    parser.add_argument('--embedding_weights', type=str, default=config['paths']['embedding_weights'], help="Path to embedding weights")
    parser.add_argument('--path_to_patches', type=str, default=config['paths']['path_to_patches'], help="Path to extracted patches")

    # Dataset configurations
    parser.add_argument('--dataset_name', type=str, default=config['dataset']['name'], choices=['RA', 'Sjogren'], help="Dataset name")
    parser.add_argument('--patch_size', type=int, default=config['dataset']['patch_size'], help='Patch size')
    parser.add_argument('--overlap', type=int, default=config['dataset']['overlap'], help='Overlap')
    parser.add_argument('--coverage', type=float, default=config['dataset']['coverage'], help='Coverage')
    parser.add_argument('--slide_level', type=int, default=config['dataset']['slide_level'], help='Slide level')
    parser.add_argument('--mask_level', type=int, default=config['dataset']['mask_level'], help='Mask level')
    parser.add_argument('--patch_batch_size', type=int, default=config['dataset']['patch_batch_size'], help='Batch size for patching')
    parser.add_argument('--train_fraction', type=float, default=config['dataset']['train_fraction'], help="Train fraction")
    parser.add_argument('--val_fraction', type=float, default=config['dataset']['val_fraction'], help="Validation fraction")
    parser.add_argument('--stain_used', type=str, default=config['dataset']['stain_used'], help='Type of stain used.')

    # Parsing configurations
    parser.add_argument('--patient_ID_parsing', type=str, default=config['parsing']['patient_ID'], help='String parsing to obtain patient ID from image filename')
    parser.add_argument('--stain_parsing', type=str, default=config['parsing']['stain'], help='String parsing to obtain stain type from image filename')
    parser.add_argument('--stain_types', type=str, default=config['parsing']['stain_types'], help='Type of stain used.')

    # Label configurations
    parser.add_argument("--label", type=str, default=config['labels']['label'], help="Name of the target label in the metadata file")
    parser.add_argument("--label_dict", type=eval, default=str(config['labels']['label_dict']), help="Dictionary mapping int labels to string labels")
    parser.add_argument("--n_classes", type=int, default=config['labels']['n_classes'], help="Number of classes")
    parser.add_argument("--patient_id", type=str, default=config['labels']['patient_id'], help="Name of column containing the patient ID")

    # Training configurations
    parser.add_argument("--hidden_dim", type=int, default=config['training']['hidden_dim'], help="Size of hidden network dimension")
    parser.add_argument("--learning_rate", type=float, default=config['training']['learning_rate'], help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=config['training']['num_epochs'], help="Number of training epochs")
    parser.add_argument('--L2_norm', type=float, default=config['training']['L2_norm'], help='weight decay')
    parser.add_argument("--batch_size", type=int, default=config['training']['batch_size'], help="batch size for training")
    parser.add_argument("--slide_batch", type=int, default=config['training']['slide_batch'], help="Slide batch size")
    parser.add_argument("--num_workers", type=int, default=config['training']['num_workers'], help="Number of workers for data loading")
    parser.add_argument("--scheduler", type=str, default=config['training']['scheduler'], help="learning rate schedule")
    parser.add_argument("--checkpoint", action="store_true", default=config['training']['checkpoint'], help="Enables checkpointing weights.")
    parser.add_argument('--seed', type=int, default=config['training']['seed'], help="Random seed")
    parser.add_argument("--attention_heads", type=int, default=config['training']['attention_heads'], help="Number of GAT heads")
    parser.add_argument("--stratified_splits", type=int, default=config['training']['stratified_splits'], help="Number of random stratified splits")

    # Model configurations
    parser.add_argument("--model_name", type=str, default=config['model']['name'])

    # Embedding configurations
    parser.add_argument("--embedding_net", type=str, default="resnet50",
                        choices=list(config['embedding_nets'].keys()),
                        help="feature extraction network used")

    # Execution flags
    parser.add_argument("--preprocess", action='store_true', default=config['execution']['preprocess'], help="Run tissue segmentation, patching of WSI, embed feature vectors, graph creation & compute RWPE.")
    parser.add_argument("--segmentation", action='store_true', default=config['execution']['segmentation'], help="Run tissue segmentation of WSI")
    parser.add_argument("--embedding", action='store_true', default=config['execution']['embedding'], help="Run feature vector extraction of the WSI patches and creation of embedding")
    parser.add_argument("--create_splits", action='store_true', default=config['execution']['create_splits'], help="Create train/val/test splits")
    parser.add_argument("--train", action='store_true', default=config['execution']['train'], help="Run ABMIL")
    parser.add_argument("--test", action='store_true', default=config['execution']['test'], help="Run testing")
    parser.add_argument("--visualise", action='store_true', default=config['execution']['visualise'], help="Run heatmap for WSI")

    args = parser.parse_args(remaining_argv)

    # Set embedding_vector_size based on the selected embedding_net
    args.embedding_vector_size = config['embedding_nets'][args.embedding_net]['size']

    return args, config


def main(args, config):

    # Run the preprocessing steps together in one go: tissue segmentation, patching of WSI, embed feature vectors & compute RWPE.
    if args.preprocess:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")

        preprocess_logger.info("Running tissue segmentation of WSIs")
        # Run tissue segmentation and patching of Whole Slide Images
        tissue_segmentation(args, preprocess_logger)
        preprocess_logger.info("Done running tissue segmentation of WSIs")

        preprocess_logger.info("Running feature vector extraction of the WSI patches and creation of embedding")
        patch_embedding(args, config, preprocess_logger)
        preprocess_logger.info("Done running feature vector extraction of the WSI patches and creation of embedding")

        preprocess_logger.info("Creating train/val/test splits")
        sss_dict_path = os.path.join(args.directory, f"train_test_strat_splits_{args.dataset_name}.pkl")
        if not os.path.exists(sss_dict_path):
            create_cross_validation_splits(args, patient_id=args.patient_id, label=args.label,
                                           test_size=1-args.train_fraction, seed=args.seed,
                                           dataset_name=args.dataset_name, directory=args.directory)
        preprocess_logger.info("Done creating train/val/test splits")

    # Run the preprocessing steps individually if needed
    if args.segmentation:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")
        # Run tissue segmentation of WSI
        preprocess_logger.info("Running tissue segmentation of WSIs")
        tissue_segmentation(args, preprocess_logger)
        preprocess_logger.info("Done running tissue segmentation of WSIs")

    if args.embedding:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")
        preprocess_logger.info("Running feature vector extraction of the WSI patches and creation of embedding")
        patch_embedding(args, config, preprocess_logger)
        preprocess_logger.info("Done running feature vector extraction of the WSI patches and creation of embedding")

    if args.create_splits:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")
        preprocess_logger.info("Creating train/val/test splits")
        sss_dict_path = os.path.join(args.directory, f"train_test_strat_splits_{args.dataset_name}.pkl")
        if not os.path.exists(sss_dict_path):
            create_cross_validation_splits(args, patient_id=args.patient_id, label=args.label,
                                           test_size=1 - args.train_fraction, seed=args.seed,
                                           dataset_name=args.dataset_name, directory=args.directory)
        preprocess_logger.info("Done creating train/val/test splits")

    if args.train:
        results_dir, train_logger = setup_results_and_logging(args, "_training")
        train_logger.info("Start training")
        train_model(args, results_dir, train_logger)
        train_logger.info("Done training")

    if args.test:
        results_dir, test_logger = setup_results_and_logging(args, "_testing")
        test_logger.info("Running testing")
        test_model(args, results_dir, test_logger)
        test_logger.info("Done testing")

    if args.visualise:
        results_dir, heatmap_logger = setup_results_and_logging(args, "_heatmaps")
        heatmap_logger.info("Generating heatmaps")
        heatmap_generation(args, results_dir, heatmap_logger)
        heatmap_logger.info("Done generating heatmaps")


if __name__ == "__main__":
    args, config = parse_arguments()
    main(args, config)


