import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim

from utils.setup_utils import get_model_config
from utils.plotting_functions_utils import plot_averaged_results
from utils.plotting_functions_utils import plot_average_roc_curve, plot_average_pr_curve

from models.KRAG_model import KRAG_Classifier
from models.MUSTANG_model import MUSTANG_Classifier
from models.patchGCN_model import PatchGCN
from models.DeepGraphConv_model import DeepGraphConv
from models.GTP_model import GTP_Classifier
from models.TransMIL_model import TransMIL
from models.CLAM_model import GatedAttention as CLAM


def process_model_output(args, output, loss_fn):
    if args.model_name == 'KRAG':
        logits, Y_prob, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'MUSTANG':
        logits, Y_prob, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'CLAM':
        bag_weight = 0.7
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        instance_loss = results_dir['instance_loss']
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss
        return logits, Y_prob, Y_hat, total_loss

    elif args.model_name == 'PatchGCN':
        logits, Y_prob, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'DeepGraphConv':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'TransMIL':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'GTP':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        total_loss = loss + results_dir['mc1'] + results_dir['o1']
        return logits, Y_prob, Y_hat, total_loss

    elif args.model_name == 'HEAT':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    elif args.model_name == 'CAMIL':
        logits, Y_prob, results_dir, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    else:

        raise ValueError(f"Unsupported model: {args.model_name}")

def load_data(args, results_dir):
    config = get_model_config(args)
    run_settings = (
        f"{args.model_name}_{config['graph_mode']}_{config['convolution']}_PE_{config['encoding_size']}"
        f"_{args.embedding_net}_{args.dataset_name}_{args.seed}_{config['heads']}_{config['pooling_ratio']}"
        f"_{args.learning_rate}_{args.scheduler}_{args.stain_type}_L1_{args.l1_norm}")

    checkpoints = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints, exist_ok = True)

    graph_dict_path = args.directory + f"/dictionaries/{config['graph_mode']}_dict_{args.dataset_name}"

    if config['encoding_size'] > 0 and config['graph_mode'] == 'krag':
        graph_dict_path += f"_positional_encoding_{config['encoding_size']}"

    graph_dict_path += f"_{args.embedding_net}_{args.stain_type}.pkl"

    with open(graph_dict_path, "rb") as file:
        graph_dict = pickle.load(file)

    # load stratified random split train/test folds
    with open(args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as splits:
        sss_folds = pickle.load(splits)

    return run_settings, checkpoints, graph_dict, sss_folds


def create_cross_validation_splits(args, patient_id, label, test_size=0.2, n_splits=5,
                                   seed=42, dataset_name="dataset", directory=".", hard_test_set=False):
    """
    Create a n-fold cross-validation split with held-out test set.
    """
    patient_labels = pd.read_csv(os.path.join(args.directory, "patient_labels.csv"))
    extracted_patches = pd.read_csv(os.path.join(args.directory, f"extracted_patches_{args.slide_level}", "extracted_patches.csv"))

    # Merge patches with patient labels
    df = pd.merge(extracted_patches, patient_labels, on=patient_id)

    # Drop duplicates to obtain unique patient IDs
    df_labels = df.drop_duplicates(subset=patient_id).reset_index(drop=True)

    if dataset_name.upper() == "CAMELYON16":
        # For CAMELYON16, use the split column to create test set
        test_data = df_labels[df_labels['split'] == 'test']
        train_val_data = df_labels[df_labels['split'] == 'train']
    elif hard_test_set:
        # For hard test set, use the Hard column to create test set
        test_data = df_labels[df_labels['Hard']]
        train_val_data = df_labels[~df_labels['Hard']]
    else:
        # For other datasets, create a held-out test set
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_index, test_index = next(sss_test.split(df_labels[patient_id], df_labels[label]))
        train_val_data = df_labels.iloc[train_val_index]
        test_data = df_labels.iloc[test_index]

    # Create 5-fold cross-validation splits on the training/validation data
    sss_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=seed)

    fold_dictionary = {}

    for i, (train_index, val_index) in enumerate(sss_cv.split(train_val_data[patient_id], train_val_data[label])):
        fold_name = f"Fold {i}"
        fold_dictionary[fold_name] = {
            "Train": list(train_val_data.iloc[train_index][patient_id]),
            "Val": list(train_val_data.iloc[val_index][patient_id]),
            "Test": list(test_data[patient_id])
        }

        # Verify no overlap
        train_set = set(fold_dictionary[fold_name]["Train"])
        val_set = set(fold_dictionary[fold_name]["Val"])
        test_set = set(fold_dictionary[fold_name]["Test"])
        assert len(train_set.intersection(val_set)) == 0, "Train and Val sets overlap"
        assert len(train_set.intersection(test_set)) == 0, "Train and Test sets overlap"
        assert len(val_set.intersection(test_set)) == 0, "Val and Test sets overlap"

    # Save the fold dictionary
    output_path = os.path.join(directory, f"train_test_strat_splits_{dataset_name}.pkl")
    with open(output_path, "wb") as file:
        pickle.dump(fold_dictionary, file)

    print(f"Cross-validation splits saved to {output_path}")

def create_stratified_splits(args, patient_id, label, train_fraction, val_fraction, splits, seed, dataset_name, directory):

    """
    Create a n-fold train/val/test split with stratified sampling on patient labels. N held-out test sets are created.
    """

    patient_labels = pd.read_csv(os.path.join(args.directory, "patient_labels.csv"))
    extracted_patches = pd.read_csv(os.path.join(args.directory, f"extracted_patches_{args.slide_level}", "extracted_patches.csv"))

    # Merge patches with patient labels
    df = pd.merge(extracted_patches, patient_labels, on=patient_id)

    # Drop duplicates to obtain unique patient IDs
    df_labels = df.drop_duplicates(subset=patient_id).reset_index(drop=True)

    # stratified split on labels
    sss = StratifiedShuffleSplit(n_splits= splits, test_size= 1 - train_fraction, random_state=seed)

    # creating a dictionary which keeps a list of the Patient IDs from the stratified training splits. Outer key is Fold, inner key is Train/Val/Test.
    fold_dictionary = {}

    for i, (train_val_index, test_index) in enumerate(sss.split(df_labels[patient_id], df_labels[label])):

        train_val_data = df_labels.iloc[train_val_index]
        val_split = StratifiedShuffleSplit(n_splits=1, test_size= val_fraction, random_state=seed)
        train_index, val_index = next(val_split.split(train_val_data[patient_id], train_val_data[label]))

        fold_name = f"Fold {i}"
        fold_dictionary[fold_name] = {
            "Train": list(train_val_data.iloc[train_index][patient_id]),
            "Val": list(train_val_data.iloc[val_index][patient_id]),
            "Test": list(df_labels.iloc[test_index][patient_id])
        }

        # Verify no overlap
        train_set = set(fold_dictionary[fold_name]["Train"])
        val_set = set(fold_dictionary[fold_name]["Val"])
        test_set = set(fold_dictionary[fold_name]["Test"])
        assert len(train_set.intersection(val_set)) == 0, "Train and Val sets overlap"
        assert len(train_set.intersection(test_set)) == 0, "Train and Test sets overlap"
        assert len(val_set.intersection(test_set)) == 0, "Val and Test sets overlap"

    with open(directory + f"/train_test_strat_splits_{dataset_name}.pkl", "wb") as file:
        pickle.dump(fold_dictionary, file)  # encode dict into Pickle
        print(f"Stratified splits saved to {directory}/train_test_strat_splits_{dataset_name}.pkl")


def prepare_data_loaders(data_dict, sss_folds):
    training_folds = []
    validation_folds = []
    testing_folds = []

    for fold, splits in sss_folds.items():
        train_dict = {k: data_dict[k] for k in splits['Train']}
        val_dict = {k: data_dict[k] for k in splits['Val']}
        test_dict = {k: data_dict[k] for k in splits['Test']}
        training_folds.append(train_dict)
        validation_folds.append(val_dict)
        testing_folds.append(test_dict)

    return training_folds, validation_folds, testing_folds

def initialise_model(args):
    if args.model_name == 'KRAG':
        model = KRAG_Classifier(in_features=args.embedding_vector_size,
                                hidden_dim=args.hidden_dim,
                                num_classes=args.n_classes,
                                heads=args.heads,
                                pooling_ratio=args.pooling_ratio,
                                walk_length=args.encoding_size,
                                conv_type=args.convolution,
                                num_layers=args.num_layers)
    elif args.model_name == 'MUSTANG':
        model = MUSTANG_Classifier(args.embedding_vector_size,
                                   edge_attr_dim=len(args.edge_types),
                                   node_attr_dim=len(args.stain_types),
                                   hidden_dim=args.hidden_dim,
                                   num_classes=args.n_classes,
                                   heads=args.heads,
                                   pooling_ratio=args.pooling_ratio,
                                   walk_length=args.encoding_size,
                                   conv_type=args.convolution,
                                   num_layers=args.num_layers,
                                   embedding_dim=10)
    elif args.model_name == 'CLAM':
        model = CLAM(args.embedding_vector_size)
    elif args.model_name == 'DeepGraphConv':
        model = DeepGraphConv(num_features=args.embedding_vector_size,
                              hidden_dim=args.hidden_dim,
                              n_classes=args.n_classes)
    elif args.model_name == 'PatchGCN':
        model = PatchGCN(num_features=args.embedding_vector_size,
                         hidden_dim=args.hidden_dim,
                         n_classes=args.n_classes)
    elif args.model_name == 'DeepGraphConv':
        model = DeepGraphConv(num_features=args.embedding_vector_size,
                     hidden_dim=args.hidden_dim,
                     n_classes=args.n_classes)
    elif args.model_name == 'GTP':
        model = GTP_Classifier(n_class=args.n_classes,
                               n_features=args.embedding_vector_size)
    elif args.model_name == 'TransMIL':
        model = TransMIL(n_classes=args.n_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()

    return model, loss_fn, optimizer, lr_scheduler

def summarise_train_results(all_results, mean_best_acc, mean_best_AUC, results_dir, run_settings):
    plot_averaged_results(all_results, results_dir + "/")

    average_best_acc = np.mean(mean_best_acc)
    std_best_acc = np.std(mean_best_acc)

    average_best_AUC = np.mean(mean_best_AUC)
    std_best_AUC = np.std(mean_best_AUC)

    summary_df = pd.DataFrame({
        'val_accuracy': [average_best_acc, std_best_acc],
        'val_AUC': [average_best_AUC, std_best_AUC]
    }, index=['mean', 'std']).T

    summary_path = f"{results_dir}/{run_settings}_train_summary_scores.csv"
    summary_df.to_csv(summary_path, index=True)

def summarise_test_results(all_results, results_dir, logger, args):
    accuracies = [r['test_accuracy'] for r in all_results]
    aucs = [r['test_auc'] for r in all_results]

    # Calculate averages and standard deviations
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Create summary dataframe
    summary_df = pd.DataFrame({
        'test_accuracy': [avg_accuracy, std_accuracy],
        'test_AUC': [avg_auc, std_auc]
    }, index=['mean', 'std']).T

    config = get_model_config(args)

    # Save summary to CSV
    run_settings = (
        f"{args.model_name}_{config['graph_mode']}_{config['convolution']}_PE_{config['encoding_size']}"
        f"_{args.embedding_net}_{args.dataset_name}_{args.seed}_{config['heads']}_{config['pooling_ratio']}"
        f"_{args.learning_rate}_{args.scheduler}_{args.stain_type}_L1_{args.l1_norm}")

    summary_path = f"{results_dir}/{run_settings}_test_summary_scores.csv"
    summary_df.to_csv(summary_path, index=True)

    logger.info(f"Average Test Accuracy: {avg_accuracy:.4f} +/- {std_accuracy:.4f}")
    logger.info(f"Average Test AUC: {avg_auc:.4f} +/- {std_auc:.4f}")

    # Plot average curves
    plot_average_roc_curve(all_results, args.n_classes, results_dir)
    plot_average_pr_curve(all_results, args.n_classes, results_dir)


def minority_sampler(train_graph_dict):
    # calculate weights for minority oversampling
    count = []
    for k, v in train_graph_dict.items():
        count.append(v[1].item())
    counter = Counter(count)
    class_count = np.array(list(counter.values()))
    weight = 1 / class_count
    samples_weight = np.array([weight[t] for t in count])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                             num_samples=len(samples_weight), replacement=True)

    return sampler

def l1_regularization(model, l1_norm):
    weights = sum(torch.abs(p).sum() for p in model.parameters())
    return weights * l1_norm

def randomly_shuffle_graph(data, seed=None):
    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Randomly shuffle the node features
    shuffled_features = data.x[torch.randperm(data.num_nodes)]
    shuffled_rw = data.random_walk_pe[torch.randperm(data.num_nodes)]

    # Randomly shuffle the edge index
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    shuffled_edge_index = edge_index[:, torch.randperm(num_edges)]

    # Create a new Data object with the shuffled node features and edge index
    shuffled_data = Data(
        x=shuffled_features,
        edge_index=shuffled_edge_index,
        random_walk_pe=shuffled_rw
    )

    return shuffled_data