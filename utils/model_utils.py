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
# from utils.plotting_functions_utils import plot_averaged_results
from utils.plotting_functions_utils import plot_average_roc_curve, plot_average_pr_curve

from models.ABMIL_model import GatedAttention as ABMIL


def process_model_output(args, output, loss_fn):
    if args.model_name == 'ABMIL':
        bag_weight = 0.7
        logits, Y_prob, label = output
        Y_hat = torch.argmax(Y_prob, dim=1)
        loss = loss_fn(logits, label)
        return logits, Y_prob, Y_hat, loss

    else:

        raise ValueError(f"Unsupported model: {args.model_name}")

def load_data(args, results_dir):
    config = get_model_config(args)
    run_settings = (
        f"{args.model_name}_{args.embedding_net}_{args.dataset_name}_"
        f"{args.seed}_{config['heads']}_{args.learning_rate}_{args.scheduler}")

    checkpoints = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints, exist_ok = True)

    graph_dict_path = args.directory + f"/dictionaries/{config['graph_mode']}_dict_{args.dataset_name}"

    graph_dict_path += f"_{args.embedding_net}_{args.stain_type}.pkl"

    with open(graph_dict_path, "rb") as file:
        graph_dict = pickle.load(file)

    # load stratified random split train/test folds
    with open(args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as splits:
        sss_folds = pickle.load(splits)

    return run_settings, checkpoints, graph_dict, sss_folds


def create_cross_validation_splits(args, patient_id, label, test_size=0.2, seed=42, dataset_name="dataset",
                                   directory="."):
    """
    Create a n-fold cross-validation split with held-out test set.
    :param n_splits:
    """
    patient_labels = pd.read_csv(os.path.join(args.directory, "patient_labels.csv"))
    extracted_patches = pd.read_csv(os.path.join(args.directory, f"extracted_patches_{args.slide_level}", "extracted_patches.csv"))

    # Merge patches with patient labels
    df = pd.merge(extracted_patches, patient_labels, on=patient_id)

    # Drop duplicates to obtain unique patient IDs
    df_labels = df.drop_duplicates(subset=patient_id).reset_index(drop=True)

    # For other datasets, create a held-out test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_index, test_index = next(sss_test.split(df_labels[patient_id], df_labels[label]))
    train_val_data = df_labels.iloc[train_val_index]
    test_data = df_labels.iloc[test_index]

    # Create 5-fold cross-validation splits on the training/validation data
    sss_cv = StratifiedShuffleSplit(n_splits=args.stratified_splits, test_size=args.val_fraction, random_state=seed)

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
    if args.model_name == 'ABMIL':
        model = ABMIL(M=args.embedding_vector_size)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.L2_norm)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()

    return model, loss_fn, optimizer, lr_scheduler

def summarise_train_results(all_results, mean_best_acc, mean_best_AUC, results_dir, run_settings):
    #plot_averaged_results(all_results, results_dir + "/")

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
    precisions = [r['test_avg_precision'] for r in all_results]
    f1_scores = [r['test_f1'] for r in all_results]
    recalls = [r['test_recall'] for r in all_results]
    macro_precisions = [r['test_precision'] for r in all_results]

    # Calculate averages and standard errors
    avg_accuracy = np.mean(accuracies)
    sem_accuracy = np.std(accuracies) / np.sqrt(np.size(accuracies))
    avg_auc = np.mean(aucs)
    sem_auc = np.std(aucs) / np.sqrt(np.size(aucs))
    avg_ap = np.mean(precisions)
    sem_ap = np.std(precisions) / np.sqrt(np.size(precisions))
    avg_f1 = np.mean(f1_scores)
    sem_f1 = np.std(f1_scores) / np.sqrt(np.size(f1_scores))
    avg_recall = np.mean(recalls)
    sem_recall = np.std(recalls) / np.sqrt(np.size(recalls))
    avg_macro_precision = np.mean(macro_precisions)
    sem_macro_precision = np.std(macro_precisions) / np.sqrt(np.size(macro_precisions))
    #
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'test_accuracy': [avg_accuracy, sem_accuracy],
        'test_AUC': [avg_auc, sem_auc],
        'test_AP': [avg_ap, sem_ap],
        'test_F1': [avg_f1, sem_f1],
        'test_recall': [avg_recall, sem_recall],
        'test_precision': [avg_macro_precision, sem_macro_precision]
    }, index=['mean', 'SE']).T

    config = get_model_config(args)

    # Save summary to CSV
    run_settings = (
        f"{args.model_name}_{args.embedding_net}_{args.dataset_name}_"
        f"{args.seed}_{config['heads']}_{args.learning_rate}_{args.scheduler}")

    summary_path = f"{results_dir}/{run_settings}_test_summary_scores.csv"
    summary_df.to_csv(summary_path, index=True)

    # Log results
    logger.info(f"Average Test Accuracy: {avg_accuracy:.4f} +/- {sem_accuracy:.4f}")
    logger.info(f"Average Test AUC: {avg_auc:.4f} +/- {sem_auc:.4f}")
    logger.info(f"Average AP: {avg_ap:.4f} +/- {sem_ap:.4f}")
    logger.info(f"Average F1 Score: {avg_f1:.4f} +/- {sem_f1:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f} +/- {sem_recall:.4f}")
    logger.info(f"Average Precision: {avg_macro_precision:.4f} +/- {sem_macro_precision:.4f}")
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Plot average confusion matrix
    avg_cm = np.mean([r['confusion_matrix'] for r in all_results], axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm / np.sum(avg_cm, axis=1)[:, None], annot=True, fmt='.2f', cmap='Blues')
    plt.title('Average Normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{results_dir}/{run_settings}_confusion_matrix.png")
    plt.close()

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
