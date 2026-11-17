"""
Training of the multiview concept bottleneck model (MVCBM) on the pediatric appendicitis data.

Copied from train.py of pediatric-mvcbm-ssmvcbm (itself based on https://github.com/i6092467/semi-supervised-multiview-cbm)
and reduced to the MVCBM on the 'app' dataset. The training code is unchanged; changes are marked [kumadai]:
  - only the 'app' dataset and the MVCBM / CBM models are kept (no MAwA, synthetic data or SSMVCBM),
  - print-outs go to the log file AND to the console/notebook (config['log_to_console'], default True),
  - a function run() that can be called from a notebook, with optional config overrides,
  - train_mvcbm() returns the test metrics of the last epoch.

Command line:
    python train_mvcbm.py --config configs/mvcbm_seq_lstm_diag.yaml --seed 42

Notebook:
    from train_mvcbm import run
    results = run("configs/mvcbm_seq_lstm_diag.yaml", seed=42, overrides={"log_directory": "/kaggle/working/runs"})
"""

import argparse
import copy
import os

# [repro] Needed for deterministic cuBLAS (LSTM); must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import sys
from collections import Counter
from os.path import join
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from datasets.app_dataset import AppendicitisDataset
from loss import calc_concept_sample_weights, calc_concept_weights, create_loss
from networks import create_model
from utils.logging import save_data_split
from utils.metrics import PRC, ROC
from utils.printing import print_epoch_val_scores
from validate import validate_epoch_mvcbm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------------------------------------------------
# [kumadai] Logging to file and console
# ---------------------------------------------------------------------------------------------------------------------
class _Tee:
    """File-like object that writes to a log file and, optionally, to a second stream (console/notebook)."""

    def __init__(self, path, console=None):
        self.file = open(path, "w", encoding="utf-8")
        self.console = console

    def write(self, text):
        self.file.write(text)
        if self.console is not None:
            self.console.write(text)
        return len(text)

    def flush(self):
        self.file.flush()
        if self.console is not None:
            self.console.flush()

    def close(self):
        self.file.close()


def _open_log(config, console):
    path = os.path.join(
        config["log_directory"],
        config["run_name"] + "_" + config["experiment_name"] + "_" + str(config["seed"]) + ".log",
    )
    return _Tee(path, console if config.get("log_to_console", True) else None)


# ---------------------------------------------------------------------------------------------------------------------
# Training code copied from train.py (tabs converted to spaces)
# ---------------------------------------------------------------------------------------------------------------------
_SCALER = None


def create_optimizer(config, model, mode):
    """
    Parse the configuration file and return a relevant optimizer object
    """
    assert config["optimizer"] in ["sgd", "adam"], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': config[mode + "_learning_rate"],
         'weight_decay': config['weight_decay']}
    ]

    if config["optimizer"] == "sgd":
        return torch.optim.SGD(optim_params)
    elif config["optimizer"] == "adam":
        return torch.optim.Adam(optim_params)


def _amp_enabled(config):
    return bool(config.get("amp", False)) and str(config["device"]).startswith("cuda")


def _autocast(config):
    """Mixed-precision context for the forward pass (no-op unless config['amp'] is True)."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=_amp_enabled(config))


def _init_scaler(config):
    global _SCALER
    enabled = _amp_enabled(config)
    try:
        _SCALER = torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        _SCALER = torch.cuda.amp.GradScaler(enabled=enabled)


def _backward_step(loss, optimizer):
    """Backward pass + optimiser step, with loss scaling when mixed precision is enabled."""
    if _SCALER is not None and _SCALER.is_enabled():
        _SCALER.scale(loss).backward()
        _SCALER.step(optimizer)
        _SCALER.update()
    else:
        loss.backward()
        optimizer.step()


def _dataset_labels(dataset):
    """Target labels of all data points, read without loading images when possible."""
    if isinstance(getattr(dataset, "labels", None), dict):
        return [v[1] for v in dataset.labels.values()]
    return [x["label"].item() for x in iter(DataLoader(dataset))]


def _labels_and_concepts(dataset, ids, num_concepts, loader=None):
    """
    Target labels and per-concept value lists for the given indices. For the appendicitis dataset the values are read
    directly from the data dictionary (identical values, but no images are loaded); otherwise falls back to iterating
    over a data loader as in the original code.
    """
    if isinstance(getattr(dataset, "labels", None), dict) and hasattr(dataset, "concept_ids"):
        vals = list(dataset.labels.values())
        y = [float(vals[i][1]) for i in ids]
        cs = [np.array(vals[i][3], dtype=np.float32)[dataset.concept_ids] for i in ids]
        all_c = [[float(c[k]) for c in cs] for k in range(num_concepts)]
        return y, all_c
    if loader is None:
        loader = DataLoader(torch.utils.data.Subset(dataset, list(ids)), batch_size=8)
    y, all_c = [], [[] for _ in range(num_concepts)]
    for x in iter(loader):
        y.extend(x["label"].tolist())
        concepts = x["concepts"].numpy()
        for k in range(num_concepts):
            all_c[k].extend(concepts[:, k].tolist())
    return y, all_c


def _create_data_loaders(config, gen, trainset, train_ids, validset=None, val_ids=None):
    """
    Construct dataloaders based on the given datasets and config arguments
    """
    train_subsampler = SubsetRandomSampler(train_ids, gen)
    if val_ids is not None:
        val_subsampler = SubsetRandomSampler(val_ids, gen)

    pm = config["device"] == "cuda"
    train_loader = DataLoader(trainset, batch_size=config["train_batch_size"], sampler=train_subsampler,
                              num_workers=config["workers"], pin_memory=pm, generator=gen, drop_last=True)
    if validset is not None and val_ids is not None:
        val_loader = DataLoader(validset, batch_size=config["val_batch_size"], sampler=val_subsampler,
                                num_workers=config["workers"], pin_memory=pm, generator=gen)
    else:
        val_loader = None
    return train_loader, val_loader


def set_bn_to_eval(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True


def _get_data(config):
    """
    Return the train, validation and test sets. [kumadai] Only the pediatric appendicitis data ('app') is supported.
    """
    if config["dataset"] != "app":
        raise NotImplementedError(f"Dataset {config['dataset']!r} is not supported, only 'app'")
    trainset = AppendicitisDataset(config, augmentation=config["augmentation"], visualize=False, train_data=True)
    testset = AppendicitisDataset(config, augmentation=False, visualize=False, train_data=False)
    validset = AppendicitisDataset(config, augmentation=False, visualize=False, train_data=True)
    return trainset, validset, testset


def _train_one_epoch_mvcbm(mode, epoch, config, model, optimizer, loss_fn, train_loader, target_class_weights,
                           concepts_class_weights):
    """
    Train an MVCBM for one epoch
    """
    running_len = 0
    running_target_loss = 0
    running_concepts_loss = [0] * config['num_concepts']
    running_summed_concepts_loss = 0
    running_total_loss = 0
    # Training mode and the number of epochs
    if mode == "j":
        num_epochs = config["j_epochs"]
    elif mode == "c":
        num_epochs = config["c_epochs"]
    elif mode == "t":
        num_epochs = config["t_epochs"]
    else:
        raise ValueError("Training mode unknown!")

    # Decrease the learning rate, if applicable
    if epoch >= 0 and config["decrease_every"] > 0 and (epoch + 1) % config["decrease_every"] == 0:
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] / config["lr_divisor"]

    with tqdm(total=len(train_loader) * config["train_batch_size"], desc=f"Epoch {epoch + 1}/{num_epochs}",
              unit="img", position=0, leave=True) as pbar:
        model.train()
        if config["model"] == "MVCBM" and config["training_mode"] == "sequential" and mode == "t":
            model.apply(set_bn_to_eval)

        for k, batch in enumerate(train_loader):
            # Address the class imbalance for target variable and categorical concepts
            labels_temp = batch["label"].numpy()
            concepts_temp = batch["concepts"].numpy()
            target_sample_weights = [target_class_weights[int(labels_temp[i])] for i in range(len(labels_temp))]
            target_sample_weights = torch.FloatTensor(target_sample_weights).to(config["device"])
            concepts_sample_weights = calc_concept_sample_weights(config, concepts_class_weights, concepts_temp)
            # NOTE: class weights have to be specified per class and per sample
            loss_fn.target_sample_weight = target_sample_weights
            loss_fn.target_class_weight = torch.FloatTensor(target_class_weights).to(config["device"])
            loss_fn.c_weights = concepts_sample_weights

            batch_images, target_true, batch_names = batch["images"].to(config["device"]), batch["label"].float().to(
                config["device"]), batch["file_names"]  # put the data on the device
            concepts_true = batch["concepts"].to(config["device"])
            additional_features = batch["features"].to(config["device"])
            if config['dataset'] == 'app':
                batch_names = np.array(list(map(list, zip(*batch_names))), dtype=object)  # transpose list

            # Mask tensor indicating which views were added for padding
            if config['dataset'] == 'app':
                mask = torch.tensor(batch_names != "padding.bmp").to(config["device"])
            elif config['dataset'] == 'mawa':
                mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])
            elif config['dataset'] == 'synthetic':
                mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])

            # Forward pass
            with _autocast(config):
                concepts_pred, target_pred_probs, target_pred_logits, attn_weights = model(
                    batch_images, mask, additional_features)
            # [repro] losses are computed in float32 (BCE is not autocast-safe)
            concepts_pred = concepts_pred.float()
            target_pred_probs = target_pred_probs.float()
            target_pred_logits = target_pred_logits.float()
            target_pred_probs = target_pred_probs.squeeze(1)
            target_pred_logits = target_pred_logits.squeeze(1)

            # Backward pass depends on the training mode of the model
            optimizer.zero_grad()
            # Compute the loss
            target_loss, concepts_loss, summed_concepts_loss, total_loss = loss_fn(
                concepts_pred, concepts_true, target_pred_probs, target_pred_logits, target_true)

            running_target_loss += target_loss.item() * batch_images.size(0)
            for concept_idx in range(len(concepts_loss)):
                running_concepts_loss[concept_idx] += concepts_loss[concept_idx].item() * batch_images.size(0)
            running_summed_concepts_loss += summed_concepts_loss.item() * batch_images.size(0)
            running_total_loss += total_loss.item() * batch_images.size(0)

            running_len += batch_images.size(0)
            if mode == "j":
                _backward_step(total_loss, optimizer)
            elif mode == "c":
                _backward_step(summed_concepts_loss, optimizer)
            else:
                _backward_step(target_loss, optimizer)

            # Update the progress bar
            pbar.set_postfix(**{"Target loss": running_target_loss / running_len,
                                "Concepts loss": running_summed_concepts_loss / running_len,
                                "Total loss": running_total_loss / running_len, "lr": optimizer.param_groups[0]["lr"]})

            pbar.update(config["train_batch_size"])

    return None


def train_mvcbm_kfold(config, gen):
    """
    Run the k-fold cross-validation for the MVCBM
    """
    # Log the print-outs
    old_stdout = sys.stdout
    log_file = _open_log(config, old_stdout)
    sys.stdout = log_file

    # ---------------------------------
    #       Prepare data
    # ---------------------------------
    kfold = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])

    trainset, validset, testset = _get_data(config=config)

    labels = np.array(_dataset_labels(trainset))  # [repro] no image loading

    # ---------------------------------
    # Create temporary directories for models and data splits
    # ---------------------------------
    checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
    splits_dir = os.path.join(config['log_directory'], 'splits')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    # Numbers of training epochs for different modules
    if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
            config["training_mode"] == "joint":
        c_epochs = config["j_epochs"]
        t_epochs = config["j_epochs"]
    elif (config["model"] == "MVCBM" or config["model"] == "CBM") and \
            config["training_mode"] == "sequential":
        c_epochs = config["c_epochs"]
        t_epochs = config["t_epochs"]
    else:
        c_epochs = None
        t_epochs = config["t_epochs"]

    c_results = np.empty((config["k_folds"], c_epochs, config["num_concepts"], (4 + 1))) if c_epochs is not None else None
    t_results = np.empty((config["k_folds"], t_epochs, (11 + 3)))  # 11 metrics + 3 loss functions
    roc = ROC(config["k_folds"], range(1, t_epochs + 1))
    pr = PRC(config["k_folds"], range(1, t_epochs + 1))

    # Iterate over all folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset, labels)):
        print()
        print("------------------------")
        print(f'FOLD {fold + 1}/{config["k_folds"]}')
        print("------------------------")
        # Instantiate dataloaders
        train_loader, val_loader = _create_data_loaders(config, gen, trainset, train_ids=train_ids,
                                                        validset=validset, val_ids=val_ids)
        test_loader = DataLoader(
            testset, batch_size=config["val_batch_size"], num_workers=config["workers"], generator=gen)
        save_data_split(splits_dir, trainset, train_ids, validset, val_ids)

        # Retrieve labels
        # [repro] read labels/concepts from the data dictionary instead of num_concepts passes over all images
        train_labels, all_c = _labels_and_concepts(trainset, train_ids, config["num_concepts"], train_loader)

        print("Length of training array", len(train_labels))
        print("Train target class distribution: ", Counter(train_labels))
        print("Train concepts class distribution: ")
        for concept_idx in range(len(all_c)):
            print("...", Counter(all_c[concept_idx]))

        val_labels, val_all_c = _labels_and_concepts(validset, val_ids, config["num_concepts"], val_loader)
        print("\nLength of validation array", len(val_labels))
        print("Validation target class distribution: ", Counter(val_labels))
        print("Validation concepts class distribution: ")
        for concept_idx in range(len(val_all_c)):
            print("...", Counter(val_all_c[concept_idx]))

        target_class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
        concepts_class_weights = calc_concept_weights(all_c)

        # Initialize the model and training objects
        model = create_model(config)
        model.to(config["device"])
        loss_fn = create_loss(config)

        # Training the concept prediction model
        if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
                config["training_mode"] == "sequential":
            print("\nStarting concepts training!\n")
            mode = "c"

            for name, child in model.named_children():
                if name.split("_")[0] == "t":
                    child.apply(freeze_module)

            c_optimizer = create_optimizer(config, model, mode)

            for epoch in range(0, c_epochs):

                _train_one_epoch_mvcbm(mode, epoch, config, model, c_optimizer, loss_fn, train_loader,
                                       target_class_weights, concepts_class_weights)
                target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, _, _, _ = \
                    validate_epoch_mvcbm(epoch, config, model, val_loader, loss_fn)

                for concept_idx in range(len(concepts_loss)):
                    c_results[fold, epoch, concept_idx, 0] = concepts_loss[concept_idx]
                    c_results[fold, epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

                print_epoch_val_scores(
                    config, mode, target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics,
                    all_cMetrics, print_all_c=(config['dataset'] == 'app'))

            # Prepare parameters for the target model training
            for name, child in model.named_children():
                if name.split("_")[0] == "t":
                    child.apply(unfreeze_module)
                else:
                    child.apply(freeze_module)

        # Sequential vs. joint optimisation
        if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
                config["training_mode"] == "sequential":
            print("\nStarting target training!\n")
            mode = "t"
            optimizer = create_optimizer(config, model, mode)
        else:
            print("\nStarting joint training!\n")
            mode = "j"
            optimizer = create_optimizer(config, model, mode)

        # Training the target prediction model / performing joint training, depending on the given configuration
        for epoch in range(0, t_epochs):

            if config["model"] != "Dummy":
                _train_one_epoch_mvcbm(mode, epoch, config, model, optimizer, loss_fn,
                                       train_loader, target_class_weights, concepts_class_weights)

            target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, conf_matrix, \
            FP_names, FN_names = validate_epoch_mvcbm(epoch, config, model, val_loader, loss_fn, fold, roc, pr)

            t_results[fold, epoch, 0] = target_loss
            t_results[fold, epoch, 1] = summed_concepts_loss
            t_results[fold, epoch, 2] = total_loss
            t_results[fold, epoch, 3:] = tMetrics.get_tMetrics()

            if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
                    config["training_mode"] == "joint":
                for concept_idx in range(len(concepts_loss)):
                    c_results[fold, epoch, concept_idx, 0] = concepts_loss[concept_idx]
                    c_results[fold, epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

            print_epoch_val_scores(
                config, mode, target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics,
                all_cMetrics, print_all_c=(config['dataset'] == 'app'))

        torch.save(model.state_dict(), join(checkpoint_dir, f"fold{fold}_model.pth"))
        print("\nTraining finished, model saved!", flush=True)
        print("Confusion matrix:")
        print(conf_matrix)

    print()
    print()
    print("------------------------")
    print(f'{config["k_folds"]}-FOLD CROSS VALIDATION COMPLETED')
    print("------------------------")
    print()

    roc.save(Path(config["log_directory"], "ROC_curves"))
    pr.save(Path(config["log_directory"], "PR_curves"))

    # Stop logging print-outs
    sys.stdout = old_stdout
    log_file.close()

    return None


def train_mvcbm(config, gen):
    """
    Train and test an MVCBM model on a single train-test split
    """
    # Log the print-outs
    old_stdout = sys.stdout
    log_file = _open_log(config, old_stdout)
    sys.stdout = log_file

    # ---------------------------------
    #       Prepare data
    # ---------------------------------
    trainset, validset, testset = _get_data(config=config)

    # Retrieve labels
    train_labels = []
    test_labels = []
    all_c = [[] for _ in range(config['num_concepts'])]

    # [repro] read labels/concepts from the data dictionary instead of loading every image
    train_labels, all_c = _labels_and_concepts(trainset, np.arange(len(trainset)), len(all_c))
    test_labels, _ = _labels_and_concepts(testset, np.arange(len(testset)), len(all_c))

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    print("Length of training array", len(train_labels))
    print("Length of test array", len(test_labels))
    print("Train target class distribution: ", Counter(train_labels))
    print("Train concepts class distribution: ")
    for concept_idx in range(len(all_c)):
        print("...", Counter(all_c[concept_idx]))
    print("Test target class distribution: ", Counter(test_labels))
    print()

    # ---------------------------------
    # Create temporary directory for models
    # ---------------------------------
    checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Numbers of training epochs
    if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
            config["training_mode"] == "joint":
        c_epochs = config["j_epochs"]
        t_epochs = config["j_epochs"]
    elif (config["model"] == "MVCBM" or config["model"] == "CBM") and \
            config["training_mode"] == "sequential":
        c_epochs = config["c_epochs"]
        t_epochs = config["t_epochs"]
    else:
        c_epochs = None
        t_epochs = config["t_epochs"]

    c_results = np.empty((c_epochs, config["num_concepts"], (4 + 1))) if c_epochs is not None else None
    t_results = np.empty((t_epochs, (17 + 3)))  # 17 metrics + 3 losses

    # Instantiate dataloaders
    train_loader, _ = _create_data_loaders(config, gen, trainset, train_ids=np.arange(len(train_labels)))
    test_loader = DataLoader(testset, batch_size=config["val_batch_size"], num_workers=config["workers"],
                             generator=gen)

    # Concept and class weights
    target_class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
    concepts_class_weights = calc_concept_weights(all_c)

    # Initialize model and training objects
    model = create_model(config)
    model.to(config["device"])
    loss_fn = create_loss(config)

    print("STARTING FINAL MODEL TRAINING!")
    print()

    # Concept learning
    if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
            config["training_mode"] == "sequential":
        print("\nStarting concepts training!\n")
        mode = "c"

        for name, child in model.named_children():
            if name.split("_")[0] == "t":
                child.apply(freeze_module)

        c_optimizer = create_optimizer(config, model, mode)

        for epoch in range(0, c_epochs):

            if config["model"] != "Dummy":
                _train_one_epoch_mvcbm(mode, epoch, config, model, c_optimizer, loss_fn,
                                       train_loader, target_class_weights, concepts_class_weights)

            # Validate the model (either every epoc or at the very end)
            if config['validate_every_epoch'] or epoch == c_epochs - 1:
                target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, _, _, _ = \
                    validate_epoch_mvcbm(epoch, config, model, test_loader, loss_fn)

                for concept_idx in range(len(concepts_loss)):
                    c_results[epoch, concept_idx, 0] = concepts_loss[concept_idx]
                    c_results[epoch, concept_idx, 1:6] = all_cMetrics[concept_idx].get_cMetrics()

                print_epoch_val_scores(config, mode, target_loss, concepts_loss,
                                       summed_concepts_loss, total_loss, tMetrics, all_cMetrics,
                                       print_all_c=(config['dataset'] == 'app'))

        # Prepare parameters for target training
        for name, child in model.named_children():
            if name.split("_")[0] == "t":
                child.apply(unfreeze_module)
            else:
                child.apply(freeze_module)

    # Sequential vs. joint optimisation
    if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
            config["training_mode"] == "sequential":
        print("\nStarting target training!\n")
        mode = "t"
        optimizer = create_optimizer(config, model, mode)
    else:
        print("\nStarting joint training!\n")
        mode = "j"
        optimizer = create_optimizer(config, model, mode)

    for epoch in range(0, t_epochs):

        _train_one_epoch_mvcbm(
            mode, epoch, config, model, optimizer, loss_fn, train_loader, target_class_weights, concepts_class_weights)

        # Validate the model (either every epoch or at the very end)
        if config['validate_every_epoch'] or epoch == t_epochs - 1:
            target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, conf_matrix, \
            FP_names, FN_names = validate_epoch_mvcbm(epoch, config, model, test_loader, loss_fn)

            t_results[epoch, 0] = target_loss
            t_results[epoch, 1] = summed_concepts_loss
            t_results[epoch, 2] = total_loss
            tmp_metrics = tMetrics.get_tMetrics()

            t_results[epoch, 3:14] = tmp_metrics[:11]
            t_results[epoch, 14:19] = np.array([tmp_metrics[11]['FPR at 0.75'], tmp_metrics[11]['FPR at 0.8'],
                                                tmp_metrics[11]['FPR at 0.9'], tmp_metrics[11]['FPR at 0.95'],
                                                tmp_metrics[11]['FPR at 0.99']])
            t_results[epoch, 19:] = tmp_metrics[12:]

            if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
                    config["training_mode"] == "joint":
                for concept_idx in range(len(concepts_loss)):
                    c_results[epoch, concept_idx, 0] = concepts_loss[concept_idx]
                    c_results[epoch, concept_idx, 1:6] = all_cMetrics[concept_idx].get_cMetrics()

            print_epoch_val_scores(
                config, mode, target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics,
                print_all_c=(config['dataset'] == 'app'))

    torch.save(model.state_dict(), join(checkpoint_dir, "final_model_" + config['run_name'] + '_' +
                                        config['experiment_name'] + ".pth"))
    print("\nTraining finished, model saved!", flush=True)

    print("\nEVALUATION ON THE TEST SET:\n")
    t_metric_names = ["t_loss", "c_loss", "total_loss", "t_ppv", "t_npv", "t_sensitivity",
                      "t_specificity", "t_accuracy", "t_balanced_accuracy", "t_f1_1", "t_f1_0", "t_f1_macro",
                      "t_auroc", "t_aupr"]

    if c_results is not None:
        print(f"Summed concepts loss on last epoch: {c_results[-1, -1, 0]}")

        if config['dataset'] != 'mawa':
            for concept_idx in range(config["num_concepts"]):
                c_metric_names = [f"sc{concept_idx}_loss", f"sc{concept_idx}_accuracy",
                                  f"sc{concept_idx}_f1_macro", f"sc{concept_idx}_auroc", f"sc{concept_idx}_aupr"]
                print(f"(Concept {concept_idx}) Test results on last epoch: ")
                for metric_idx, metric_name in enumerate(c_metric_names):
                    print(f"    {metric_name}: {c_results[-1, concept_idx, metric_idx]}")

        print(f"Averaged concepts test metrics on last epoch:")
        c_metric_names = [f"loss", f"accuracy", f"f1_macro", f"auroc", f"aupr"]
        for metric_idx, metric_name in enumerate(c_metric_names):
            print(f"    {metric_name}: {sum([c_results[-1, i, metric_idx] for i in range(config['num_concepts'])]) / config['num_concepts']}")

    print(f"(Target) Test results on last epoch: ")
    for metric_idx, metric_name in enumerate(t_metric_names):
        print(f"    {metric_name}: {t_results[-1, metric_idx]}")
    print("Confusion matrix:")
    print(conf_matrix)
    print("List of false positive images: ", FP_names)
    print("List of false negative images: ", FN_names)

    # Stop logging print-outs
    sys.stdout = old_stdout
    log_file.close()

    # [kumadai] return the test metrics of the last epoch so that the notebook can collect them
    results = {name: float(t_results[-1, idx]) for idx, name in enumerate(t_metric_names)}
    if c_results is not None:
        for concept_idx in range(config["num_concepts"]):
            for metric_idx, metric_name in enumerate(["loss", "accuracy", "f1_macro", "auroc", "aupr"]):
                results[f"c{concept_idx}_{metric_name}"] = float(c_results[-1, concept_idx, metric_idx])
    results["confusion_matrix"] = np.asarray(conf_matrix).tolist()
    return results


# ---------------------------------------------------------------------------------------------------------------------
# [kumadai] Entry points
# ---------------------------------------------------------------------------------------------------------------------
PATH_KEYS = ["log_directory", "model_directory", "images", "test_images", "dict_file", "dict_file_test"]


def load_config(config_path, seed=None, overrides=None):
    """
    Read a YAML configuration and apply the seed and the overrides (a dict with config keys). Relative paths in the
    config are resolved against the project folder, so the result does not depend on the current working directory.
    """
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        config_path = os.path.join(PROJECT_DIR, config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["filename"] = config_path
    config.update(copy.deepcopy(overrides or {}))
    if seed is not None:
        config["seed"] = int(seed)
    for key in PATH_KEYS:
        if config.get(key) and not os.path.isabs(config[key]):
            config[key] = os.path.join(PROJECT_DIR, config[key])
    if config["model"] not in ["MVCBM", "CBM"]:
        raise ValueError(f"train_mvcbm.py only trains MVCBM/CBM models, the config has {config['model']!r}")
    return config


def check_config(config):
    """Fail early, with a clear message, if a file or folder in the config does not exist."""
    missing = [f"{key}: {config[key]}" for key in ["images", "test_images", "dict_file", "dict_file_test"]
               if not os.path.exists(config[key])]
    weights = os.path.join(config["model_directory"], "resnet", "resnet18-5c106cde.pth")
    if not os.path.exists(weights):
        missing.append(f"ResNet-18 weights: {weights}")
    if missing:
        raise FileNotFoundError("Not found:\n  " + "\n  ".join(missing))
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config['device'] is 'cuda' but no GPU is available (Kaggle: Settings > Accelerator > GPU)")


def run(config_path, seed=None, overrides=None):
    """
    Train (and test) an MVCBM as specified by the config. Returns the config that was used and, for a final training
    run (validate: False), the test metrics of the last epoch; for cross-validation (validate: True), None.
    """
    config = load_config(config_path, seed=seed, overrides=overrides)
    check_config(config)

    # Ensure reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    gen = torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    # [repro] warn instead of crash for CUDA ops without a deterministic implementation (e.g. adaptive pooling backward)
    torch.use_deterministic_algorithms(True, warn_only=True)
    _init_scaler(config)
    os.makedirs(config["log_directory"], exist_ok=True)

    train = train_mvcbm_kfold if config["validate"] else train_mvcbm
    old_stdout = sys.stdout
    try:
        results = train(config, gen)
    finally:
        # restore the console also when training fails (the training functions redirect sys.stdout)
        if sys.stdout is not old_stdout:
            log_file, sys.stdout = sys.stdout, old_stdout
            log_file.close()
    return config, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="overrides the seed in the config")
    args = parser.parse_args()
    run(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
