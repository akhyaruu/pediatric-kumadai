"""
Helpers for running the training in a Kaggle notebook (pediatric-kumadai.ipynb).

The preprocessed data (results of prepare_data.py, made locally) are uploaded as a private Kaggle dataset that
contains the folders
    preprocessed_400/                         images, cropped and resized
    dicts_diagnosis/imputed/final/            app_data_train, app_data_test (data dictionaries)
Kaggle mounts the dataset read-only under /kaggle/input while the outputs are written to /kaggle/working.
"""

import os

# Needed for deterministic cuBLAS (LSTM); must be set before CUDA is used (same as in train_mvcbm.py).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import platform
import sys

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

RESNET_FILE = "resnet18-5c106cde.pth"
# Torchvision download of the same file (used only when the weights are not in the repository)
RESNET_URL = "https://download.pytorch.org/models/" + RESNET_FILE

IMAGES_DIR = "preprocessed_400"
DICT_TRAIN = os.path.join("dicts_diagnosis", "imputed", "final", "app_data_train")
DICT_TEST = os.path.join("dicts_diagnosis", "imputed", "final", "app_data_test")


def on_kaggle():
    return os.path.isdir(KAGGLE_INPUT) and os.path.isdir(KAGGLE_WORKING)


def find_data(input_root=KAGGLE_INPUT, max_depth=6):
    """
    Search input_root for the preprocessed data and return a dict with the paths of the images folder and of the
    train/test data dictionaries. The exact location inside the Kaggle dataset does not matter (Kaggle may add a
    folder level, e.g. when a zip file is uploaded).
    """
    found = {"images": [], "dict_file": [], "dict_file_test": []}
    root_depth = input_root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(input_root, followlinks=True):
        if dirpath.count(os.sep) - root_depth >= max_depth:
            dirnames[:] = []
        if os.path.basename(dirpath) == IMAGES_DIR:
            found["images"].append(dirpath)
            dirnames[:] = []  # do not walk the 1,700 images
            continue
        for name in filenames:
            path = os.path.join(dirpath, name)
            if path.endswith(os.sep + DICT_TRAIN):
                found["dict_file"].append(path)
            elif path.endswith(os.sep + DICT_TEST):
                found["dict_file_test"].append(path)

    problems = []
    for key, what in [("images", f"folder '{IMAGES_DIR}'"), ("dict_file", DICT_TRAIN), ("dict_file_test", DICT_TEST)]:
        if len(found[key]) == 0:
            problems.append(f"{what} not found under {input_root}")
        elif len(found[key]) > 1:
            problems.append(f"{what} found more than once: {found[key]}")
    if problems:
        raise FileNotFoundError(
            "\n".join(problems) + "\nAdd the preprocessed dataset to the notebook (Add Input) and check its folders."
        )

    paths = {key: values[0] for key, values in found.items()}
    paths["test_images"] = paths["images"]
    return paths


def ensure_resnet_weights(model_directory=os.path.join(PROJECT_DIR, "pretrained_models")):
    """Make sure the ResNet-18 weights used by the encoder exist, download them if they are missing (needs Internet)."""
    path = os.path.join(model_directory, "resnet", RESNET_FILE)
    if os.path.exists(path):
        print(f"ResNet-18 weights: {path}")
        return path
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"ResNet-18 weights missing, downloading {RESNET_URL}")
    torch.hub.download_url_to_file(RESNET_URL, path)
    return path


def kaggle_overrides(data_paths, run_name, output_root=os.path.join(KAGGLE_WORKING, "runs")):
    """Config overrides: data from the Kaggle dataset, logs/checkpoints in /kaggle/working/runs/<run_name>."""
    overrides = dict(data_paths)
    overrides["log_directory"] = os.path.join(output_root, run_name)
    return overrides


def print_environment():
    import torch

    print(f"python {platform.python_version()} ({sys.executable})")
    print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 2 ** 30:.1f} GiB, compute capability {props.major}.{props.minor}")
    else:
        print("GPU: not available (Kaggle: Settings > Accelerator > GPU)")


def summarize(results_per_seed, metrics=("t_auroc", "t_aupr", "t_balanced_accuracy", "t_f1_macro", "c0_auroc", "c1_auroc")):
    """Table of the test metrics per seed, plus their mean and standard deviation over seeds."""
    import pandas as pd

    rows = {seed: {m: res.get(m) for m in metrics} for seed, res in results_per_seed.items() if res is not None}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "seed"
    if len(table) > 1:
        table.loc["mean"] = table.mean()
        table.loc["std"] = table.iloc[:-1].std(ddof=1)
    return table
