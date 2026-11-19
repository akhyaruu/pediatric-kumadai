"""
Preprocessing pipeline for the pediatric appendicitis ultrasound data (Windows and Linux).
Based on notebooks/Preprocess_app_data.ipynb of https://github.com/i6092467/semi-supervised-multiview-cbm (CC BY-NC 4.0).

Pipeline (target: diagnosis):
  1. deepfill : remove GUI markers/annotations with DeepFill (GPU), for ALL images
  2. crop     : crop + zero-pad every image to 400x400
  3. dict     : build the data dictionary {US_Number: [images, label, tabular features, concepts]}
  4. split    : patient-level, label-stratified train/test split (90/10) -> train/test dictionaries
  5. impute   : KNN imputation (fit on train only) + standardisation of real-valued tabular features

Differences to the original notebook (see README_PREPROCESSING.md):
  * DeepFill and cropping are applied to all images once, and the train/test split is done afterwards at the level of
    patients. Both steps are per-image operations, so the result is identical to splitting first, without copying
    2+ GB of raw images.
  * The authors' fixed test-set patient list is not in the repository, so a new split is generated. It is
    stratified by the diagnosis (the notebook used a non-stratified GroupShuffleSplit) and saved to
    test_patient_codes.csv so that every model is evaluated on exactly the same patients.

Usage (from the repository folder):
  python prepare_data.py --raw "C:\\...\\US_Pictures" --xlsx "C:\\...\\pediatric.xlsx" --out "C:\\...\\data_app"
  python prepare_data.py ... --steps dict split impute     # re-run only some steps
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

# Real-valued features that are standardised (same list as in the notebook; only those that are selected in
# datasets/generate_app_data.TABULAR_FEATURES are used). Scores (Alvarado, PAS) and binary features are not scaled.
REAL_VALUED_ALL = [
    "Age",
    "Height",
    "Weight",
    "BMI",
    "Body_Temperature",
    "WBC_Count",
    "Neutrophil_Percentage",
    "CRP",
    "Appendix_Diameter",
]
# Concept set: the first 5 concepts of the paper (was 2: Appendix_on_US, Free_Fluids). The paper uses the first 9 of:
#   ['Appendix_on_US', 'Free_Fluids', 'Appendix_Wall_Layers', 'Target_Sign', 'Surrounding_Tissue_Reaction',
#    'Pathological_Lymph_Nodes', 'Bowel_Wall_Thickening', 'Coprostasis', 'Meteorism', 'Enteritis',
#    'Appendix_Diameter', 'Perforation', 'Appendicular_Abscess', 'Conglomerate_of_Bowel_Loops',
#    'Gynecological_Findings']
# Concepts must also be in TABULAR_FEATURES. The configs select the first num_concepts concepts of this list, so
# configs/mvcbm_seq_lstm_diag.yaml (2 concepts) and configs/mvcbm_seq_lstm_diag_5c.yaml (5 concepts) both work with
# the same dictionaries; keep the order of the first two concepts unchanged.
CONCEPTS = [
    "Appendix_on_US",
    "Free_Fluids",
    "Appendix_Wall_Layers",
    "Target_Sign",
    "Surrounding_Tissue_Reaction",
]
ALL_STEPS = ["deepfill", "crop", "dict", "split", "impute"]
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


DEEPFILL_WEIGHTS = os.path.join(
    "DeepFill", "pretrained_model", "deepfillv2_WGAN_G_epoch40_batchsize4.pth"
)


def step_deepfill(args, d):
    # The pretrained DeepFill weights are part of this repository
    weights = os.path.join(REPO_DIR, DEEPFILL_WEIGHTS)
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"DeepFill weights not found: {weights}")
    from DeepFill.run_preprocessing import deep_fill

    deep_fill(
        args.raw,
        d["df_tmp"],
        d["df_mask"],
        d["deepfilled"],
        os.path.join(REPO_DIR, "DeepFill", "patterns"),
        resume=not args.fresh,
    )


def step_crop(args, d):
    from datasets.preprocessing import preprocess

    preprocess(
        {
            "images": d["deepfilled"],
            "target": d["cropped"],
            "padding": "constant",
            "debug": False,
            "no_black_triangles": False,
        }
    )


def step_dict(args, d):
    from datasets.generate_app_data import generate_files

    generate_files(
        {
            "info_file": args.xlsx,
            "image_dir": d["cropped"],
            "output_dir": d["dicts"],
            "output_file": "app_data_all",
            "blacklist": args.blacklist,
            "label": "Diagnosis",
            "true_label": ["appendicitis"],
            "false_label": ["no appendicitis"],
            "concepts": CONCEPTS,
        }
    )


def step_split(args, d):
    from sklearn.model_selection import train_test_split

    with open(os.path.join(d["dicts"], "app_data_all")) as f:
        data = json.load(f)
    keys = np.array(list(data.keys()))
    labels = np.array([data[k][1] for k in keys])
    codes_file = os.path.join(d["dicts"], "test_patient_codes.csv")
    if args.test_codes:
        test_codes = set(
            str(int(c)) for c in np.atleast_1d(np.genfromtxt(args.test_codes))
        )
        print(f"Using predefined test patients from {args.test_codes}")
    else:
        _, test_keys = train_test_split(
            keys, test_size=args.test_size, stratify=labels, random_state=args.seed
        )
        test_codes = set(test_keys.tolist())
    train = {k: v for k, v in data.items() if k not in test_codes}
    test = {k: v for k, v in data.items() if k in test_codes}
    assert not set(train) & set(test)
    np.savetxt(codes_file, np.array(sorted(int(k) for k in test)), fmt="%d")
    for name, part in [("app_data_train", train), ("app_data_test", test)]:
        with open(os.path.join(d["dicts"], name), "w") as f:
            json.dump(part, f)
        y = np.array([v[1] for v in part.values()])
        n_img = sum(len(v[0]) for v in part.values())
        print(
            f"{name}: {len(part)} patients, {n_img} images, appendicitis = {y.sum()} ({y.mean():.1%})"
        )
    print(f"Test patient codes saved to {codes_file}")


def _decompose(data, features):
    keys = list(data.keys())
    image_names = [data[k][0] for k in keys]
    y = np.array([data[k][1] for k in keys])
    X = pd.DataFrame([data[k][2] for k in keys], columns=features)
    return keys, image_names, X, y


def step_impute(args, d):
    from sklearn.impute import KNNImputer
    from datasets.generate_app_data import TABULAR_FEATURES

    missing = [c for c in CONCEPTS if c not in TABULAR_FEATURES]
    assert (
        not missing
    ), f"Concepts {missing} must also be listed in TABULAR_FEATURES (generate_app_data.py)"
    real_valued = [v for v in REAL_VALUED_ALL if v in TABULAR_FEATURES]
    print(
        f"Tabular features: {TABULAR_FEATURES}\nStandardised: {real_valued}\nConcepts: {CONCEPTS}"
    )
    with open(os.path.join(d["dicts"], "app_data_train")) as f:
        train = json.load(f)
    with open(os.path.join(d["dicts"], "app_data_test")) as f:
        test = json.load(f)
    keys, names, X, y = _decompose(train, TABULAR_FEATURES)
    keys_t, names_t, X_t, y_t = _decompose(test, TABULAR_FEATURES)

    # Same procedure as impute_final() in the notebook: fit on train only
    imputer = KNNImputer(n_neighbors=1)
    X = pd.DataFrame(imputer.fit_transform(X), columns=TABULAR_FEATURES)
    X_t = pd.DataFrame(imputer.transform(X_t), columns=TABULAR_FEATURES)
    X_norm, X_t_norm = copy.deepcopy(X), copy.deepcopy(X_t)
    for var in real_valued:
        mean, std = np.mean(X[var].values), np.std(X[var].values)
        X_norm[var] = (X[var] - mean) / std
        X_t_norm[var] = (X_t[var] - mean) / std

    out_dir = os.path.join(d["dicts"], "imputed", "final")
    os.makedirs(out_dir, exist_ok=True)
    for name, ks, ns, Xn, yy in [
        ("app_data_train", keys, names, X_norm, y),
        ("app_data_test", keys_t, names_t, X_t_norm, y_t),
    ]:
        conc = Xn[CONCEPTS]
        mapping = {
            k: [
                ns[i],
                int(yy[i]),
                Xn.iloc[i].values.tolist(),
                conc.iloc[i].values.tolist(),
            ]
            for i, k in enumerate(ks)
        }
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(mapping, f)
    print(f"Imputed dictionaries written to {out_dir}")
    print("\nUse these paths in the training configs:")
    print(f"  images:         {d['cropped']}")
    print(f"  test_images:    {d['cropped']}")
    print(f"  dict_file:      {os.path.join(out_dir, 'app_data_train')}")
    print(f"  dict_file_test: {os.path.join(out_dir, 'app_data_test')}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw",
        required=True,
        help="folder with the original ultrasound images (US_Pictures)",
    )
    parser.add_argument(
        "--xlsx", required=True, help="Excel file with the tabular data"
    )
    parser.add_argument(
        "--out", required=True, help="output folder for all intermediate and final data"
    )
    parser.add_argument("--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS)
    parser.add_argument(
        "--blacklist",
        default=None,
        help="optional text file with image names to exclude",
    )
    parser.add_argument(
        "--test-codes",
        default=None,
        help="optional CSV with patient codes of a fixed test set",
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="DeepFill: discard results of an earlier run (default: resume and skip finished images)",
    )
    args = parser.parse_args()
    # [repro] resolve user-given paths before changing into the project folder
    for k in ["raw", "xlsx", "out", "blacklist", "test_codes"]:
        if getattr(args, k):
            setattr(args, k, os.path.abspath(getattr(args, k)))

    os.chdir(REPO_DIR)
    d = {
        "df_tmp": os.path.join(args.out, "deepfill_tmp", "images"),
        "df_mask": os.path.join(args.out, "deepfill_tmp", "masks"),
        "deepfilled": os.path.join(args.out, "deepfilled"),
        "cropped": os.path.join(args.out, "preprocessed_400"),
        "dicts": os.path.join(args.out, "dicts_diagnosis"),
    }
    for p in d.values():
        os.makedirs(p, exist_ok=True)

    steps = {
        "deepfill": step_deepfill,
        "crop": step_crop,
        "dict": step_dict,
        "split": step_split,
        "impute": step_impute,
    }
    for name in ALL_STEPS:
        if name in args.steps:
            print(f"\n========== {name} ==========")
            steps[name](args, d)


if __name__ == "__main__":
    main()
