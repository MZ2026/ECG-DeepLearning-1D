import ast
import numpy as np
import pandas as pd
import os
import wfdb


# The script/function is designed to prepare and manage the PTB-XL ECG dataset
# for binary ECG classification tasks involving Atrial Fibrillation (AFIB).

# Supported Modes:
#   • Filtered mode (use_filtered=True):  Only NORM and AFIB samples are included.
#       - Labels: NORM = 0, AFIB = 1
#   • Full mode (use_filtered=False):  All ECG classes are included.
#       - Labels: AFIB = 1, all other classes (including NORM) = 0

# Main functionalities:
# 1. Split dataset into stratified folds for training, validation, and testing
# 2. Optionally filter to include only AFIB and NORM samples
# 3. Convert ECG diagnosis codes (scp_codes) to numeric binary labels
# 4. Assign ECG sampling frequency (100Hz or 500Hz)
# 5. Print dataset summaries and class distributions (per split and overall)
# 6. Load ECG records using WFDB (with optional metadata)


# Example usage:
#   train_df, val_df, test_df, freq = setup_dataset(
#       df,
#       train_folds=[1,3],
#       val_folds=[9],
#       test_folds=[10],
#       use_500Hz=False,
#       use_filtered=True
#   )

#   sample_path = train_df.iloc[0]["filename"]
#   signal, info = load_ecg(sample_path, base_dir="ptb-xl-dataset-1.0.3",  return_meta=True)



# Function 1: Select folds for train/val/test with optional binary filtering.
def select_folds(df, train_folds, val_folds, test_folds, filter_norm_afib=False):
    """
    Split PTB-XL dataset into training, validation, and test subsets based on folds.
    PTB-XL comes pre-divided into 10 stratified folds (1–10).
    This function allows flexible selection of folds and optional NORM/AFIB filtering.
    """

    # Ensures all fold variables are lists (e.g., train_folds = [1,2,3]).
    if isinstance(train_folds, int): train_folds = [train_folds]
    if isinstance(val_folds, int):   val_folds = [val_folds]
    if isinstance(test_folds, int):  test_folds = [test_folds]

    # Split DataFrame and selects rows where strat_fold is in the specified folds.
    train_df = df[df["strat_fold"].isin(train_folds)].copy()
    val_df   = df[df["strat_fold"].isin(val_folds)].copy()
    test_df  = df[df["strat_fold"].isin(test_folds)].copy()

    # Filter PTB-XL DataFrame to keep only NORM/AFIB or keep all samples
    def filter_data(df, use_filtered):
        if use_filtered:
            # Keep only NORM and AFIB
            return df[df["scp_codes"].apply(
                lambda s: "NORM" in ast.literal_eval(s) or "AFIB" in ast.literal_eval(s)
            )].copy()
        else:
            # Keep all samples
            return df.copy()
        

    # Convert labels into binary (AFIB=1, NORM=0 eller all others=0)
    def label_to_binary(scp_str: str, use_filtered=True) -> int:
        scp = ast.literal_eval(scp_str)
        if use_filtered:
            if "AFIB" in scp:
                return 1
            if "NORM" in scp:
                return 0
            return np.nan  # exclude all others
        else:
            # AFIB = 1, all others = 0
            return 1 if "AFIB" in scp else 0

    # Apply filtering and labeling
    train_df = filter_data(train_df, filter_norm_afib)
    val_df   = filter_data(val_df, filter_norm_afib)
    test_df  = filter_data(test_df, filter_norm_afib)

    for subset in [train_df, val_df, test_df]:
        subset["label"] = subset["scp_codes"].apply(lambda s: label_to_binary(s, filter_norm_afib))

    


    # Summary
    print("Selected folds:")
    print(f" Train: Fold {train_folds} ({len(train_df)} samples)")
    print(f" Validation: Fold {val_folds} ({len(val_df)} samples)")
    print(f" Test: Fold {test_folds} ({len(test_df)} samples)")
    print(f" Total: {len(train_df) + len(val_df) + len(test_df)} samples\n")

    return train_df, val_df, test_df


# Function 2: Count label/class distribution
def count_labels(df):
    """Count how many NORM, AFIB, and other samples exist."""
    norm_count = 0
    afib_count = 0
    other_count = 0

    for s in df["scp_codes"]:
        scp = ast.literal_eval(s)
        if "NORM" in scp:
            norm_count += 1
        elif "AFIB" in scp:
            afib_count += 1
        else:
            other_count += 1

    return norm_count, afib_count, other_count


# Function 3: Assign ECG signal frequency and filenames
def set_signal_frequency(train_df, val_df, test_df, use_500Hz=False):
    if use_500Hz:
        for subset in [train_df, val_df, test_df]:
            subset["filename"] = subset["filename_hr"]
        sampling_frequency = 500
        print("Selected ECG sampling rate: 500Hz")
    else:
        for subset in [train_df, val_df, test_df]:
            subset["filename"] = subset["filename_lr"]
        sampling_frequency = 100
        print("Selected ECG sampling rate: 100Hz")

    return train_df, val_df, test_df, sampling_frequency


# Function 4: Main wrapper — setup_dataset()
def setup_dataset(df, train_folds, val_folds, test_folds, use_500Hz=False, use_filtered=False):
    """
    Prepares training, validation, and test datasets with optional filtering and sampling rate.
    
    """

    # Select folds (and filter if needed)
    train_df, val_df, test_df = select_folds(
        df, train_folds, val_folds, test_folds, filter_norm_afib=use_filtered
    )

    # Assign signal frequency and filenames
    train_df, val_df, test_df, sampling_frequency = set_signal_frequency(train_df, val_df, test_df, use_500Hz)

    # Dataset summary
    if use_filtered:
     print("Selected dataset mode: Filtered (AFIB vs NORM)")
    else:
     print("Selected dataset mode: Full (AFIB vs ALL other classes)\n")

    

    # to print class counts per split
    def print_counts(df_split, name):
        n_afib = (df_split["label"] == 1).sum()
        n_other = (df_split["label"] == 0).sum()
        total = len(df_split)
        label_name = "NORM" if use_filtered else "Non-AFIB"
        afib_percent = (n_afib / total) * 100
        print(f"{name:<10} | AFIB: {n_afib:<6} | {label_name}: {n_other:<6} | Total: {total:<6} | AFIB%: {afib_percent:5.2f}%")

    print_counts(train_df, "Train")
    print_counts(val_df,   "Validation")
    print_counts(test_df,  "Test")

    # Overall totals
    combined_df = pd.concat([train_df, val_df, test_df])
    n_afib = (combined_df["label"] == 1).sum()
    n_other = (combined_df["label"] == 0).sum()
    total = len(combined_df)
    label_name = "NORM" if use_filtered else "Non-AFIB"
    afib_percent = (n_afib / total) * 100

    print("\nTotals:")
    label_name = "NORM" if use_filtered else "Non-AFIB"
    print(f" AFIB: {n_afib:<6} | {label_name}: {n_other:<6} | Total: {total:<6} | AFIB%: {afib_percent:5.2f}%\n")

    return train_df, val_df, test_df, sampling_frequency



# Funciton 5: Load a single ECG record from the PTB-XL dataset using WFDB.
def load_ecg(record_path, base_dir=None, return_meta=False):

    # Automatically detect dataset directory if not explicitly provided
    if base_dir is None:
        default_dir = os.path.join(os.getcwd(), "ptb-xl-dataset-1.0.3")
        base_dir = default_dir if os.path.exists(default_dir) else os.getcwd()

    # Construct the full path to the ECG file
    full_path = os.path.join(base_dir, record_path)

    # Check that both header (.hea) and signal (.dat) files exist
    for ext in [".hea", ".dat"]:
        if not os.path.exists(full_path + ext):
            raise FileNotFoundError(f"Missing file: {full_path + ext}")

    record = wfdb.rdrecord(full_path)
    # Convert to float32 NumPy array for model compatibility
    ecg = record.p_signal.astype(np.float32)

    # Optionally return metadata (sampling frequency, lead names)
    if return_meta:
        return ecg, {"fs": record.fs, "leads": record.sig_name}
    return ecg
