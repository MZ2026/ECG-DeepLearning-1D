import ast
import numpy as np
import pandas as pd


# PTB-XL Dataset Utility Functions
# These functions help you:
# 1. Filter the dataset to include only specific ECG labels (e.g., NORM or AFIB)
# 2. Convert ECG diagnosis codes into numeric labels
# 3. Select stratified folds for training/validation/testing
# 4. Configure dataset frequency (100Hz or 500Hz)
# 5. Print dataset statistics including total and class counts


# Function 1: Filter to keep only NORM/AFIB
def filter_data(df):
    """Filter PTB-XL DataFrame to include only NORM and AFIB samples."""
    return df[df["scp_codes"].apply(
        lambda s: "NORM" in ast.literal_eval(s) or "AFIB" in ast.literal_eval(s)
    )].copy()


# Function 2: Convert scp_codes to binary label (0=NORM, 1=AFIB)
def label_to_binary(scp_str: str) -> int:
    """Convert scp_codes string to binary label: AFIB→1, NORM→0."""
    scp = ast.literal_eval(scp_str)
    if "AFIB" in scp:
        return 1
    if "NORM" in scp:
        return 0
    return np.nan


def get_primary_label(scp_str):
    """Extract the first diagnosis code from scp_codes string."""
    scp = ast.literal_eval(scp_str)
    return list(scp.keys())[0] if len(scp) > 0 else None



# Function 3: Select folds for train/val/test
def select_folds(df, train_folds, val_folds, test_folds, filter_norm_afib=False):
    """
    Split PTB-XL dataset into training, validation, and test subsets based on folds.
    
    PTB-XL comes pre-divided into 10 stratified folds (1–10).
    This function allows flexible selection of folds and optional NORM/AFIB filtering.
    """

    # Ensure all fold inputs are lists
    if isinstance(train_folds, int): train_folds = [train_folds]
    if isinstance(val_folds, int):   val_folds = [val_folds]
    if isinstance(test_folds, int):  test_folds = [test_folds]

    # Split DataFrame
    train_df = df[df["strat_fold"].isin(train_folds)].copy()
    val_df   = df[df["strat_fold"].isin(val_folds)].copy()
    test_df  = df[df["strat_fold"].isin(test_folds)].copy()

    # Filter to NORM/AFIB if required
    if filter_norm_afib:
        train_df = filter_data(train_df)
        val_df   = filter_data(val_df)
        test_df  = filter_data(test_df)

        # Add binary labels
        for subset in [train_df, val_df, test_df]:
            subset["label"] = subset["scp_codes"].apply(label_to_binary)

    else:
        # Add a single primary label for full multi-class dataset
        for subset in [train_df, val_df, test_df]:
            subset["primary_label"] = subset["scp_codes"].apply(get_primary_label)

    


    # Summary
    print("Selected folds:")
    print(f" Train: Fold {train_folds} ({len(train_df)} samples)")
    print(f" Validation: Fold {val_folds} ({len(val_df)} samples)")
    print(f" Test: Fold {test_folds} ({len(test_df)} samples)")
    print(f" Total: {len(train_df) + len(val_df) + len(test_df)} samples\n")

    return train_df, val_df, test_df


#Count class distribution
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


# Function 4: Main wrapper — setup_dataset()
def setup_dataset(df, train_folds, val_folds, test_folds, use_500Hz=False, use_filtered=False):
    """
    Prepares training, validation, and test datasets with optional filtering and sampling rate.
    
    """

    # Select folds (and filter if needed)
    train_df, val_df, test_df = select_folds(
        df, train_folds, val_folds, test_folds, filter_norm_afib=use_filtered
    )

    # Choose signal frequency and assign correct filename column to each split
    if use_500Hz:
        for subset in [train_df, val_df, test_df]:
            subset["filename"] = subset["filename_hr"]
        frequency_rate = 500
        print("Using ECG signal (500 Hz)\n")
    else:
        for subset in [train_df, val_df, test_df]:
            subset["filename"] = subset["filename_lr"]
        frequency_rate = 100
        print("Using ECG signal (100 Hz)\n")

    # Dataset summary
    total_samples = len(train_df) + len(val_df) + len(test_df)
    if use_filtered:
     print("Using Filtered (Only NORM/AFIB)")
    else:
     print("Using full dataset (all classes)")
    print(f"Sampling rate: {frequency_rate}Hz")

    

    # Class distribution
    combined_df = pd.concat([train_df, val_df, test_df])
    norm_count, afib_count, other_count = count_labels(combined_df)

    if use_filtered:
        print("\nClass Summary (filtered):")
        print(f" NORM: {norm_count}")
        print(f" AFIB: {afib_count}")
    else:
        print("\nClass Summary (full):")
        print(f" NORM: {norm_count}")
        print(f" AFIB: {afib_count}")
        print(f" Other: {other_count}")

    print() 

   
    return train_df, val_df, test_df, frequency_rate
