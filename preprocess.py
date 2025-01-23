"""
can be run using:

# Basic usage (will create train/test/validation splits in data/ directory)
python preprocess.py --csv-path my_data.csv

# Customize split sizes
python preprocess.py --csv-path my_data.csv --test-size 0.15 --val-size 0.15
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    """
    return pd.read_csv(csv_path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from dataframe
    """
    return df.drop_duplicates()


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for missing value imputation
    """
    return df


def split_and_save_data(df: pd.DataFrame,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> None:
    """
    Split data into train, validation, and test sets and save to CSV files
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set
        random_state (int): Random seed for reproducibility
    """
    # create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # first split: separate test set
    train_val, test = train_test_split(df,
                                      test_size=test_size,
                                      random_state=random_state)

    # second split: separate validation set from training set
    val_ratio = val_size / (1 - test_size)  # Adjust validation ratio
    train, val = train_test_split(train_val,
                                 test_size=val_ratio,
                                 random_state=random_state)

    # save all sets
    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/validation.csv', index=False)
    test.to_csv('data/test.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Preprocess CSV data for machine learning')
    parser.add_argument('--csv-path', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for test set (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Proportion of data for validation set (default: 0.1)')

    args = parser.parse_args()

    # load data
    df = load_data(args.csv_path)

    # remove duplicates
    df = remove_duplicates(df)

    # (placeholder) impute missing values
    df = impute_missing_values(df)

    # split and save data
    split_and_save_data(df, test_size=args.test_size, val_size=args.val_size)


if __name__ == '__main__':
    main()
