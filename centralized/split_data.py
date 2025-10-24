# centralized/split_data.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMN_NAME = 'label'
RANDOM_SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser('Split preprocessed CSV data into training and testing sets (Centralized Setting).')
    parser.add_argument('--input_path', default='../dataset/processed_data_no_missing.csv',
                        help='Path to the preprocessed CSV file')
    parser.add_argument('--save_dir', default='../dataset/centralized_split', help='Directory to save the split CSV files')
    parser.add_argument('--test_split_ratio', type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_arguments()
    df = pd.read_csv(args.input_path)

    X = df.drop(LABEL_COLUMN_NAME, axis=1)
    y = df[LABEL_COLUMN_NAME]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split_ratio,
                                                        random_state=RANDOM_SEED, stratify=y)

    print("label of train data distribution:")
    print(y_train.value_counts(normalize=True))
    print("-" * 30)
    print("label of test data distribution:")
    print(y_test.value_counts(normalize=True))

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data[LABEL_COLUMN_NAME] = y_train
    test_data[LABEL_COLUMN_NAME] = y_test

    train_data.to_csv(os.path.join(save_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(save_dir, 'test_data.csv'), index=False)



if __name__ == '__main__':
    main()

