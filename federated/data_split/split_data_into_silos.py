# federated/data_split/split_data_into_silos.py

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMN_NAME = 'label'
RANDOM_SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser('Split preprocessed data for Federated Learning simulation.')
    parser.add_argument('--input_path', default='../../dataset/processed_data_no_missing.csv',
                        help='Path to the preprocessed CSV file')
    parser.add_argument('--save_dir', default='../../dataset/federated_split',
                        help='Base directory to save the federated split data')
    parser.add_argument('--num_silos', type=int, default=4,
                        help='The number of silos to partition the data into')
    parser.add_argument('--test_split_ratio', type=float, default=0.3,
                        help='Test split ratio for the data within each silo')
    return parser.parse_args()


def main():
    args = parse_arguments()
    all_data = pd.read_csv(args.input_path)

    print(f"Splitting data into {args.num_silos} silos...")

    # Distribute df according to the label
    normal_df = all_data[all_data[LABEL_COLUMN_NAME] == 0]
    fraud_df = all_data[all_data[LABEL_COLUMN_NAME] == 1]

    # Shuffling the sample
    normal_df = normal_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    fraud_df = fraud_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Divide each group into num_silos parts
    normal_parts = np.array_split(normal_df, args.num_silos)
    fraud_parts = np.array_split(fraud_df, args.num_silos)

    intermediate_folder = os.path.join(args.save_dir, f'num_silos_{args.num_silos}')

    data_stats = {}
    for i in range(args.num_silos):
        silo_number = i + 1

        silo_data = pd.concat([normal_parts[i],fraud_parts[i]])

        X = silo_data.drop(LABEL_COLUMN_NAME, axis=1)
        y = silo_data[LABEL_COLUMN_NAME]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_split_ratio,
            random_state=RANDOM_SEED,
            stratify=y
        )


        output_dir = os.path.join(intermediate_folder, f'silo_{silo_number}')
        os.makedirs(output_dir, exist_ok=True)

        train_data = X_train.copy()
        train_data[LABEL_COLUMN_NAME] = y_train

        test_data = X_test.copy()
        test_data[LABEL_COLUMN_NAME] = y_test

        train_path = os.path.join(output_dir, 'train_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        data_stats[silo_number] = {

        }
        # print(f"silo_{silo_number} data saved successfully. Label distribution in silo:")
        # print(silo_data[LABEL_COLUMN_NAME].value_counts(normalize=True))
        # print("Count in each silo:")
        # print(silo_data[LABEL_COLUMN_NAME].value_counts(normalize=False))

    # Dataframe of stats
    # silo_id, total_samples, total_normal, total_fraud, train_samples, train_normal, train_fraud, test_samples, test_normal, test_fraud

    # sum(total_normal from all silos) == total normal in all_data
    # same for fraud
    # assert()

    print("\nFederated data splitting process completed.")
    print(all_data[LABEL_COLUMN_NAME].value_counts(normalize=False))
if __name__ == '__main__':
    main()
