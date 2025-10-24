import time
import pandas as pd
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def parse_arguments():
    parser = argparse.ArgumentParser('Train and evaluate multiple ML models & simple DL model.')
    parser.add_argument( '--input_dir', default='../dataset/centralized_split',
                        help='Directory containing the split train_data.csv and test_data.csv')
    return parser.parse_args()


def main():
    args = parse_arguments()


    print("Loading data...")
    train_path = os.path.join(args.input_dir, 'train_data.csv')
    test_path = os.path.join(args.input_dir, 'test_data.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    print("Data loaded successfully.")
    print("-" * 50)

    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    for name, model in models.items():
        print(f"[{name}]")

        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")

        # prediction
        y_pred = model.predict(X_test_scaled)

        # results
        print(classification_report(y_test, y_pred))



if __name__ == '__main__':
    main()