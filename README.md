## Project Structure

```text
FDS-FL-2025
├── centralized             # Scripts for Centralized setting
|   ├── preprocess_data.py  # (1) Preprocess raw data
|   └── split_data.py       # (2) Split preprocessed data into train/test
|
├── federated               # All logic for Federated Learning
|   ├── data_split          #   Script for splitting data for FL
|   |   └── split_data_into_silos.py
|   |
|   ├── model               #   Model architecture definitions
|   |   ├── mlp.py          # (e.g., MLP model)
|   |   └── model_utils.py  # (Model builder utility)
|   |
|   ├── training            #   Main FL training logic
|   |   ├── client          #   (1) Client-side logic
|   |   |   ├── Base_Client.py
|   |   |   └── FedAvg_Client.py
|   |   |
|   |   └── server          #   (2) Server-side logic
|   |       ├── Base_Server.py
|   |       └── FedAvg_Server.py
|   |
|   └── utils.py            #   Other utilities (e.g., set_seed)
|
├── main.py                 # (MAIN) Main script to run FL experiments
└── vpfl.yml                # Conda environment configuration file
```
## How to Run?

1.  Create and activate the conda environment using the `vpfl.yml` file:
    * `conda env create -f vpfl.yml`
    * `conda activate vpfl`

2.  Prepare the datasets.
    * Place your raw data file (e.g., `raw.csv`) inside a directory named `dataset/` at the project root.
    * Run the preprocessing and splitting scripts in order. The scripts will use the default paths:
        1.  `python centralized/preprocess_data.py`
        2.  `python centralized/split_data.py`
        3.  `python federated/data_split/split_data_into_silos.py`
    * This will generate all necessary `train_data.csv` and `test_data.csv` files in `dataset/centralized_split` and `dataset/federated_split`.

3.  For the **Centralized (Traditional ML)** experiment, run the `source.py` script:
    * `python centralized/source.py`
    * This script will automatically load the centralized data and run models like Random Forest, XGBoost, etc.

4.  For the **Federated Learning (FL)** experiment, run the `main.py` script:
    * `python main.py {desired experiment settings}`
    * All arguments are set in `main.py`. You can run with defaults:
        * `python main.py`
    * Or override settings like:
        * `python main.py --global_rounds 20 --num_silos 5 --lr 0.01`
    * Refer to `main.py` for the full list of available arguments.
