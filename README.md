## Project Structure

```text
FDS-FL-2025
├── centralized                      # Scripts for Centralized setting
|   ├── preprocess_data.py           # Preprocess raw data
|   └── split_data.py                # Split preprocessed data into train/test
|
├── federated                        # All logic for Federated Learning
|   ├── data_split                   # Script for splitting data for FL
|   |   └── split_data_into_silos.py # Partitions the entire dataset into N client 'silos'
|   |
|   ├── model                        # Model architecture definitions
|   |   ├── mlp.py                   # The MLP model structure used in experiments
|   |   └── model_utils.py           # Model builder helper
|   |
|   ├── training                     # Main FL training logic
|   |   ├── client                   # Defines individual client behavior
|   |   |   ├── Base_Client.py       # Abstract client class
|   |   |   └── FedAvg_Client.py     # Implements FedAvg local training and testing logic
|   |   |
|   |   └── server                   # Defines central server behavior
|   |       ├── Base_Server.py       # Abstract server class
|   |       └── FedAvg_Server.py     # Implements FedAvg model aggregation and the main loop
|   |
|   └── utils.py                     # Other utilities (e.g., set_seed)
|
├── main.py                          # (MAIN) Main script to run FL experiments
└── vpfl.yml                         # Conda environment configuration file
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
