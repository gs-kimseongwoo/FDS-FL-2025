## Project Structure

```text
FDS-FL-2025
├── centralized                      # Scripts for Centralized setting
|   ├── preprocess_data.py           # Preprocess raw data
|   ├── source.py                    # Trains and evaluates ML models (RF, XGBoost, etc.)
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
        
        **(1) Preprocessing & Centralized Split:**
        ```bash
        cd centralized
        python preprocess_data.py
        python split_data.py
        cd .. 
        ```

        **(2) Federated Split:**
        ```bash
        cd federated/data_split
        python split_data_into_silos.py
        cd ../..
        ```
    * This will generate all necessary files in `dataset/centralized_split` and `dataset/federated_split`.

3.  For the **Centralized (Traditional ML)** experiment:
    * (Run from inside the `centralized` folder)
    
        ```bash
        cd centralized
        python source.py
        cd ..
        ```

4.  For the **Federated Learning (FL)** experiment:
    * (This one is run from the root directory)
    
        ```bash
        python main.py {desired experiment settings}
        ```
    * All arguments are set in `main.py`. You can run with defaults:
        * `python main.py`
    * Or override settings like:
        * `python main.py --global_rounds 20 --num_silos 5 --lr 0.01`
    * Refer to `main.py` for the full list of available arguments.
