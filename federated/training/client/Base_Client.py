# federated/training/client/Base_Client.py

import torch
import pandas as pd
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset


class Base_Client(ABC):
    def __init__(self, client_id, device, args):
        self.client_id = client_id
        self.device = device
        self.args = args

        self.optim = args.optim
        self.lr = args.lr
        self.ep = args.ep

        self.data_sizes = {'train': 0, 'test': 0} # Lazy initialization
        self.train_loader, self.test_loader = None, None
        self.load_client_data()


    def load_client_data(self):
        dataset_dir = f'{self.args.data_dir}/num_silos_{self.args.num_silos}'
        train_path = f'{dataset_dir}/silo_{self.client_id}/train_data.csv'
        test_path = f'{dataset_dir}/silo_{self.client_id}/test_data.csv'

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = torch.tensor(train_df.drop('label', axis=1).values, dtype=torch.float32)
        y_train = torch.tensor(train_df['label'].values, dtype=torch.float32)

        X_test = torch.tensor(test_df.drop('label', axis=1).values, dtype=torch.float32)
        y_test = torch.tensor(test_df['label'].values, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        self.data_sizes['train'] = len(train_dataset)
        self.data_sizes['test'] = len(test_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.bs, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.bs, shuffle=False)



    @abstractmethod
    def local_train(self, passed_model):
        pass


    @abstractmethod
    def local_test(self, model):
        pass


    @abstractmethod
    def _set_model_mode(self, model, mode):
        pass


    def get_data_size(self, data_type):
        assert data_type in ['train', 'test'], "data_type must be either 'train' or 'test'"
        return self.data_sizes[data_type]
