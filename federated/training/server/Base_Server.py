# federated/training/server/Base_Server.py

from tqdm import tqdm
from abc import ABC, abstractmethod



class Base_Server(ABC):
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args

        self.client_ids = list(range(1, args.num_silos + 1)) # Assuming client IDs are 1-indexed
        self.clients = {}
        self.initialize_server()


    def initialize_server(self):
        self._setup_clients()


    def _setup_clients(self):
        for client_id in tqdm(self.client_ids, desc="Initializing clients"):
            self.clients[client_id] = self._create_client(client_id)


    @abstractmethod
    def _create_client(self, client_id):
        pass


    @abstractmethod
    def run(self):
        pass


    @abstractmethod
    def run_federated_training(self):
        pass


    @abstractmethod
    def run_federated_evaluation(self):
        pass
