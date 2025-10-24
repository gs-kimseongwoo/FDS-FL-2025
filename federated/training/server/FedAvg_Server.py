# federated/training/server/FedAvg_Server.py

import copy
import torch
from federated.training.server.Base_Server import Base_Server
from federated.training.client.FedAvg_Client import FedAvg_Client



class FedAvg_Server(Base_Server):
    def __init__(self, model, device, args):
        super().__init__(model, device, args)
        self.model.to(self.device)


    def _create_client(self, client_id):
        return FedAvg_Client(client_id, self.device, self.args)


    def run(self):
        self.run_federated_training()
        self.run_federated_evaluation()


    def run_federated_training(self):
        global_rounds = self.args.global_rounds  # e.g., 10
        print(f"Starting Federated Training for {global_rounds} rounds...")

        for rnd in range(global_rounds):
            # Dispatch the global model to every client, perform local training
            updated_models = {}  # (client_id: updated_model)
            client_losses = []
            for client_id in self.clients:
                client = self.clients[client_id]
                # Create a client-side copy of the global model
                trained_model, client_loss = client.local_train(copy.deepcopy(self.model))
                updated_models[client_id] = trained_model
                client_losses.append(client_loss)

            # Aggregate the client models to update the global model
            self.model = self.aggregate_models(updated_models)

            # Optionally, print the average training loss across all clients in this round
            avg_loss = sum(client_losses) / len(client_losses)
            print(f"[Server Round: {rnd+1}/{global_rounds}] Avg Training Loss after Round {rnd + 1}: {avg_loss:.4f}")



    def run_federated_evaluation(self):
        print("\n[Server] Starting Federated Evaluation...")

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for client_id in self.clients:
            client = self.clients[client_id]
            # Evaluate on local test data (local_test returns (loss, precision, recall, f1))
            perf_metrics = client.local_test(copy.deepcopy(self.model))

            total_precision += perf_metrics['precision']
            total_recall += perf_metrics['recall']
            total_f1 += perf_metrics['f1']

        num_clients = len(self.clients)
        avg_precision = total_precision / num_clients
        avg_recall = total_recall / num_clients
        avg_f1 = total_f1 / num_clients

        print(f"[Server] Global Test Metrics \n"
              f"Precision (macro): {avg_precision:.4f}, "
              f"Recall (macro): {avg_recall:.4f}, "
              f"F1 (macro): {avg_f1:.4f}")


    def aggregate_models(self, updated_models):
        # Prepare data
        client_ids = list(updated_models.keys())
        client_n = self._get_client_data_sizes(client_ids)
        client_state_dicts = {cid: updated_models[cid].state_dict() for cid in client_ids}

        # Set up aggregation
        new_model = copy.deepcopy(self.model)
        global_state_dict = new_model.state_dict()

        self._aggregate_model_parameters(client_ids, client_n, client_state_dicts, global_state_dict)
        # Load weights to the new_model
        new_model.load_state_dict(global_state_dict)
        return new_model


    def _get_client_data_sizes(self, client_ids):
        # Extract client data sizes for weighting
        client_n = {}
        for cid in client_ids:
            client = self.clients[cid]
            client_n[cid] = client.get_data_size('train')
        return client_n


    def _aggregate_model_parameters(self, client_ids, client_n, client_state_dicts, global_state_dict):
        # Aggregate non-encoder params using data size weighting
        total_cnt = sum(client_n[cid] for cid in client_ids)

        for key in global_state_dict.keys():
            agg = torch.zeros_like(global_state_dict[key])
            contrib_str = ""
            for cid in client_ids:
                w = client_n[cid] / total_cnt
                agg.add_(client_state_dicts[cid][key], alpha=w)
                contrib_str += f"{cid}: {w:.4f} | "
            # print(f'[Agg] [Key={key}] Contributions: {contrib_str}')
            global_state_dict[key] = agg
