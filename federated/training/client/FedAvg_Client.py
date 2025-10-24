# federated/training/client/FedAvg_Client.py

import torch
from federated.training.client.Base_Client import Base_Client
from sklearn.metrics import precision_score, recall_score, f1_score



class FedAvg_Client(Base_Client):
    """Client for FedAvg, which uses a shared model across all clients."""

    def __init__(self, client_id, device, args):
        super().__init__(client_id, device, args)


    def local_train(self, model):
        self._set_model_mode(model, mode='train')
        optimizer = self._get_optimizer(model)

        # BCE with logits loss for binary classification
        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        loss_list = []
        for epoch in range(self.ep):
            epoch_loss = 0.0

            for batch_idx, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device).float() # Cast y to float for BCEWithLogitsLoss
                optimizer.zero_grad()

                outputs = model(X)
                # Squeeze the outputs for binary classification
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Avg loss over this epochs
            avg_loss = epoch_loss / len(self.train_loader)
            loss_list.append(avg_loss)

        client_loss = sum(loss_list) / len(loss_list)
        # print(f'Client {self.client_id} | training loss: {client_loss:.4f}')
        return model, client_loss



    @torch.no_grad()
    def local_test(self, model):
        self._set_model_mode(model, mode='eval')

        y_true_list, y_pred_list = [], []
        for batch_idx, (X, y) in enumerate(self.test_loader):
            X, y = X.to(self.device), y.to(self.device).float()
            outputs = model(X)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            y_pred_list.extend(preds.detach().cpu().numpy())
            y_true_list.extend(y.detach().cpu().numpy())

        # Calculate performance metrics
        recall = recall_score(y_true_list, y_pred_list, average='macro')
        precision = precision_score(y_true_list, y_pred_list, average='macro')
        f1 = f1_score(y_true_list, y_pred_list, average='macro')

        return {
            'client_id': self.client_id,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }



    def _set_model_mode(self, model, mode):
        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval()


    def _get_optimizer(self, model):
        if self.optim == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        raise ValueError(f"Unsupported optimizer: {self.optim}")