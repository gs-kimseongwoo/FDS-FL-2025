# main.py

import torch
import argparse
from federated.utils import set_seed
from federated.model.model_utils import get_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=1, help='GPU index to use')
    parser.add_argument('--data_dir', default='dataset/federated_split', help='Directory containing the federated split data')
    parser.add_argument('--num_silos', type=int, default=4, help='Number of silos/clients')
    parser.add_argument('--method', type=str, default='fedavg', help='Federated learning method to use (e.g., fedavg)')

    # Architecture
    parser.add_argument('--input_dim', type=int, default=50, help='Input feature dimension')
    parser.add_argument('--model_type', type=str, default='mlp', help='Type of model to use (e.g., mlp)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the model')

    # Training hyperparameters
    parser.add_argument('--global_rounds', type=int, default=10, help='Number of global training rounds')
    parser.add_argument('--ep', type=int, default=3, help='Number of local training epochs')
    parser.add_argument('--bs', type=int, default=32, help='Local batch size')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    return parser.parse_args()



def main():
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"
    model = get_model(args.model_type, args)

    if args.method == 'fedavg':
        from federated.training.server.FedAvg_Server import FedAvg_Server
        server = FedAvg_Server(model, device, args)
        server.run()
    else:
        raise ValueError(f"Unsupported method: {args.method}")




if __name__ == '__main__':
    main()