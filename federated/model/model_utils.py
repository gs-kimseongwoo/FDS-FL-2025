# federated/model/model_utils.py


def get_model(model_type, args):
    if model_type == 'mlp':
        from federated.model.mlp import MLP
        return MLP(args)
    raise ValueError(f"Unsupported model type: {model_type}")