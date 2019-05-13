import torch


def get_device(device="gpu"):

    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            raise Exception("CUDA device not available")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise Exception("Only 'gpu' and 'cpu' devices are supported.")

