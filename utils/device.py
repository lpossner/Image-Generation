import torch


def detect_device(preferred_device):
    if preferred_device == "mps":
        if torch.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS is not available as a device on this platform")
    if preferred_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA is not available as a device on this platform")
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
