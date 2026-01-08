import torch


def detect_device(preferred_device):
    """
    Detect and return appropriate PyTorch device.

    Checks availability of requested device (MPS/CUDA/CPU) and returns the
    appropriate torch.device. If no preference is specified, automatically
    selects the best available device (CUDA > MPS > CPU).

    Args:
        preferred_device (str): Preferred device type ('mps', 'cuda', 'cpu', or None).

    Returns:
        torch.device: PyTorch device object for the selected device.

    Raises:
        RuntimeError: If the preferred device is not available on this platform.
    """
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
