from datetime import datetime

import os

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False


class Logger:
    """
    Main logger interface for training metrics.

    Automatically selects between Neptune cloud logging (if available and configured)
    or local text file logging.
    """

    def __init__(self, checkpoint_name, log_directory, neptune_project, neptune_token):
        """
        Initialize the logger.

        Args:
            checkpoint_name (str): Name for the checkpoint/experiment.
            log_directory (str): Directory for local log files.
            neptune_project (str): Neptune.ai project name (empty string to disable).
            neptune_token (str): Neptune.ai API token (empty string to disable).
        """
        self.checkpoint_name = checkpoint_name
        if NEPTUNE_AVAILABLE and neptune_project and neptune_token:
            self.logger = NeptuneLogger(neptune_project, neptune_token)
        else:
            self.logger = TextLogger(log_directory, checkpoint_name)
        
    def log_metrics(self, metrics, epoch, epochs):
        """
        Log training metrics.

        Args:
            metrics (dict): Dictionary of metric names and values.
            epoch (int): Current epoch number (0-indexed).
            epochs (int): Total number of epochs.
        """
        self.logger.log_metrics(metrics, epoch, epochs)

    def update_progress(self, progress, metrics, epoch, epochs):
        """
        Update progress bar with current metrics.

        Args:
            progress (tqdm): tqdm progress bar object.
            metrics (dict): Dictionary of metric names and values.
            epoch (int): Current epoch number (0-indexed).
            epochs (int): Total number of epochs.
        """
        self.logger.update_progress(progress, metrics, epoch, epochs)


class NeptuneLogger:
    """
    Neptune.ai cloud-based logging implementation.

    Logs metrics to Neptune.ai platform for experiment tracking and visualization.
    """

    def __init__(self, neptune_project, neptune_token):
        """
        Initialize Neptune logger and create a new run.

        Args:
            neptune_project (str): Neptune.ai project name (format: 'workspace/project').
            neptune_token (str): Neptune.ai API authentication token.
        """
        self.run = neptune.init_run(
            project=neptune_project,
            api_token=neptune_token,
        )

    def log_metrics(self, metrics, epoch, epochs):
        """
        Log metrics to Neptune.ai.

        Args:
            metrics (dict): Dictionary of metric names and values.
            epoch (int): Current epoch number (0-indexed, unused).
            epochs (int): Total number of epochs (unused).
        """
        for key, value in metrics.items():
            self.run[key].append(value)

    @staticmethod
    def update_progress(progress, metrics, epoch, epochs):
        """
        Update tqdm progress bar description with current metrics.

        Args:
            progress (tqdm): tqdm progress bar object.
            metrics (dict): Dictionary of metric names and values.
            epoch (int): Current epoch number (0-indexed).
            epochs (int): Total number of epochs.
        """
        description = f"Epoch {epoch+1}/{epochs}, " + ", ".join(
            [f"{key}: {value:.4f}" for key, value in metrics.items()]
        )
        progress.set_description(description)

    def __del__(self):
        """Stop Neptune run when logger is destroyed."""
        self.run.stop()


class TextLogger:
    """
    Local text file logging implementation.

    Writes training metrics to timestamped log files on disk.
    """

    def __init__(self, log_directory, checkpoint_name):
        """
        Initialize text logger with log file path.

        Args:
            log_directory (str): Directory to save log files.
            checkpoint_name (str): Name for the checkpoint/experiment.
        """
        self.log_directory = log_directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_name = f"{checkpoint_name}_{timestamp}.log"
        self.logfile_path = os.path.join(self.log_directory, self.log_name)

    def log_metrics(self, metrics, epoch, epochs):
        """
        Append metrics to log file with timestamp.

        Args:
            metrics (dict): Dictionary of metric names and values.
            epoch (int): Current epoch number (0-indexed).
            epochs (int): Total number of epochs.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.logfile_path, "a") as f:
            metrics_str = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
            log_line = f"[{timestamp}] Epoch {epoch+1}/{epochs} - {metrics_str}"
            f.write(log_line + "\n")

    @staticmethod
    def update_progress(progress, metrics, epoch, epochs):
        """
        Update tqdm progress bar description with current metrics.

        Args:
            progress (tqdm): tqdm progress bar object.
            metrics (dict): Dictionary of metric names and values.
            epoch (int): Current epoch number (0-indexed).
            epochs (int): Total number of epochs.
        """
        description = f"Epoch {epoch+1}/{epochs}, " + ", ".join(
            [f"{key}: {value:.4f}" for key, value in metrics.items()]
        )
        progress.set_description(description)
