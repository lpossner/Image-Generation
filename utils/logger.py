from datetime import datetime

import os

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False


class Logger:

    def __init__(self, checkpoint_name, log_directory, neptune_project, neptune_token):
        self.checkpoint_name = checkpoint_name
        if NEPTUNE_AVAILABLE and neptune_project and neptune_token:
            self.logger = NeptuneLogger(neptune_project, neptune_token)
        else:
            self.logger = TextLogger(log_directory, checkpoint_name)
        
    def log_metrics(self, metrics, epoch, epochs):
        self.logger.log_metrics(metrics, epoch, epochs)
    
    def update_progress(self, progress, metrics, epoch, epochs):
        self.logger.update_progress(progress, metrics, epoch, epochs)


class NeptuneLogger:

    def __init__(self, neptune_project, neptune_token):
        self.run = neptune.init_run(
            project=neptune_project,
            api_token=neptune_token,
        )

    def log_metrics(self, metrics, epoch, epochs):
        for key, value in metrics.items():
            self.run[key].append(value)

    @staticmethod
    def update_progress(progress, metrics, epoch, epochs):
        description = f"Epoch {epoch+1}/{epochs}, " + ", ".join(
            [f"{key}: {value:.4f}" for key, value in metrics.items()]
        )
        progress.set_description(description)
    
    def __del__(self):
        self.run.stop()


class TextLogger:

    def __init__(self, log_directory, checkpoint_name):
        self.log_directory = log_directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_name = f"{checkpoint_name}_{timestamp}.log"
        self.logfile_path = os.path.join(self.log_directory, self.log_name)

    def log_metrics(self, metrics, epoch, epochs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.logfile_path, "a") as f:
            metrics_str = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
            log_line = f"[{timestamp}] Epoch {epoch+1}/{epochs} - {metrics_str}"
            f.write(log_line + "\n")

    @staticmethod
    def update_progress(progress, metrics, epoch, epochs):
        description = f"Epoch {epoch+1}/{epochs}, " + ", ".join(
            [f"{key}: {value:.4f}" for key, value in metrics.items()]
        )
        progress.set_description(description)
