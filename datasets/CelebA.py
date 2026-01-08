import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


class CelebADataset(Dataset):
    """
    PyTorch Dataset for CelebA face images.

    Loads and preprocesses images from a directory for training generative models.
    Images are resized to 64x64 and converted to tensors.
    """

    def __init__(self, image_directory):
        """
        Initialize the CelebA dataset.

        Args:
            image_directory (str): Path to directory containing CelebA JPG images.

        Raises:
            ValueError: If the image directory is empty.
        """
        self.image_directory = image_directory
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images
            transforms.ToTensor(),  # Convert to tensor
        ])
        self.image_filenames = [filename for filename in os.listdir(image_directory) if filename.lower().endswith('.jpg')]
        self.image_filenames.sort()
        if not len(self.image_filenames):
            raise ValueError("Image directory empty")

    def __len__(self):
        """
        Get the number of images in the dataset.

        Returns:
            int: Total number of JPG images in the directory.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Load and preprocess a single image.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (3, 64, 64) with values in [0, 1].
        """
        image_path = os.path.join(self.image_directory, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image (RGB)
        
        if self.transform:
            image = self.transform(image)

        return image


class CelebADatalLoader(DataLoader):
    """
    Custom DataLoader for CelebA dataset.

    Wraps CelebADataset with batching and shuffling for training.
    """

    def __init__(self, image_directory, batch_size):
        """
        Initialize the CelebA DataLoader.

        Args:
            image_directory (str): Path to directory containing CelebA JPG images.
            batch_size (int): Number of images per batch.
        """
        dataset = CelebADataset(image_directory)
        super().__init__(dataset, batch_size=batch_size, shuffle=True)
