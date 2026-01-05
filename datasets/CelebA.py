import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


class CelebADataset(Dataset):

    def __init__(self, img_directory):
        self.img_directory = img_directory
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images
            transforms.ToTensor(),  # Convert to tensor
        ])
        self.image_filenames = [filename for filename in os.listdir(img_directory) if filename.lower().endswith('.jpg')]
        self.image_filenames.sort()
        if not len(self.image_filenames):
            raise ValueError("Image directory empty")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_directory, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel image (RGB)
        
        if self.transform:
            image = self.transform(image)

        return image


class CelebADatalLoader(DataLoader):

    def __init__(self, img_directory, batch_size):
        dataset = CelebADataset(img_directory)
        super().__init__(dataset, batch_size=batch_size, shuffle=True)
