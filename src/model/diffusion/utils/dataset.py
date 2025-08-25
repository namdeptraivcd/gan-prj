import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# @TODO
class TextImageDataset(Dataset):
    def __init__(self, path, image_size=64):
        self.path = path
        self.image_files = [f for f in os.listdir(path) if f.endswith(".png")]
        self.captions = open(os.path.join(path, "captions.txt")).read().splitlines()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.image_files[idx])).convert("RGB")
        return self.transform(img), self.captions[idx]
