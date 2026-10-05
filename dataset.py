from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class MedicalSegmentationDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.images = sorted((self.root / "images").glob("*.png"))
        self.masks = sorted((self.root / "masks").glob("*.png"))

        if len(self.images) != len(self.masks):
            raise RuntimeError("Image/mask count mismatch.")
        if not self.images:
            raise RuntimeError(f"No data found in {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(
            Image.open(self.images[index]).convert("RGB"),
            dtype=np.float32
        ) / 255.0

        mask = np.asarray(
            Image.open(self.masks[index]).convert("L"),
            dtype=np.float32
        ) / 255.0

        mask = (mask >= 0.5).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)[None]

        return image, mask, self.images[index].name
