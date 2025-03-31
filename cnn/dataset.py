# cnn/dataset.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class RingCropDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["label"].isin(["good", "bad"])]
        self.img_paths = self.df["cell_label"].apply(
            lambda x: Path(csv_path).parent / "crops" / f"cell_{int(x):03d}.png"
        )
        self.labels = self.df["label"].map({"bad": 0, "good": 1}).tolist()
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths.iloc[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
