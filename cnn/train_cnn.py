# cnn/train_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from dataset import RingCropDataset
from sklearn.metrics import classification_report
import json
from pathlib import Path
from datetime import datetime

def train_cnn(csv_path, out_dir="cnn_model", epochs=10, batch_size=32, lr=1e-4):
    dataset = RingCropDataset(csv_path)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = resnet18(pretrained=False, num_classes=2)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    print("\nValidation Report:")
    print(classification_report(y_true, y_pred, target_names=["bad", "good"]))

    # Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(out_dir) / "cnn_model.pt")

    with open(Path(out_dir) / "train_metadata.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "csv": str(csv_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        }, f, indent=2)

    print(f"✅ Saved model to {Path(out_dir) / 'cnn_model.pt'}")
