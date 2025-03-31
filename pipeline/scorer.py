# pipeline/scorer.py
from abc import ABC, abstractmethod
import numpy as np
import json
from pipeline.logger import get_logger
from pipeline.utils import load_cnn_model


log = get_logger(__name__)


class BaseScorer(ABC):
    @abstractmethod
    def score(self, features: dict) -> float:
        pass


class RuleBasedScorer(BaseScorer):
    def __init__(self, weights: dict, bias: float = 0.0):
        self.weights = weights
        self.bias = bias

    def score(self, features: dict) -> float:
        score = self.bias
        for k, w in self.weights.items():
            if k not in features:
                log.warning(f"Missing feature '{k}' in input — using 0")
                continue
            score += w * features.get(k, 0)
        return score

    def to_dict(self):
        return {
            "weights": self.weights,
            "bias": self.bias
        }

    @classmethod
    def from_dict(cls, d):
        return cls(weights=d["weights"], bias=d.get("bias", 0))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.debug(f"RuleBasedScorer saved to {path}")

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        return RuleBasedScorer.from_dict(d)


class MLScorer(BaseScorer):
    def __init__(self, model, feature_order):
        self.model = model
        self.feature_order = feature_order

    def score(self, features: dict) -> float:
        x = np.array([[features.get(f, 0.0) for f in self.feature_order]])
        score = self.model.predict_proba(x)[0][1]  # prob of class 1
        return float(score)

    def save(self, path):
        import joblib
        joblib.dump({"model": self.model, "feature_order": self.feature_order}, path)
        log.debug(f"MLScorer saved to {path}")

    @staticmethod
    def load(path):
        import joblib
        d = joblib.load(path)
        return MLScorer(model=d["model"], feature_order=d["feature_order"])

class CNNScorer(BaseScorer):
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = load_cnn_model(model_path, device)

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def score(self, image: np.ndarray) -> float:
        from PIL import Image
        import torch

        pil = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        x = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
        return float(probs[1])  # prob of class "good"
