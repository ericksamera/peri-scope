from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw

from config import CROP_PADDING
from pipeline.logger import get_logger

log = get_logger(__name__)

@dataclass
class CellSample:
    cell_id: int
    region: object
    cell_mask: np.ndarray
    ring_mask: Optional[np.ndarray] = None
    features: Optional[Dict[str, float]] = None
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def bounding_box(self):
        return self.region.bbox

    def centroid(self):
        return self.region.centroid

    def crop(self, image: np.ndarray) -> np.ndarray:
        minr, minc, maxr, maxc = self.bounding_box()
        return image[minr:maxr, minc:maxc]

    def ring_crop(self) -> Optional[np.ndarray]:
        if self.ring_mask is not None:
            return self.crop(self.ring_mask)
        return None

    def to_row(self) -> Dict[str, Any]:
        return {
            "cell_label": self.cell_id,
            "score": self.score,
            **(self.features or {}),
            **self.meta,
            "bbox_minr": self.region.bbox[0],
            "bbox_minc": self.region.bbox[1],
            "bbox_maxr": self.region.bbox[2],
            "bbox_maxc": self.region.bbox[3],
        }

    def has_ring(self) -> bool:
        return self.ring_mask is not None and np.any(self.ring_mask)

    def set_score(self, score: float):
        self.score = float(score)

    def set_features(self, features: Dict[str, float]):
        self.features = features

    def set_ring_mask(self, ring_mask: np.ndarray):
        self.ring_mask = ring_mask

    def save_debug(self, protein_img, membrane_img, ring_mask, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        center_r, center_c = np.round(self.region.centroid).astype(int)
        minr = max(center_r - CROP_PADDING, 0)
        maxr = min(center_r + CROP_PADDING, protein_img.shape[0])
        minc = max(center_c - CROP_PADDING, 0)
        maxc = min(center_c + CROP_PADDING, protein_img.shape[1])

        def crop(img): return img[minr:maxr, minc:maxc]

        protein_crop = crop(protein_img)
        membrane_crop = crop(membrane_img)
        ring_crop = crop((ring_mask & self.cell_mask).astype(np.uint8))
        cell_crop = crop(self.cell_mask.astype(np.uint8))

        def to_rgb(normed, channel="green", alpha=0.6):
            base = (normed * 255).astype(np.uint8)
            blank = np.zeros_like(base)
            if channel == "green":
                return np.stack([blank, (base * alpha).astype(np.uint8), blank], axis=-1)
            if channel == "red":
                return np.stack([(base * alpha).astype(np.uint8), blank, blank], axis=-1)

        def save_rgb(data, path):
            Image.fromarray(data).save(out_dir / path)

        save_rgb(to_rgb((protein_crop - protein_crop.min()) / (np.ptp(protein_crop) + 1e-6), "green"), f"cell_{self.cell_id:03d}_protein.png")
        save_rgb(to_rgb((membrane_crop - membrane_crop.min()) / (np.ptp(membrane_crop) + 1e-6), "red"), f"cell_{self.cell_id:03d}_membrane.png")

        # Overlay image
        base = Image.fromarray(to_rgb((protein_crop - protein_crop.min()) / (np.ptp(protein_crop) + 1e-6), "green"))
        overlay = base.convert("RGBA")
        draw = ImageDraw.Draw(overlay, "RGBA")

        for y, x in zip(*np.where(cell_crop)):
            draw.point((x, y), fill=(255, 255, 0, 60))  # yellow
        for y, x in zip(*np.where(ring_crop)):
            draw.point((x, y), fill=(255, 0, 255, 160))  # magenta

        overlay.save(out_dir / f"cell_{self.cell_id:03d}_overlay.png")
        log.debug(f"Saved per-cell debug overlay for cell {self.cell_id}")

    @classmethod
    def from_region(cls, region, label_img):
        mask = (label_img == region.label)
        return cls(cell_id=region.label, region=region, cell_mask=mask)