import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import regionprops
from pathlib import Path
from PIL import Image, ImageDraw
from pipeline.logger import get_logger
from config import CROP_PADDING, RING_OVERLAY_ALPHA

log = get_logger(__name__)

def tile_images(images, rows, cols):
    if not images:
        raise ValueError("No images to tile.")

    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(img, (x, y))

    return grid

def overlay_rings(protein_img, membrane_img, cell_labels, ring_mask, out_dir="debug_overlays"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    overlay_path = Path(out_dir) / "overlay.png"

    protein_norm = (protein_img - protein_img.min()) / (np.ptp(protein_img) + 1e-6)
    membrane_norm = (membrane_img - membrane_img.min()) / (np.ptp(membrane_img) + 1e-6)

    base_rgb = np.stack([
        protein_norm,                  # R = protein
        membrane_norm,                # G = membrane
        np.zeros_like(protein_norm)   # B = none
    ], axis=-1)

    # Apply magenta ring overlay
    base_rgb[ring_mask] = 0.6 * base_rgb[ring_mask] + 0.4 * np.array([1.0, 0.0, 1.0])

    overlay = label2rgb(cell_labels, image=base_rgb, alpha=0.2, bg_label=0)

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title("Protein + Membrane + Rings + Cells")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200)
    plt.close()
    log.info(f"Saved global overlay to {overlay_path}")


def save_ring_crop(protein_img, ring_mask, cell_mask, out_path, overlay=True, padding=CROP_PADDING, label_text=None, label_color=(255, 255, 255)):
    """
    Saves a cropped image centered on a cell, showing only its own ring mask.
    """
    # Get bounding box and center from the cell mask
    
    props = regionprops(cell_mask.astype(np.uint8))
    if not props:
        print("No region found in mask")
        return

    minr, minc, maxr, maxc = props[0].bbox
    center_r = (minr + maxr) // 2
    center_c = (minc + maxc) // 2

    # Expand to centered crop
    minr = max(center_r - padding, 0)
    maxr = min(center_r + padding, protein_img.shape[0])
    minc = max(center_c - padding, 0)
    maxc = min(center_c + padding, protein_img.shape[1])

    # Get local crop
    crop = protein_img[minr:maxr, minc:maxc]
    norm_crop = (crop - crop.min()) / (np.ptp(crop) + 1e-6)

    # Green base image
    green_img = np.zeros((crop.shape[0], crop.shape[1], 3), dtype=np.uint8)
    green_img[..., 1] = (norm_crop * 255).astype(np.uint8)

    img = Image.fromarray(green_img)

    if overlay:
        # Ring + cell intersection to isolate this cell’s ring only
        ring_crop = (ring_mask & cell_mask)[minr:maxr, minc:maxc]
        draw = ImageDraw.Draw(img, "RGBA")
        for y, x in zip(*np.where(ring_crop)):
            draw.point((x, y), fill=(255, 0, 255, RING_OVERLAY_ALPHA))
        if label_text:
            draw.text((5, 5), label_text, fill=label_color + (255,))

    img.save(out_path)
    print(f"Saved per-cell crop to {out_path}")