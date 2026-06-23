import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ─── Konfigurasi Augmentasi ───────────────────────────────────────────────────
INPUT_DIR = "/Users/fahmitaris/Downloads/Palm vein Eksperimen/preprocessed_results/1"
IMG_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg")

AUG_FLIP        = True          # Flip horizontal
AUG_ROT_POS     = 10            # Rotasi +10°
AUG_ROT_NEG     = -10           # Rotasi -10°
AUG_TRANSLATE   = 0.05          # Translate 5% dari ukuran gambar
AUG_BRIGHTNESS  = 0.15          # ±brightness factor


# ─── Fungsi Augmentasi ────────────────────────────────────────────────────────

def flip_horizontal(img):
    return cv2.flip(img, 1)


def rotate(img, angle):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def translate(img, tx_ratio, ty_ratio):
    h, w = img.shape[:2]
    tx = int(w * tx_ratio)
    ty = int(h * ty_ratio)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness(img, factor):
    """factor > 0: terang, factor < 0: gelap  (range ~0.15)"""
    img_float = img.astype(np.float32)
    img_float = img_float * (1 + factor)
    return np.clip(img_float, 0, 255).astype(np.uint8)


def augment_image(img):
    """Kembalikan dict berisi semua variasi augmentasi."""
    augs = {}
    augs["Original"]            = img.copy()
    augs["Flip Horizontal"]     = flip_horizontal(img)
    augs["Rotate +10°"]         = rotate(img,  AUG_ROT_POS)
    augs["Rotate -10°"]         = rotate(img,  AUG_ROT_NEG)
    augs["Translate 5%"]        = translate(img, AUG_TRANSLATE, AUG_TRANSLATE)
    augs["Brightness +0.15"]    = adjust_brightness(img,  AUG_BRIGHTNESS)
    augs["Brightness -0.15"]    = adjust_brightness(img, -AUG_BRIGHTNESS)
    return augs


# ─── Visualisasi ──────────────────────────────────────────────────────────────

def show_augmentations(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Tidak bisa membaca: {img_path}")
        return

    augs = augment_image(img)
    n_cols = len(augs)

    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))
    fig.suptitle(f"Augmentasi — {Path(img_path).name}", fontsize=13, fontweight="bold")

    for ax, (title, aug_img) in zip(axes, augs.items()):
        ax.imshow(aug_img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_all_images(input_dir: str):
    image_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])

    if not image_files:
        print(f"[ERROR] Tidak ada gambar di: {input_dir}")
        return

    print(f"Ditemukan {len(image_files)} gambar. Menampilkan augmentasi...\n")

    for img_path in image_files:
        print(f"  → {Path(img_path).name}")
        show_augmentations(img_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_all_images(INPUT_DIR)
