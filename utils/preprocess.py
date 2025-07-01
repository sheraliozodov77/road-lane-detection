"""
utils/preprocess.py

Preprocessing script to:
- Match lane detection image–mask pairs from driver_161_90frame
- Resize both to 448x256 resolution
- Binarize lane masks
- Split into train/val/test sets
- Save organized datasets to `data/processed/`
"""

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
IMG_ROOT = 'data/driver_161_90frame'
MASK_ROOT = 'data/laneseg_label_w16'
OUT_ROOT = 'data/processed'
IMG_SIZE = (448, 256)

def ensure_dirs():
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{OUT_ROOT}/{split}/images', exist_ok=True)
        os.makedirs(f'{OUT_ROOT}/{split}/masks', exist_ok=True)

def collect_pairs():
    print(" Collecting matching image and mask pairs...")
    image_paths, mask_paths = [], []
    for img_path in tqdm(glob.glob(os.path.join(IMG_ROOT, '**/*.jpg'), recursive=True)):
        rel_path = os.path.relpath(img_path, IMG_ROOT)
        mask_path = os.path.join(MASK_ROOT, rel_path).replace('.jpg', '.png')
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    return image_paths, mask_paths

def resize_and_save(src_img, src_mask, dst_img, dst_mask):
    img = cv2.imread(src_img)
    mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, IMG_SIZE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = (mask > 0).astype(np.uint8) * 255

    cv2.imwrite(dst_img, img)
    cv2.imwrite(dst_mask, mask)

def preprocess():
    ensure_dirs()
    image_paths, mask_paths = collect_pairs()
    print(f" Total valid pairs found: {len(image_paths)}")

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=0.3, random_state=42
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=0.5, random_state=42
    )

    for split, imgs, masks in [
        ('train', train_imgs, train_masks),
        ('val', val_imgs, val_masks),
        ('test', test_imgs, test_masks)
    ]:
        print(f"\n Processing {split} set: {len(imgs)} samples")
        for img_path, mask_path in tqdm(zip(imgs, masks), total=len(imgs)):
            rel_path = os.path.relpath(img_path, IMG_ROOT)
            filename = rel_path.replace(os.sep, '_')  # e.g., "06030819_0755.MP4_00000.jpg"
            dst_img = os.path.join(f'{OUT_ROOT}/{split}/images', filename)
            dst_mask = os.path.join(f'{OUT_ROOT}/{split}/masks', filename.replace('.jpg', '.png'))
            resize_and_save(img_path, mask_path, dst_img, dst_mask)

    print("\n Preprocessing complete! Processed images saved to 'data/processed/'")

if __name__ == '__main__':
    preprocess()
