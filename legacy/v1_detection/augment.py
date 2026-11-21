import os
import cv2
import numpy as np
import random

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- What this script does ----
# Takes each image and creates multiple transformed versions
# This artificially expands our small dataset into a balanced one

INPUT = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "data", "augmented")

# How many augmented versions to create per image
# We set different targets per class to balance them out
TARGETS = {
    "impact_beating": 10,  # 61 × 10 = 610 images
    "gunshot": 50,         # 8  × 50 = 400 images
    "drip": 1              # 1032 × 1 = 1032 images (already enough)
}

# ── AUGMENTATION FUNCTIONS ──────────────────────────────────────────

def rotate(img, angle):
    """Rotate image by given angle around its center"""
    # Get image dimensions
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix — think of it as a mathematical recipe
    # that says "rotate every pixel by this angle around this point"
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))

def flip(img, direction):
    """
    Flip image like a mirror
    direction 0  = vertical flip (upside down)
    direction 1  = horizontal flip (left-right mirror)
    direction -1 = both
    """
    return cv2.flip(img, direction)

def adjust_brightness(img, factor):
    """
    Change brightness by multiplying pixel values
    factor > 1 = brighter
    factor < 1 = darker
    Pixel values are capped at 255 (pure white)
    """
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def add_noise(img):
    """
    Add random noise — simulates camera grain or poor lighting
    This makes the model robust to imperfect crime scene photos
    """
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)

def zoom(img, factor):
    """
    Crop center and resize back — simulates zooming in
    factor = 0.8 means we keep the central 80% of the image
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    # Calculate crop coordinates to keep center
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h))  # resize back to 640x640

def blur(img):
    """
    Apply slight Gaussian blur — simulates out of focus photos
    (5,5) is the kernel size — bigger = more blur
    """
    return cv2.GaussianBlur(img, (5, 5), 0)

# ── LIST OF ALL TRANSFORMS ──────────────────────────────────────────
# Each entry is a (name, function) pair
# We'll randomly pick from these to create each augmented version

def get_transforms():
    return [
        ("rot90",    lambda img: rotate(img, 90)),
        ("rot180",   lambda img: rotate(img, 180)),
        ("rot270",   lambda img: rotate(img, 270)),
        ("fliph",    lambda img: flip(img, 1)),
        ("flipv",    lambda img: flip(img, 0)),
        ("bright",   lambda img: adjust_brightness(img, 1.3)),
        ("dark",     lambda img: adjust_brightness(img, 0.7)),
        ("noise",    lambda img: add_noise(img)),
        ("zoom",     lambda img: zoom(img, 0.85)),
        ("blur",     lambda img: blur(img)),
        ("rot45",    lambda img: rotate(img, 45)),
        ("rot135",   lambda img: rotate(img, 135)),
    ]

# ── MAIN AUGMENTATION LOOP ──────────────────────────────────────────

for category, num_augments in TARGETS.items():
    
    input_folder  = os.path.join(INPUT, category)
    output_folder = os.path.join(OUTPUT, category)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images in this category
    images = [f for f in os.listdir(input_folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n{category}: {len(images)} original images")
    print(f"  Creating {num_augments} augmented versions each...")
    
    saved = 0
    
    for img_file in images:
        # Read the original image
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Always save the original copy first
        base_name = os.path.splitext(img_file)[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.jpg"), img)
        saved += 1
        
        # Get all transforms and shuffle them randomly
        transforms = get_transforms()
        random.shuffle(transforms)
        
        # Apply num_augments transforms to this image
        for i in range(min(num_augments, len(transforms))):
            transform_name, transform_fn = transforms[i]
            
            # Apply the transformation
            augmented = transform_fn(img)
            
            # Save with descriptive name so we know what was applied
            out_name = f"{base_name}_{transform_name}.jpg"
            cv2.imwrite(os.path.join(output_folder, out_name), augmented)
            saved += 1
    
    print(f"  Done! {saved} total images saved → {output_folder}")

print("\n✅ Augmentation complete!")
print(f"Check your data/augmented/ folder")