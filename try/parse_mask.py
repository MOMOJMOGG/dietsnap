import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd


def apply_mask(original_image, mask):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary_mask = (mask == 255).astype(np.uint8)
    
    # 擴展為3通道
    binary_mask_3ch = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    masked_image = original_image * binary_mask_3ch  # 保留原圖像素，其他變0
    return masked_image

def filter_valid_mask(mask_info_path):
    df = pd.read_csv(mask_info_path)
    valid_masks = df[
        (df['category_id'] != 0) &
        (df['category_count_ratio'] > 0.9) &
        (df['mask_count_ratio'] > 0.1)
    ]
    return valid_masks

if __name__ == "__main__":
    input = "output/image/input.jpg"
    sam_mask_dir = "output/image/sam_mask"
    mask_info_path = "output/image/sam_mask_label/semantic_masks_category.txt"
    
    input_img = cv2.imread(input)
    
    valid_masks = filter_valid_mask(mask_info_path)
    
    for _, row in valid_masks.iterrows():
        mask_id = row['id']
        category = row['category_name']
        
        mask_path = os.path.join(sam_mask_dir, f"{mask_id}.png")
        mask_img = cv2.imread(mask_path)
        
        if mask_img is None:
            print(f"[!] Warning: Mask {mask_path} not found.")
            continue
        
        masked_img = apply_mask(input_img, mask_img)
        cv2.imshow("SHOW", masked_img)
        k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    