import gradio as gr
import torch
import torch.nn.functional as F
import requests
import numpy as np
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from torchvision.models import resnet50

from transformers import ViTImageProcessor
from transformers import ViTForImageClassification

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import json

MODEL_CHECKPOINT = "ckpts/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
OUTPUT_MODE = "binary_mask"


# 模型名稱
CLASSIFICATION_MODEL_DIR = "./models/vit-base-food101"
CLASSIFICATION_LABELS_PATH = "./models/classes.txt"

def load_segmentation_model():
    # 加載 FoodSeg 預訓練模型 (假設為 Torch 模型)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
    _ = sam.to(device="cuda")
    generator = SamAutomaticMaskGenerator(sam, output_mode=OUTPUT_MODE, **amg_kwargs)
    
    foodseg_model = torch.jit.load(MODEL)  # 替換成您的模型路徑
    foodseg_model.eval()
    
    return foodseg_model

# 讀取類別標籤
def load_labels(label_file):
    with open(label_file, "r") as f:
        label_list = [line.strip() for line in f.readlines()]
    return {i: label_list[i] for i in range(len(label_list))}

def load_classification_model():
    # 加載模型和特徵提取器
    print("📌 使用本地模型")
    model = ViTForImageClassification.from_pretrained(CLASSIFICATION_MODEL_DIR)
    feature_extractor = ViTImageProcessor.from_pretrained(CLASSIFICATION_LABELS_PATH)
        
    return model, feature_extractor

# 加載模型與特徵提取器
foodseg_model = load_segmentation_model()
food101_model, food101_model_feature_extractor = load_classification_model()

# 加載類別標籤
food101_classes = load_labels(CLASSIFICATION_LABELS_PATH)


def segment_food(image):
    """ 使用 FoodSeg 預訓練模型分割食物 """
    image = np.array(image)
    input_tensor = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        mask = foodseg_model(input_tensor)
    mask = (mask.squeeze().numpy() > 0.5).astype(np.uint8)  # 產生二值化遮罩
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image, mask

def preprocess_image(image_path_or_url, feature_extractor):
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        
    else:
        image = Image.open(image_path_or_url)
        
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def classify_food(segmented_image):
    """ 使用 Food101 模型對分割後的食物進行分類 """
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_tensor = transform(Image.fromarray(segmented_image)).unsqueeze(0)
    with torch.no_grad():
        output = food101_model(input_tensor)
    class_idx = output.argmax().item()
    food_class = food101_classes[str(class_idx)]
    return food_class

def get_nutrition_db():
    return {
        "pizza": {"calories": 266, "protein": 11, "fat": 10, "carbs": 33},
        "hamburger": {"calories": 295, "protein": 17, "fat": 14, "carbs": 30},
        "salad": {"calories": 152, "protein": 2, "fat": 10, "carbs": 14},
    }

def calculate_nutrition(mask, food_class):
    """ 計算食物像素比例，估算營養成分 """
    total_pixels = mask.size
    food_pixels = np.count_nonzero(mask)
    food_ratio = food_pixels / total_pixels

    food_nutrition_db = get_nutrition_db()
    nutrition = food_nutrition_db.get(food_class, {"calories": 0, "protein": 0, "fat": 0, "carbs": 0})
    nutrition_scaled = {k: v * food_ratio for k, v in nutrition.items()}
    return nutrition_scaled

def plot_nutrition(nutrition):
    """ 繪製營養分析圖表 """
    labels = nutrition.keys()
    values = nutrition.values()
    plt.figure(figsize=(5, 3))
    plt.bar(labels, values, color=['red', 'green', 'blue', 'orange'])
    plt.ylabel("Amount per 100g")
    plt.title("Estimated Nutrition Info")
    plt.tight_layout()
    plt.savefig("nutrition_plot.png")
    return "nutrition_plot.png"

def process_image(image):
    """ 結合所有步驟，處理食物照片 """
    segmented_image, mask = segment_food(image)
    food_class = classify_food(segmented_image)
    nutrition = calculate_nutrition(mask, food_class)
    plot_path = plot_nutrition(nutrition)

    return segmented_image, food_class, nutrition, plot_path

with gr.Blocks() as demo:
    gr.Markdown("## 食物影像分析")
    image_input = gr.Image(type="pil")
    segmented_output = gr.Image()
    label_output = gr.Textbox(label="Food Class")
    nutrition_output = gr.JSON(labe="Nutrition Info")
    plot_output = gr.Image()
    
    submit_btn = gr.Button("Analyze")
    
    submit_btn.click(
        process_image,
        inputs=image_input,
        outputs=[segmented_output, label_output, nutrition_output, plot_output]
    )
    
demo.launch()