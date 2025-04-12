
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification

from pathlib import Path
from PIL import Image
import requests
import torch
import torch.nn.functional as F

# 模型名稱
MODEL_NAME = "nateraw/vit-base-food101"

# 本地模型儲存路徑
MODEL_DIR = "./models/vit-base-food101"
LABELS_PATH = "./models/classes.txt"

# 檢查是否存在本地模型
USE_LOCAL_MODEL = Path(MODEL_DIR).exists()

# 讀取類別標籤
def load_labels(label_file):
    with open(label_file, "r") as f:
        label_list = [line.strip() for line in f.readlines()]
    return {i: label_list[i] for i in range(len(label_list))}

def load_models():
    # 加載模型和特徵提取器
    if USE_LOCAL_MODEL:
        print("📌 使用本地模型")
        model = ViTForImageClassification.from_pretrained(MODEL_DIR)
        feature_extractor = ViTImageProcessor.from_pretrained(MODEL_DIR)
        
    else:
        print("📌 下載並使用 Hugging Face 預訓練模型")
        model = ViTForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        
        # 保存到本地
        model.save_pretrained(MODEL_DIR)
        feature_extractor.save_pretrained(MODEL_DIR)
    
    return model, feature_extractor

def preprocess_image(image_path_or_url, feature_extractor):
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        
    else:
        image = Image.open(image_path_or_url)
        
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def predict_food(image_path_or_url, model, feature_extractor, food101_labels):
    inputs = preprocess_image(image_path_or_url, feature_extractor)
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(probs, dim=-1).item()
    predicted_label = food101_labels[predicted_class_idx]
    confidence = probs[0, predicted_class_idx].item()
    
    print(f"🛠 預測類別索引: {predicted_class_idx}")
    print(f"🍽 預測食物類別: {predicted_label} (置信度: {confidence:.4f})")
    return predicted_label, confidence

# 預測食物類別, 返回前5個最大置信度類別
def predict_top5_food(image_path_or_url, model, feature_extractor, food101_labels):
    inputs = preprocess_image(image_path_or_url, feature_extractor)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)  # 歸一化概率

    # 取得前5個最大置信度類別
    top5_probs, top5_class_idxs = torch.topk(probs, k=5, dim=-1)

    # 轉換索引到實際類別名稱
    top5_results = [(food101_labels[idx.item()], top5_probs[0][i].item()) for i, idx in enumerate(top5_class_idxs[0])]

    print(f"📷 圖片: {image_path_or_url}")
    for rank, (label, conf) in enumerate(top5_results, start=1):
        print(f"🥇 Top {rank}: {label} (置信度: {conf:.4f})")

    return top5_results

if __name__ == "__main__":
    # 加載類別標籤
    food101_labels = load_labels(LABELS_PATH)
    
    # 加載模型與特徵提取器
    model, feature_extractor = load_models()
    
    for i in range(6):
        image_url = f"testing/{i}.jpg"
        predict_top5_food(image_url,
                          model,
                          feature_extractor,
                          food101_labels)
        