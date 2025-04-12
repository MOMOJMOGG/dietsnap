import cv2
import gradio as gr
import numpy as np
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import requests
import time

from FoodSAM.semantic import main
from FoodSAM.semantic import args

# MODEL_DIR   = "./models/vit-base-food101"
# SAFETENSOR  = "./models/vit-base-food101/model.safetensors"
# LABELS_PATH = "./models/classes.txt"

SAM_INPUT_IMG  = "./output/image/input.jpg"
SAM_MASK_DIR   = "./output/image/sam_mask"
MASK_INFO_PATH = "./output/image/sam_mask_label/semantic_masks_category.txt"
SAM_MAP_DIR = "./output/image/sam_map"

def request_food101_prediction(masked_img_folder):
    url = "http://localhost:5001/classify"
    response = requests.post(url, json={"folder": masked_img_folder})
    
    if response.status_code == 200:
        results = response.json()
        for r in results:
            print(f"{r['img']} → {r['label']}, {r['conf']}")
        return results
    
    else:
        print("Error:", response.text)
        return None
    
# class Classification:
    
#     def __init__(self):
#         global MODEL_DIR
#         global LABELS_PATH
#         global SAFETENSOR
        
#         self.model_dir = MODEL_DIR
#         self.labels_path = LABELS_PATH
#         self.safetensor = SAFETENSOR
        
#         self.model = None
#         self.feature_extractor = None
#         self.labels = None
        
#         self.load_model()
#         self.load_labels()
        
#     def load_model(self):
#         # 加載模型和特徵提取器
#         if Path(self.model_dir).exists():
#             print("📌 使用本地模型")
#             # 讀 config
#             config = ViTConfig.from_pretrained(self.model_dir)
#             self.model = ViTForImageClassification(config)
            
#             # 建立模型結構
#             state_dict = load_file(self.safetensor)
#             self.model.load_state_dict(state_dict)
            
#             self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_dir)
        
#         else:
#             print("📌 下載並使用 Hugging Face 預訓練模型")

#     def load_labels(self):
#         with open(self.labels_path, "r") as f:
#             label_list = [line.strip() for line in f.readlines()]
        
#         self.labels = {i: label_list[i] for i in range(len(label_list))}
        
#     def preprocess_image(self, image_path):
#         image = Image.open(image_path)
#         inputs = self.feature_extractor(images=image, return_tensors="pt")
#         return inputs
    
#     def predict_food(self, image_path):
#         inputs = self.preprocess_image(image_path)
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         logits = outputs.logits
#         probs = F.softmax(logits, dim=-1)
#         predicted_class_idx = torch.argmax(probs, dim=-1).item()
#         predicted_label = self.labels[predicted_class_idx]
#         confidence = probs[0, predicted_class_idx].item()
        
#         print(f"🛠 預測類別索引: {predicted_class_idx}")
#         print(f"🍽 預測食物類別: {predicted_label} (置信度: {confidence:.4f})")
#         return predicted_label, confidence

#     def predict_top5_food(self, image_path):
#         inputs = self.preprocess_image(image_path)
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         logits = outputs.logits
#         probs = F.softmax(logits, dim=-1)  # 歸一化概率
        
#         # 取得前5個最大置信度類別
#         top5_probs, top5_class_idxs = torch.topk(probs, k=5, dim=-1)
    
#         # 轉換索引到實際類別名稱
#         top5_results = [(self.labels[idx.item()], top5_probs[0][i].item()) for i, idx in enumerate(top5_class_idxs[0])]

#         print(f"📷 圖片: {image_path}")
#         for rank, (label, conf) in enumerate(top5_results, start=1):
#             print(f"🥇 Top {rank}: {label} (置信度: {conf:.4f})")
        
#         return top5_results

class Segmentation:
    
    def __init__(self):
        global SAM_INPUT_IMG
        global SAM_MASK_DIR
        global MASK_INFO_PATH
        global SAM_MAP_DIR
        
        self.sam_input_img  = SAM_INPUT_IMG
        self.sam_mask_dir   = SAM_MASK_DIR
        self.mask_info_path = MASK_INFO_PATH
        self.sam_map_dir = SAM_MAP_DIR
    
    def apply_mask(self, original_image, mask):
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary_mask = (mask == 255).astype(np.uint8)
        
        # 擴展為3通道
        binary_mask_3ch = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        masked_image = original_image * binary_mask_3ch  # 保留原圖像素，其他變0
        return masked_image

    @staticmethod
    def filter_valid_mask(df, mask_ratio):
        valid_masks = df[
            (df['category_id'] != 0) &
            (df['category_count_ratio'] > 0.9) &
            (df['mask_count_ratio'] > mask_ratio)
        ]
        return valid_masks
    
    def parse_valid_masks(self, valid_masks):
        input_img = cv2.imread(self.sam_input_img)
        
        if not Path(self.sam_map_dir).exists():
            os.mkdir(self.sam_map_dir)
        
        masked_img_list = []
        for _, row in valid_masks.iterrows():
            mask_id = row['id']
            category = row['category_name']
            
            mask_path = os.path.join(self.sam_mask_dir, f"{mask_id}.png")
            save_path = os.path.join(self.sam_map_dir,  f"{mask_id}.png")
            mask_img = cv2.imread(mask_path)
            
            if mask_img is None:
                print(f"[!] Warning: Mask {mask_path} not found.")
                continue
            
            masked_img = self.apply_mask(input_img, mask_img)
            cv2.imwrite(save_path, masked_img)
            masked_img_list.append(save_path)
        
        return masked_img_list

    def group_classify_results(self, classify_results):
        img_list = []
        group_results = []
        for data in classify_results:
            img = cv2.imread(data['img'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # group_results.append((img, data['label'], data['conf']))
            img_list.append(img)
            group_results.append((data['img'], data['label'], data['conf']))
        
        return img_list, group_results

segment    = Segmentation()

def process_img(img, mask_ratio):
    start = time.time()
    print(img)
    if main(args, img):
        end = time.time()
        result_img  = "output/image/enhance_vis.png"
        result_file = "output/image/sam_mask_label/semantic_masks_category.txt"
        df = pd.read_csv(result_file)
        valid_mask = Segmentation.filter_valid_mask(df, mask_ratio)
        masked_list = segment.parse_valid_masks(valid_mask)
        
        classify_results = request_food101_prediction(segment.sam_map_dir)
        img_list, classify_results = segment.group_classify_results(classify_results)
        print(classify_results)
        
        time_cost = f"Cost: {round(end - start, 3)} s"
        
        return result_img, df, valid_mask, img_list, classify_results, time_cost
    
    else:
        return None, "預測結果: NaN"

with gr.Blocks(css="""
        .gradio-container { max-width: 1400px !important; }
        .gallery-img img { max-height: 360px !important; max-width: 360px !important; object-fit: contain; }
    """) as dietsnap:
    gr.Markdown("## 🍱 DietSnap 食物辨識展示")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="filepath", label="📤 上傳圖片")
                    mask_ratio = gr.Number(0.01, label="分割比例")
                    submit_btn = gr.Button("🚀 開始辨識", variant="primary")
            
            with gr.Row():
                seg_img = gr.Image(type="numpy", label="🖼️ 分割圖示", show_label=True)
                
            with gr.Row():
                cost = gr.Text(label="⏱️ 處理耗時")
                
        with gr.Column(scale=2):
            
            with gr.Row():
                with gr.Column(scale=1):
                    gallery = gr.Gallery(label="🔍 預覽圖片", elem_classes=["gallery-img"])
                    
                with gr.Column(scale=1):
                    cls_table = gr.Dataframe(
                        headers=["圖片路徑", "類別", "信心度"],
                        datatype=["str", "str", "number"],
                        row_count=None,
                        label="📊 分類結果"
                    )
                    
            with gr.Row():
                with gr.Column(scale=1):
                    seg_table = gr.Dataframe(label="📋 分割結果", interactive=False)
                        
                with gr.Column(scale=1):
                        filter_table = gr.Dataframe(label="📋 過濾結果", interactive=False)
     
                
    # 隱藏 flag
    dietsnap.flagging = None
    input_img.change(fn=process_img, inputs=[input_img, mask_ratio], 
                     outputs=[seg_img, seg_table, filter_table, gallery, cls_table, cost])
    
dietsnap.launch()
# dietsnap = gr.Interface(fn=process_img,
#                         inputs=gr.Image(type="filepath", label="上傳圖片"),
#                         outputs=[gr.Image(type="numpy", label="分割圖示"),
#                                  gr.Dataframe(label="分割結果"),
#                                  gr.Dataframe(label="過濾結果"),
#                                  gr.Gallery(label="預覽圖片", height="auto"),
#                                  gr.Dataframe(
#                                      headers=["圖片路徑", "類別", "信心度"],
#                                      datatype=["str", "str", "number"],
#                                      row_count=None,
#                                      label="分類結果"
#                                  ),
#                                  gr.Text(label="耗時")],
#                         title="DietSnap 食物辨識初步展示")
