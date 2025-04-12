
import cv2
from dotenv import load_dotenv
import gradio as gr
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
import numpy as np
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import requests
import time

from FoodSAM.semantic import main
from FoodSAM.semantic import args

SAM_INPUT_IMG  = "./output/image/input.jpg"
SAM_MASK_DIR   = "./output/image/sam_mask"
MASK_INFO_PATH = "./output/image/sam_mask_label/semantic_masks_category.txt"
SAM_MAP_DIR    = "./output/image/sam_map"
SAM_CATEGORY_ID_FILE = "./FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt"


# 讀取 .env
load_dotenv()

# 取得 APP ID, KEYS
app_id = os.getenv("EDAMAM_ID")
app_key = os.getenv("EDAMAM_KEYS")

def get_food_data(food):
    url = f"https://api.edamam.com/api/food-database/v2/parser?app_id={app_id}&app_key={app_key}&ingr={food}"
    response = requests.get(url)
    data = response.json()
    if "hints" not in data:
        return f"No data found for {food}"

    food_nutrients = data['parsed'][0]['food']['nutrients']
    print(food_nutrients)
    return food_nutrients
    
class Segmentation:
    
    def __init__(self):
        global SAM_INPUT_IMG
        global SAM_MASK_DIR
        global MASK_INFO_PATH
        global SAM_MAP_DIR
        
        self.sam_input_img  = SAM_INPUT_IMG
        self.sam_mask_dir   = SAM_MASK_DIR
        self.mask_info_path = MASK_INFO_PATH
        self.sam_map_dir    = SAM_MAP_DIR
        self.sam_category_id_file = SAM_CATEGORY_ID_FILE
        
        self.category_list = []
        self.load_category_id()
    
    def load_category_id(self):
        with open(self.sam_category_id_file, 'r') as f:
            category_lines = f.readlines()
            self.category_list = [' '.join(line.split('\t')[1:]).strip() for line in category_lines]
    
    def get_class_name(self, class_id):
        if class_id < len(self.category_list):
            return self.category_list[class_id]
    
        return f"unknown_{class_id}"
    
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
        h, w = input_img.shape[:2]
        area = h * w
        
        # 計算 mask_count 欄位：mask_count_ratio × area
        if 'mask_count_ratio' in valid_masks.columns:
            valid_masks['mask_count'] = valid_masks['mask_count_ratio'] * area
        else:
            raise ValueError("valid_masks 必須包含欄位 'mask_count_ratio'")
    
        if not Path(self.sam_map_dir).exists():
            os.mkdir(self.sam_map_dir)
        
        masked_img_list = []
        for _, row in valid_masks.iterrows():
            mask_id  = row['id']
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
            
        
        return masked_img_list, valid_masks
    
    def parse_result_mask(self, mask_img_path):
        mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        h, w= mask_img.shape[:2]
        total_area = h * w
        
        unique, counts = np.unique(mask_img, return_counts=True)
        pixel_stats = dict(zip(unique, counts))
        
        records = []
        total_pixels = 0
        for class_id, pixel_count in pixel_stats.items():
            class_name = self.get_class_name(class_id)
            ratio = pixel_count / total_area
            
            if class_name != 'background':
                records.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'pixel_count': pixel_count,
                    'ratio': round(ratio, 3)
                })
                total_pixels += pixel_count
                
        return records, total_pixels

    def calc_food_weight(self, pixel_count, total_pixels, assumed_weight):
        return round((pixel_count / total_pixels) * assumed_weight / 100, 3)
        
    def analyze_nutrition(self, records, total_pixels, assumed_total_weight):
        results = []
        final = {
            'food': 'total',
            'energy': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'carbohydrate': 0.0,
            'fiber': 0.0
        }
        for record in records:
            food = record['class_name']
            food_weight = self.calc_food_weight(record['pixel_count'], total_pixels, assumed_total_weight)
            
            nutrient = get_food_data(food)
            food_nutrient = {
                'food': food,
                'energy': round(nutrient['ENERC_KCAL'] * food_weight, 1),
                'protein': round(nutrient['PROCNT'] * food_weight, 1),
                'fat': round(nutrient['FAT'] * food_weight, 1),
                'carbohydrate': round(nutrient['CHOCDF'] * food_weight, 1),
                'fiber': round(nutrient['FIBTG'] * food_weight, 1)
            }
            
            final['energy'] += food_nutrient['energy']
            final['protein'] += food_nutrient['protein']
            final['fat'] += food_nutrient['fat']
            final['carbohydrate'] += food_nutrient['carbohydrate']
            final['fiber'] += food_nutrient['fiber']
        
            results.append(food_nutrient)
            
        final['energy'] = round(final['energy'], 1)
        final['protein'] = round(final['protein'], 1)
        final['fat'] = round(final['fat'], 1)
        final['carbohydrate'] = round(final['carbohydrate'], 1)
        final['fiber'] = round(final['fiber'], 1)
        results.append(final)
        return results
    
    def plot_nutrition_bar(self, nutrition_df):
        # 分開資料
        energy_df = nutrition_df[['food', 'energy']].set_index('food')
        rest_df = nutrition_df.set_index('food')[['protein', 'fat', 'carbohydrate', 'fiber']]
        
        # 建立子圖畫布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左圖：Energy
        energy_df.plot(kind="bar", ax=ax1, legend=False, color="orange")
        ax1.set_title("各食物熱量 (Energy)")
        ax1.set_ylabel("kcal")
        ax1.set_xlabel("食物")
        ax1.grid(axis='y')
        
        # 右圖：其餘營養素
        rest_df.plot(kind="bar", ax=ax2)
        ax2.set_title("營養成分分析（蛋白質、脂肪、碳水、纖維）")
        ax2.set_ylabel("含量 (g)")
        ax2.set_xlabel("營養成分")
        ax2.grid(axis='y')
        ax2.legend(title="食物")
        
        plt.tight_layout()
        plot_path = "./results/nutrition_plot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        return plot_path

segment = Segmentation()

def clear_output_folder(folder_path="output"):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 刪除檔案或符號連結
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 刪除資料夾
            except Exception as e:
                print(f"無法刪除 {file_path}: {e}")
                       
def process_img(img, mask_ratio, food_weight):
    start = time.time()
    if main(args, img):
        end = time.time()
        result_img  = "output/image/enhance_vis.png"
        result_mask = "output/image/enhance_mask.png"
        result_file = "output/image/sam_mask_label/semantic_masks_category.txt"
        df = pd.read_csv(result_file)
        # valid_mask = Segmentation.filter_valid_mask(df, mask_ratio)
        # masked_list, parsed_df = segment.parse_valid_masks(valid_mask)
        
        segment_records, total_pixels = segment.parse_result_mask(result_mask)
        food_nutrition = segment.analyze_nutrition(segment_records, total_pixels, food_weight)
        total_nutrition = food_nutrition[-1]
        nutrition_df = pd.DataFrame(food_nutrition)
        plt_path = segment.plot_nutrition_bar(nutrition_df)
        
        time_cost = f"Cost: {round(end - start, 3)} s"
        
        return (result_img, df, nutrition_df, plt_path, time_cost,
                total_nutrition['energy'], total_nutrition['protein'],
                total_nutrition['fat'], total_nutrition['carbohydrate'],
                total_nutrition['fiber'])
        # return result_img, df, valid_mask, masked_list, parsed_df, time_cost
    
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
                    input_img   = gr.Image(type="filepath", label="📤 上傳圖片")
                    mask_ratio  = gr.Number(0.01, label="分割比例")
                    food_weight = gr.Number(400, label="食物預估總克數")
                    submit_btn  = gr.Button("🚀 開始辨識", variant="primary")
            
            with gr.Row():
                seg_img = gr.Image(type="numpy", label="🖼️ 分割圖示", show_label=True)
                
            with gr.Row():
                cost = gr.Text(label="⏱️ 處理耗時")
                
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column(scale=1):
                    nutrition_img = gr.Image(type="numpy", label="📊 營養成分分析", show_label=True)
                # with gr.Column(scale=1):
                #     gallery = gr.Gallery(label="🔍 預覽圖片", elem_classes=["gallery-img"])
                    
                # with gr.Column(scale=1):
                #     parse_table = gr.Dataframe(label="📊 營養標示結果", interactive=False)
                    
            with gr.Row():
                with gr.Column(scale=1):
                    food_energy  = gr.Number(label="熱量 (kcal)")
                    food_protein = gr.Number(label="蛋白質 (g)")
                    food_fat   = gr.Number(label="脂肪 (g)")
                    food_carb  = gr.Number(label="碳水 (g)")
                    food_fiber = gr.Number(label="纖維 (g)")
                        
                with gr.Column(scale=1):
                    parse_table = gr.Dataframe(label="📊 營養標示結果", interactive=False)
                    # filter_table = gr.Dataframe(label="📋 過濾結果", interactive=False)
                    seg_table = gr.Dataframe(label="📋 分割結果", interactive=False)
          
    # 隱藏 flag
    dietsnap.flagging = None
    input_img.change(fn=process_img, inputs=[input_img, mask_ratio, food_weight], 
                     outputs=[seg_img, seg_table, parse_table, nutrition_img, cost,
                              food_energy, food_protein, food_fat, food_carb, food_fiber])
                    #  outputs=[seg_img, seg_table, filter_table, gallery, parse_table, cost])
    
dietsnap.launch()