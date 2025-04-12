# 🥗 DietSnap - 智慧食物分割與營養分析平台 | Smart Food Segmentation and Nutrition Analysis Platform

## 📌 專案介紹 | Project Overview
**DietSnap** 是一套結合影像分割與營養分析的智慧飲食應用，透過上傳一張餐點照片，即可自動辨識其中的食材並推算出其營養資訊。
DietSnap is a smart dietary tool that combines food image segmentation with nutrition analysis. Upload a photo of a meal, and the system automatically detects food types and estimates their nutritional contents.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ✨ 專案亮點 | Highlights
- ✅ 食物分割模型整合 FoodSAM
- ✅ 利用像素面積估算重量
- ✅ 連接 Edamam API 自動查詢營養成分
- ✅ 支援可視化營養圖表輸出
- ✅ 支援 Gradio 視覺化展示

- ✔️ Integrated with FoodSAM model for accurate segmentation
- ✔️ Estimate food weight by pixel proportion
- ✔️ Automatic nutrient lookup using Edamam Nutrition API
- ✔️ Dual-style nutrition visualization: energy + macros
- ✔️ Interactive Gradio interface for demo and testing

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🧩 系統架構 | System Architecture
```
輸入圖片 → FoodSAM 分割 → 語義遮罩合併
       ↓                         ↓
  pixel 統計 → 重量估算 → Edamam API → 營養資料整理
                                     ↓
                          圖表輸出 + 表格呈現 (Gradio)
```
```
Input image → FoodSAM segmentation → mask merge
       ↓                        ↓
Pixel ratio → weight estimation → Edamam API → nutrition analysis
                                     ↓
             Chart & table output via Gradio interface
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🔁 使用流程 | Workflow
1. 上傳餐點圖片 Upload a meal image
2. 執行食物分割 Segmentation with FoodSAM
3. 統計像素面積並估算重量 Estimate food weight from pixel ratio
4. 傳送食物名稱與重量至 Edamam API
5. 取得各食材之營養成分 Get nutrition data from Edamam
6. 顯示圖表與表格分析結果 Display analysis in charts & tables

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📦 套件使用 | Key Packages
這部分列出了專案中使用的主要框架和函式庫。

This section lists the major frameworks and libraries used to bootstrap your project.

- `opencv-python`： 影像處理
- `numpy / pandas`： 統計與表格管理
- `matplotlib`： 圖表繪製
- `requests / dotenv`： API 串接與環境變數管理
- `gradio`： 前端 UI 展示
- `FoodSAM`: 食物分割模型

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 💻 安裝說明 | Installation
```bash
# 建立虛擬環境 (建議)
# Python 版本: 3.7.16
python -m venv venv


source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝必要套件
pip install -r requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🛠️ 前置說明 | Prerequisites
1. 你需要一個 [Edamam Nutrition API](https://www.edamam.com/) 的帳號與 API 金鑰。[安裝教學](https://molly1024.medium.com/edamam-api-%E6%95%99%E5%AD%B8-%E8%BC%95%E9%AC%86%E6%8E%8C%E6%8F%A1%E9%A3%9F%E7%89%A9%E5%88%86%E6%9E%90%E7%9A%84%E6%8A%80%E5%B7%A7-%E5%88%86%E5%88%A5%E7%94%A8-python-%E5%8F%8A-nodejs-%E4%BD%9C%E7%AF%84%E4%BE%8B-8779403703f)
2. 安裝好 PyTorch 與 [FoodSAM](https://github.com/jamesjg/FoodSAM) 相依環境（若使用分割功能）。

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🔧 設定說明 | Configuration
請在根目錄建立 `.env` 檔案，內容如下：
```
EDAMAM_APP_ID=your_app_id
EDAMAM_APP_KEY=your_app_key
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🚀 啟動說明 | Getting Started
```bash
# 啟動 Gradio 介面
python app.py
```
進入後即可上傳圖片並檢視分析結果。

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📁 專案結構 | Project Structure
```
DietSnap/
├── FoodSAM/                 # 食物分割模型運行代碼
├── mmseg/                   # 食物分割模型運行必要套件
├── try/                     # 開發用腳本 (開發用)
├── app.py                   # Gradio UI 主程式
├── classification_app.py    # Food101分類模型運行代碼 (開發用)
├── classify_flask.py        # 分類 API 伺服器 (開發用)
├── segment_with_flask.py    # 食物分割搭配食物分類腳本 (開發用)
├── requirements.txt         # 安裝環境套件
└── .env.example             # 設定檔範本
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📸 專案展示 | Showcase
![Image][demo-url-01]

> 左圖為每類食物熱量，右圖為蛋白質、脂肪、碳水、纖維等營養指標分布

> Left: Energy by food; Right: Protein, Fat, Carbs, Fiber breakdown


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[demo-url-01]: https://github.com/MOMOJMOGG/dietsnap/blob/main/demo/final_demo.png
