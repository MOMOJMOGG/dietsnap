import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'

import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
from torch.utils.data import random_split


dataset = Food101(root="./data", download=True, transform=transforms.ToTensor())

print(f"數據集: {dataset}")
print(f"數據集總數量: {len(dataset)}")
print(f"類別數量: {len(dataset.classes)}")


label_file = "data/food-101/meta/labels.txt"

labels = []

with open(label_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        labels.append(line)

class_counts = {}
for _, label in dataset:
    key = labels[label]
    class_counts[key] = class_counts.get(key, 0) + 1
    
    
df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
df = df.sort_values(by="Count", ascending=False)

print(df.head(10))

plt.figure(figsize=(20, 8))
sns.barplot(x=df["Class"][:10], y=df["Count"][:10])
# sns.barplot(x=df["Class"], y=df["Count"])
plt.xticks(rotation=45)
plt.title("Food101 數據集中前 10 種食物的數量")
# plt.title("Food101 數據集中全食物的數量")
plt.xlabel("食物類別")
plt.ylabel("圖片數量")
plt.show()
plt.savefig("food_count.png") 