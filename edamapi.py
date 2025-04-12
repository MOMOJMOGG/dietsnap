from dotenv import load_dotenv
import os
import requests

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
    print(data)
    print(food_nutrients)
    
    #? ENERC_KCAL >> Energy (kcal) | 熱量
    #? PROCNT >> Protein (g) | 蛋白質
    #? FAT >> Total Fat (g) | 總脂肪
    #? CHOCDF >> Carbohydrate, by difference (g) | 碳水化合物
    #? FIBTG >> Fiber, total dietary (g) | 膳食纖維

get_food_data('bread')