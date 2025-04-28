# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# FastAPIインスタンス作成
app = FastAPI()

# モデルとSHAP explainerをロード
model = joblib.load("model.pkl")

# 🔥 日本語カラム名マッピング
feature_name_mapping = {
    "GenHlth": "自覚的健康状態",
    "HighBP": "高血圧の有無",
    "Age": "年齢区分",
    "BMI": "体格指数（BMI）",
    "HighChol": "高コレステロールの有無"
}

# リクエストデータ用モデル
class InputData(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

# 予測エンドポイント
@app.post("/predict")
def predict(data: InputData):
    # 入力データをnumpy配列に変換
    input_array = np.array([[getattr(data, field) for field in data.__fields__]])

    # 予測
    pred_proba = model.predict_proba(input_array)[0][1]
    pred_label = model.predict(input_array)[0]

    # メッセージ生成
    if pred_label == 1:
        advice_message = f"""糖尿病予備軍または糖尿病のリスクが高い傾向が見られました。
特に以下の要素がリスクに関与している可能性があります：

- {feature_name_mapping['GenHlth']}: 自覚的健康状態が良くない可能性（1=最高、5=最低）
- {feature_name_mapping['HighBP']}: 高血圧がある場合はリスク上昇
- {feature_name_mapping['Age']}: 年齢が高くなるほどリスク上昇
- {feature_name_mapping['BMI']}: BMI（体格指数）が高い場合はリスク上昇（25以上で要注意）
- {feature_name_mapping['HighChol']}: 高コレステロールがある場合はリスク上昇

➡ 生活習慣（運動、食事、体重管理）、血圧・コレステロールの管理に取り組み、必要に応じて専門医にご相談ください。
"""
    else:
        advice_message = "現在のところ糖尿病リスクは高くないと推測されますが、引き続き健康管理に努めましょう。"

    return {
        "prediction": int(pred_label),
        "probability": float(pred_proba),
        "advice": advice_message
    }