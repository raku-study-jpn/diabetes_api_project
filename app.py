# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# モデルとSHAP explainerをロード
model = joblib.load("model.pkl")
explainer = joblib.load("shap_explainer.pkl")

# FastAPIアプリ生成
app = FastAPI()

# 入力データ型の定義
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

# 特徴量名リスト
feature_names = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# 予測エンドポイント
@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[getattr(data, feature) for feature in feature_names]])

    # 予測
    pred_proba = model.predict_proba(input_array)[0, 1]
    prediction = int(pred_proba >= 0.3)

    # SHAP値計算
    shap_values = explainer.shap_values(input_array)
    shap_contributions = shap_values[0]

    # 上位寄与特徴量（大きい順に3つ出す）
    top_features_idx = np.argsort(-np.abs(shap_contributions))[:3]
    top_features = [(feature_names[i], shap_contributions[i]) for i in top_features_idx]

    # 改善アドバイス作成（例）
    advice = []
    for feature, value in top_features:
        if feature == "BMI" and input_array[0][feature_names.index("BMI")] > 25:
            advice.append("体重管理（BMI低下）を意識しましょう")
        elif feature == "Smoker" and input_array[0][feature_names.index("Smoker")] == 1:
            advice.append("禁煙を検討しましょう")
        elif feature == "PhysActivity" and input_array[0][feature_names.index("PhysActivity")] == 0:
            advice.append("適度な運動を始めましょう")
        elif feature == "GenHlth":
            advice.append("健康状態の改善（睡眠・食事習慣）を意識しましょう")
        # 必要ならさらに条件を追加できる！

    return {
        "diabetes_risk": f"{pred_proba:.2f}",
        "predicted_label": prediction,
        "top_features": [{"feature": f, "shap_value": round(v, 4)} for f, v in top_features],
        "advice": advice
    }
