from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Creating blueprint
class InsuranceData(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # <-- change this
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading trained model
model = joblib.load("insurance_model.pkl")

# Home endpoint
@app.get("/")
def home():
    return {"message": "Insurance Prediction API is Running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InsuranceData):

    features = [[
        data.age,
        data.sex,
        data.bmi,
        data.children,
        data.smoker,
        data.region
    ]]

    prediction = model.predict(features)

    return {
        "predicted_charges": float(prediction[0])
    }