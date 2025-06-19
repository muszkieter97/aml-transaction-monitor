from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

# Inicjalizacja aplikacji FastAPI
app = FastAPI(title="AML Fraud Detection API")

# Wczytaj model i scaler z katalogu models/
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/dummy_columns.txt") as f:
    DUMMY_COLUMNS = [line.strip() for line in f.readlines()]

# 👉 Definicja pojedynczej transakcji
class Transaction(BaseModel):
    amount: float
    hour: float
    transaction_type: str
    country: str

# 👉 Batch – lista transakcji
class TransactionBatch(BaseModel):
    transactions: List[Transaction]

# 👉 Endpoint do predykcji batchowej
@app.post("/predict_batch")
def predict_batch(payload: TransactionBatch):
    # Konwersja listy transakcji do listy słowników
    records = [t.dict() for t in payload.transactions]
    df = pd.DataFrame(records)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["transaction_type", "country"], drop_first=False)

    # Uzupełnij brakujące kolumny
    for col in DUMMY_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[DUMMY_COLUMNS]

    # Skalowanie
    df[["amount", "hour"]] = scaler.transform(df[["amount", "hour"]])

    # Predykcja
    preds = model.predict(df)
    labels = ["fraud" if p == 1 else "ok" for p in preds]

    return {"predictions": labels}
