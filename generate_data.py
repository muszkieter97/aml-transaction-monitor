import pandas as pd
import numpy as np

np.random.seed(42)

n_customers = 100
n_transactions = 3000

customers = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

data = {
    "transaction_id": [f"TX_{i:06d}" for i in range(n_transactions)],
    "customer_id": np.random.choice(customers, size=n_transactions),
    "amount": np.random.exponential(scale=500, size=n_transactions).round(2),
    "transaction_type": np.random.choice(["wire", "card", "crypto", "cash"], size=n_transactions, p=[0.5, 0.3, 0.1, 0.1]),
    "country": np.random.choice(["PL", "DE", "RU", "UA", "US", "GB", "NG"], size=n_transactions, p=[0.4, 0.2, 0.05, 0.05, 0.1, 0.15, 0.05]),
    "hour": np.random.randint(0, 24, size=n_transactions),
    "is_fraud": 0
}

df = pd.DataFrame(data)

#Tworzenie sztucznych FRAUDów
fraud_idx = np.random.choice(df.index, size=50, replace=False)
df.loc[fraud_idx, "amount"] *= 20  # Bardzo duża kwota
df.loc[fraud_idx, "country"] = np.random.choice(["RU", "NG", "UA"], size=50)
df.loc[fraud_idx, "is_fraud"] = 1

df.to_csv("data/transactions.csv", index=False)
print("Wygenerowano dane w data/transactions.csv")
