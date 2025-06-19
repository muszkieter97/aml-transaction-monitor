import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Wczytanie danych z pliku flags.csv
flags = pd.read_csv("data/flags.csv")

# Przygotowanie danych
X = flags[["amount", "transaction_type", "country", "hour"]]
y = flags["user_flag"].map({"ok": 0, "fraud": 1})  # etykiety: 0 = OK, 1 = fraud

# One-hot encoding
X = pd.get_dummies(X, columns=["transaction_type", "country"], drop_first=True)

# Skalowanie kolumn numerycznych
scaler = StandardScaler()
X[["amount", "hour"]] = scaler.fit_transform(X[["amount", "hour"]])

# Trening modelu
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Zapis modelu i skalera do folderu models/
joblib.dump(clf, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model i scaler zapisane w katalogu models/")

with open("models/dummy_columns.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")