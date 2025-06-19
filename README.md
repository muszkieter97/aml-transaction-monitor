# 🚨 Suspicious Transaction Detector

This project is an interactive machine learning tool for detecting potentially fraudulent transactions in real time. Built with:

- 🐍 Python
- 🎛️ Streamlit (frontend)
- ⚡ FastAPI (backend API)
- 🌲 Isolation Forest / One-Class SVM / Random Forest (ML models)
- ☁️ Deployed on AWS EC2

---

## 💡 Features

- Upload your own CSV file or use default sample
- Select between:
  - Isolation Forest (unsupervised)
  - One-Class SVM (unsupervised)
  - Random Forest (supervised with feedback)
- Filter suspicious transactions by country, type, amount
- Visualize anomalies with charts
- Manually flag transactions (fraud/legit)
- Train supervised model on your labels (flags.csv)
- Use a REST API for predictions (via FastAPI)

---

## 🖥️ Demo (AWS EC2)

- 🌐 App: [http://13.60.66.0:8501](http://13.60.66.0:8501)
- ⚙️ API Docs: [http://13.60.66.0:8000/docs](http://13.60.66.0:8000/docs)

> ⚠️ Data used is simulated for demonstration purposes.

---

## 📸 Screenshots

### 🔍 Filtering & Visualization  
![Filtering](docs/filter_view.png)

### 🏷️ Manual Flagging + Summary  
![Flagging](docs/flagging.png)

### ⚡ Quick Prediction via API  
![API](docs/api_response.png)

---

## 🧠 Tech Stack

| Component     | Tool                |
|---------------|---------------------|
| Language      | Python              |
| Frontend      | Streamlit           |
| Backend API   | FastAPI             |
| ML Models     | Scikit-learn        |
| Data Handling | pandas, NumPy       |
| Charts        | seaborn, matplotlib |
| Deployment    | AWS EC2 (Ubuntu)    |

---

## ⚙️ Installation (local)

```bash
git clone https://github.com/muszkieter97/aml-transaction-monitor.git
cd aml-transaction-monitor
python -m venv venv
venv\Scripts\activate     # or source venv/bin/activate on Linux
pip install -r requirements.txt

# Run frontend
streamlit run app/streamlit_app.py

# (optional) Run backend
uvicorn app.api:app --reload

---



