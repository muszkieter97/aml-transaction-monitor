# 🚨 Suspicious Transaction Detector

This is a complete **Machine Learning project** for detecting potentially fraudulent transactions, built with:

- 🐍 Python
- 🎛️ Streamlit (Frontend)
- ⚡ FastAPI (Backend API)
- 🌲 Isolation Forest / One-Class SVM / Random Forest (ML Models)
- ☁️ Deployed on AWS EC2

---

## 💡 Features

- Upload your own transaction data
- Choose between 3 detection models:
  - Isolation Forest (unsupervised)
  - One-Class SVM (unsupervised)
  - Random Forest (supervised – trained on manually flagged transactions)
- Filter, explore and visualize suspicious data
- Manually flag transactions and save them for model training
- Run live predictions via FastAPI
- View client-level summaries

---

## 📸 Demo

🖥️ Live demo hosted on EC2:  
[http://13.60.66.0:8501](http://13.60.66.0:8501)

> 💡 Backend API docs: [http://13.60.66.0:8000/docs](http://13.60.66.0:8000/docs)

---

## 🧠 Tech Stack

| Area             | Tool                |
|------------------|---------------------|
| Language         | Python              |
| Frontend         | Streamlit           |
| Backend API      | FastAPI             |
| ML Models        | Scikit-Learn        |
| Data Handling    | pandas, NumPy       |
| Visualizations   | seaborn, matplotlib |
| Deployment       | AWS EC2 (Ubuntu)    |

---

## 🗂️ Folder Structure

aml-transaction-monitor/
├── app/
│ ├── streamlit_app.py # Streamlit frontend
│ └── api.py # FastAPI backend
├── data/ # Input CSVs and feedback
├── models/ # Trained ML models (.pkl)
├── notebooks/ # Experiments / EDA
├── requirements.txt
└── README.md


---

## 🔧 Installation (optional)

You can run this project locally using:

```bash
git clone https://github.com/muszkieter97/aml-transaction-monitor.git
cd aml-transaction-monitor
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Linux
pip install -r requirements.txt
streamlit run app/streamlit_app.py


Author Mateusz Muszkiet
Senior Compliance Associate
Gdańsk, Poland
