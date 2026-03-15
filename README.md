# 🔍 FakeGuard AI — AI Fake News Detector

A full-stack web application that uses Machine Learning (NLP + TF-IDF + Logistic Regression) to classify news as **Real** or **Fake** with a confidence score.

---

## Tech Stack
- **Frontend**: HTML, CSS (glassmorphism dark-theme), JavaScript  
- **Backend**: Python + Flask
- **AI/ML**: Scikit-learn, NLTK, TF-IDF Vectorizer, Logistic Regression  
- **Database**: SQLite (local history)

---

## ⚡ Quick Start

### 1. Install Python & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (One-time)
Download **Fake.csv** and **True.csv** from:  
👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place both files in the project root, then:
```bash
python prepare_data.py   # Merges, labels, and preprocesses the data
python train_model.py    # Trains the model and saves model.pkl + vectorizer.pkl
```

### 3. Run Flask Server
```bash
python app.py
```

Open your browser → **http://127.0.0.1:8000/**

---

## 📁 Project Structure
```
fakenews_project/
├── app.py                 ← Flask application
├── index.html             ← Home page
├── history.html           ← History page
├── prepare_data.py        ← Data preprocessing script
├── train_model.py         ← Model training script
├── generate_mock_dataset.py ← Optional mock data generator
├── requirements.txt       ← Python dependencies
└── .gitignore             ← Git ignore rules
```

## 📡 API Endpoints
| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/` | Home page |
| GET    | `/history/` | History page |
| POST   | `/api/predict/` | Predict real/fake |
| GET    | `/api/history/` | Last 20 predictions (JSON) |

**POST `/api/predict/` body example:**
```json
{ "text": "Scientists discover a new planet..." }
```
**Response:**
```json
{ "prediction": "Real News", "confidence": "96.42", "is_real": true }
```
