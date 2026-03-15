import os
import re
import json
import pickle
import sqlite3
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from flask import Flask, render_template, request, jsonify

# ── Download NLP resources ────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
_STOP_WORDS = set(stopwords.words('english'))
_STEMMER    = PorterStemmer()

# ── Initialize Flask App ──────────────────────────────────────────────────────
# Using '.' for standard folders so everything is in the root directory.
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# ── Load Trained Model (loaded once at startup) ───────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH  = os.path.join(_BASE_DIR, 'model.pkl')
_VEC_PATH    = os.path.join(_BASE_DIR, 'vectorizer.pkl')

try:
    _model      = pickle.load(open(_MODEL_PATH, 'rb'))
    _vectorizer = pickle.load(open(_VEC_PATH,   'rb'))
    _MODEL_READY = True
except FileNotFoundError:
    _model = _vectorizer = None
    _MODEL_READY = False

# ── SQLite Database Setup ─────────────────────────────────────────────────────
DB_PATH = os.path.join(_BASE_DIR, 'db.sqlite3')

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_text TEXT NOT NULL,
                prediction VARCHAR(20) NOT NULL,
                confidence_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def _preprocess(text: str) -> str:
    """Apply the same NLP pipeline used during training."""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [_STEMMER.stem(w) for w in words if w not in _STOP_WORDS]
    return ' '.join(words)

# ── Page Views ────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history/')
def history_page():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Get last 20 records
        cursor.execute('SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT 20')
        records = cursor.fetchall()

    real_count = sum(1 for r in records if r['prediction'] == 'Real News')
    fake_count = len(records) - real_count

    # Convert records to dict for easy use in template, though Row acts like dict
    return render_template('history.html', records=records, real_count=real_count, fake_count=fake_count)

# ── API Views ─────────────────────────────────────────────────────────────────
@app.route('/api/predict/', methods=['POST'])
def predict_news():
    """
    POST /api/predict/
    Body: {"text": "<news text>"}
    Response: {"prediction": "Real News"|"Fake News", "confidence": "XX.XX%"}
    """
    if not _MODEL_READY:
        return jsonify({'error': 'Model not loaded. Please run the training scripts first.'}), 503

    try:
        body = request.get_json()
        if not body:
            return jsonify({'error': 'Invalid JSON body.'}), 400
    except Exception:
        return jsonify({'error': 'Invalid request.'}), 400

    raw_text = body.get('text', '').strip()
    if not raw_text:
        return jsonify({'error': 'No text provided.'}), 400
    if len(raw_text) < 10:
        return jsonify({'error': 'Text is too short. Please provide more content.'}), 400

    # Preprocess → Vectorize → Predict
    processed   = _preprocess(raw_text)
    vectorized  = _vectorizer.transform([processed])
    prediction  = _model.predict(vectorized)[0]          # 0=Fake, 1=Real
    proba       = _model.predict_proba(vectorized)[0]    # List of probabilities
    confidence  = float(proba[prediction]) * 100

    label = "Real News" if prediction == 1 else "Fake News"

    # Persist to database
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_history (news_text, prediction, confidence_score)
            VALUES (?, ?, ?)
        ''', (raw_text[:600], label, round(confidence, 2)))
        conn.commit()

    return jsonify({
        'prediction': label,
        'confidence': f"{confidence:.2f}",
        'is_real':    bool(prediction == 1),
    })

@app.route('/api/history/', methods=['GET'])
def history_api():
    """GET /api/history/ → last 20 predictions as JSON."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT 20')
        records = cursor.fetchall()
        
    data = []
    for r in records:
        text = r['news_text'][:120] + ('…' if len(r['news_text']) > 120 else '')
        # Format datetime if it comes out as string
        dt_str = r['created_at']
        try:
            # SQLite default timestamp format
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            formatted_date = dt.strftime('%b %d, %Y  %H:%M')
        except ValueError:
            formatted_date = dt_str
            
        data.append({
            'text':       text,
            'prediction': r['prediction'],
            'confidence': r['confidence_score'],
            'is_real':    r['prediction'] == 'Real News',
            'created_at': formatted_date,
        })
        
    return jsonify({'history': data})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
