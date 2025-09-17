# main.py
import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Optional heavy libs: load them only if available (safe fallbacks)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
except Exception:
    load_model = None
    image = None
    np = None

import uvicorn
import pandas as pd
import pickle

# ---------------- configuration & logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartfarmx")

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
HERE = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(HERE, '..'))

# Serve frontend static files at /app (adjust path if different)
frontend_path = os.path.abspath(os.path.join(HERE, '..', 'smartfarmx_frontend'))
if os.path.exists(frontend_path):
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    logger.warning(f"Frontend path not found (expected): {frontend_path}")

# ---------------- SQLite analytics and FAQ ----------------
DB_PATH = os.path.join(HERE, "smartfarmx.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT,
        details TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS faq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )''')
    # Seed FAQ rows if empty
    c.execute('SELECT COUNT(*) FROM faq')
    if c.fetchone()[0] == 0:
        c.executemany('INSERT INTO faq (question, answer) VALUES (?, ?)', [
            ("How to use Disease Detection?", "Upload a clear leaf image and click Detect."),
            ("What data do I need for Crop Recommendation?", "Soil type, temperature, rainfall and season."),
            ("How often does Analytics refresh?", "Every 5 minutes by default on the dashboard.")
        ])
    conn.commit()
    conn.close()

init_db()

# ---------------- Model & data loading (safe) ----------------
# Attempt to load disease model and crop model and reference files; tolerate missing files.

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet50_final.keras')
model = None
if load_model:
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            logger.info(f"Loaded image model from {MODEL_PATH}")
        else:
            logger.warning(f"Image model not found at {MODEL_PATH}; image predictions will be disabled.")
    except Exception as e:
        logger.exception(f"Failed to load image model: {e}")
else:
    logger.warning("TensorFlow/Keras not available in environment; image predictions disabled.")

# Crop recommendation model (scikit-learn pickle)
CROP_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'RandomForest.pkl')
crop_model = None
crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
try:
    if os.path.exists(CROP_MODEL_PATH):
        with open(CROP_MODEL_PATH, 'rb') as f:
            crop_model = pickle.load(f)
        logger.info(f"Loaded crop model from {CROP_MODEL_PATH}")
    else:
        logger.warning(f"Crop model not found at {CROP_MODEL_PATH}; fallback heuristics will be used.")
except Exception as e:
    logger.exception(f"Failed to load crop model: {e}")
    crop_model = None

# Fertilizer reference CSV
FERT_CSV = os.path.join(BASE_DIR, 'Data-processed', 'fertilizer.csv')
try:
    if os.path.exists(FERT_CSV):
        fert_df = pd.read_csv(FERT_CSV)
        logger.info(f"Loaded fertilizer reference from {FERT_CSV}")
    else:
        fert_df = pd.DataFrame()
        logger.warning(f"Fertilizer CSV not found at {FERT_CSV}")
except Exception as e:
    fert_df = pd.DataFrame()
    logger.exception(f"Error loading fertilizer CSV: {e}")

# ---------------- disease_description.json robust loader ----------------
candidate_paths = [
    os.path.join(HERE, "data", "disease_description.json"),  # backend/data/
    os.path.join(HERE, "disease_description.json"),         # backend/
    os.path.join(BASE_DIR, "disease_description.json"),     # project root
]

disease_desc = {}
DESC_PATH = None
for p in candidate_paths:
    if os.path.exists(p):
        DESC_PATH = p
        try:
            with open(DESC_PATH, encoding='utf-8') as f:
                disease_desc = json.load(f)
            logger.info(f"Loaded disease_description.json from: {DESC_PATH}")
        except Exception as e:
            logger.exception(f"Failed to load JSON at {p}: {e}")
        break

if not DESC_PATH:
    logger.warning("disease_description.json not found. Checked: " + ", ".join(candidate_paths))

def find_field_with_variants(info: dict, base: str, lang_suffix: str):
    """
    Look for possible key variants and return the first non-empty value.
    """
    candidates = []
    # canonical
    candidates.append(f"{base}{lang_suffix}")
    if lang_suffix:
        candidates.append(f"{base} {lang_suffix}")
        candidates.append(f"{base}{lang_suffix.replace('_','')}")
        candidates.append(f"{base}-{lang_suffix.replace('_','')}")
        candidates.append(f"{base}{lang_suffix.replace('_',' ')}")
    candidates.append(base)
    candidates.append(base.lower())

    for c in candidates:
        if c in info and info[c] not in (None, ""):
            return info[c]
    return None

# ---------------- utility helpers ----------------
def preprocess_image(path_or_file):
    """
    Expects path to image file. Returns numpy array shaped for model input.
    """
    if image is None or np is None:
        raise RuntimeError("Image preprocessing libraries (tensorflow.keras) unavailable.")
    img = image.load_img(path_or_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

def log_event(event: str, details: str = "") -> None:
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('INSERT INTO analytics (event, details) VALUES (?, ?)', (event, details))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception(f"Failed to log event: {e}")

# ---------------- root redirect ----------------
@app.get("/")
def root():
    # redirect to frontend mount
    return RedirectResponse(url="/app/")

# ---------------- test endpoint to verify disease JSON ----------------
@app.get("/api/diseases/")
def list_diseases():
    """
    Returns whether disease_description.json loaded, sample keys, and checked path.
    Useful for debugging localization issues.
    """
    if not disease_desc:
        return {"ok": False, "message": "disease_description.json not loaded or empty.", "checked_path": DESC_PATH}
    sample = {}
    for k, v in list(disease_desc.items())[:200]:
        sample[k] = list(v.keys()) if isinstance(v, dict) else []
    return {"ok": True, "total_diseases": len(disease_desc), "sample_keys": sample, "checked_path": DESC_PATH}

# ---------------- disease detection (uses helper + fallback) ----------------
@app.post("/api/disease-detect/")
async def disease_detect(file: UploadFile = File(...), language: str = Form('en')):
    # Save uploaded file to a temp location
    contents = await file.read()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, 'wb') as f:
        f.write(contents)

    # If model not available, return friendly error
    if model is None:
        # cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse({"error": "Image model not available on server."}, status_code=500)

    try:
        img = preprocess_image(temp_path)
        preds = model.predict(img)
        pred_idx = int(np.argmax(preds))
        prob = float(np.max(preds))
        disease = None
        # class_names: if the disease_desc keys correspond to model classes, use that mapping; otherwise fallback
        class_names = list(disease_desc.keys()) if disease_desc else []
        if class_names and pred_idx < len(class_names):
            disease = class_names[pred_idx]
        else:
            # fallback name
            disease = f"class_{pred_idx}"
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.exception("Prediction failed")
        return JSONResponse({"error": "Model prediction failed", "details": str(e)}, status_code=500)

    # cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # description lookup with variants and lang fallback
    desc_obj = disease_desc.get(disease, {}) if disease_desc else {}

    lang_map = {'en': '', 'hi': '_hi', 'mr': '_mr'}
    lang_suffix = lang_map.get(language, '')
    
    # Debug logging
    logger.info(f"Language: {language}, Suffix: {lang_suffix}, Disease: {disease}")
    logger.info(f"Available keys in desc_obj: {list(desc_obj.keys())}")

    description = find_field_with_variants(desc_obj, "Description", lang_suffix) or find_field_with_variants(desc_obj, "Description", "")
    cause = find_field_with_variants(desc_obj, "Cause", lang_suffix) or find_field_with_variants(desc_obj, "Cause", "")
    treatment = find_field_with_variants(desc_obj, "Treatment", lang_suffix) or find_field_with_variants(desc_obj, "Treatment", "")
    prevention = find_field_with_variants(desc_obj, "Prevention", lang_suffix) or find_field_with_variants(desc_obj, "Prevention", "")
    
    # Debug logging for results
    logger.info(f"Found description: {description[:50] if description else 'None'}...")
    logger.info(f"Found cause: {cause[:50] if cause else 'None'}...")

    result = {
        "disease": disease,
        "probability": round(prob, 4),
        "description": description or "N/A",
        "cause": cause or "N/A",
        "treatment": treatment or "N/A",
        "prevention": prevention or "N/A",
    }

    try:
        log_event('disease_detection', json.dumps({"disease": disease, "probability": result["probability"], "lang": language}))
    except Exception:
        pass

    return result

# ---------------- crop recommendation ----------------
@app.post("/api/crop-recommend/")
def crop_recommend(
    N: float = Form(None),
    P: float = Form(None),
    K: float = Form(None),
    temperature: float = Form(None),
    humidity: float = Form(None),
    ph: float = Form(None),
    rainfall: float = Form(None)
):
    feature_values = [N, P, K, temperature, humidity, ph, rainfall]
    recommendation = {}
    # If trained model available and all features present, use it
    if crop_model is not None and all(v is not None for v in feature_values):
        try:
            X = np.array([feature_values], dtype=float)
            pred = crop_model.predict(X)[0]
            recommendation = {"recommended_crop": str(pred)}
        except Exception as e:
            logger.exception("Crop model prediction failed")
            recommendation = {"recommended_crop": "Unavailable", "error": str(e)}
    else:
        # fallback heuristic using fert_df average nearest
        try:
            if all(v is not None for v in [N, P, K]) and not fert_df.empty and 'Crop' in fert_df.columns:
                tmp = fert_df.copy()
                tmp['dist'] = (tmp['N']-float(N))**2 + (tmp['P']-float(P))**2 + (tmp['K']-float(K))**2
                best = tmp.sort_values('dist').iloc[0]
                recommendation = {"recommended_crop": best['Crop']}
            else:
                recommendation = {"recommended_crop": "Wheat"}
        except Exception:
            recommendation = {"recommended_crop": "Wheat"}

    try:
        log_event('crop_recommendation', json.dumps({"N": N, "P": P, "K": K}))
    except Exception:
        pass

    return recommendation

# ---------------- fertilizer suggestion ----------------
@app.post("/api/fertilizer-suggest/")
def fertilizer_suggest(crop: str = Form(...), n: float = Form(...), p: float = Form(...), k: float = Form(...)):
    recommendation = {"fertilizer": "Balanced NPK", "quantity": "N/A"}
    try:
        if not fert_df.empty and 'Crop' in fert_df.columns:
            row = fert_df[fert_df['Crop'].str.lower() == crop.strip().lower()]
            if not row.empty:
                row = row.iloc[0]
                # ensure numeric values
                try:
                    target = {"N": float(row['N']), "P": float(row['P']), "K": float(row['K'])}
                except Exception:
                    target = {"N": None, "P": None, "K": None}
                current = {"N": float(n), "P": float(p), "K": float(k)}
                # compute delta (positive => add)
                delta = {}
                for key_ in ['N','P','K']:
                    t = target.get(key_)
                    c = current.get(key_)
                    if t is None or c is None:
                        delta[key_] = None
                    else:
                        delta[key_] = round(t - c, 2)
                recommendation.update({
                    "target": target,
                    "current": current,
                    "delta": delta,
                    "guidance": f"Adjust by N:{delta.get('N')}, P:{delta.get('P')}, K:{delta.get('K')} to reach recommended levels"
                })
    except Exception as e:
        logger.exception("Fertilizer suggestion failed: " + str(e))

    try:
        log_event('fertilizer_suggestion', json.dumps({"crop": crop}))
    except Exception:
        pass

    return recommendation

# ---------------- weather (with API key placeholder) ----------------
@app.get("/api/weather/")
def get_weather(location: str):
    # TODO: Replace with your actual weather API key
    # Example: WEATHER_API_KEY = "your_openweathermap_api_key_here"
    # weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}"
    
    # Placeholder response until API key is added
    weather = {
        "location": location, 
        "forecast": "Sunny", 
        "temperature": 30,
        "humidity": 65,
        "wind_speed": 10,
        "description": "Clear sky",
        "note": "Add your weather API key to get real-time data"
    }
    try:
        log_event('weather_fetch', json.dumps({"location": location}))
    except Exception:
        pass
    return weather

# ---------------- chatbot endpoint ----------------
@app.post("/api/chatbot/")
def chatbot_response(message: str = Form(...)):
    # TODO: Replace with your actual chatbot API key
    # Example: CHATBOT_API_KEY = "your_chatbot_api_key_here"
    # chatbot_url = f"https://api.your-chatbot-service.com/chat?key={CHATBOT_API_KEY}&message={message}"
    
    # Placeholder response until API key is added
    responses = [
        "I'm here to help with agricultural questions! Add your chatbot API key to enable full functionality.",
        "Ask me about farming techniques, crop diseases, or agricultural best practices.",
        "I can assist with plant care, soil management, and farming advice.",
        "Need help with agricultural problems? I'm ready to assist once the API is configured."
    ]
    
    # Simple keyword-based responses for now
    message_lower = message.lower()
    if any(word in message_lower for word in ['disease', 'sick', 'problem', 'issue']):
        response = "For plant disease issues, try our Disease Detection feature! Upload a leaf image for instant diagnosis."
    elif any(word in message_lower for word in ['crop', 'plant', 'grow']):
        response = "Use our Crop Recommendation tool to find the best crops for your soil and climate conditions."
    elif any(word in message_lower for word in ['fertilizer', 'nutrient', 'npk']):
        response = "Check our Fertilizer Suggestion feature for personalized NPK recommendations."
    elif any(word in message_lower for word in ['weather', 'rain', 'temperature']):
        response = "Our Weather tab provides current conditions and forecasts for your location."
    else:
        response = responses[hash(message) % len(responses)]
    
    try:
        log_event('chatbot_query', json.dumps({"message": message[:100]}))
    except Exception:
        pass
    
    return {"response": response, "note": "Add your chatbot API key for enhanced AI responses"}

# ---------------- analytics endpoint ----------------
@app.get("/api/analytics/")
def get_analytics(range: str = '7d'):
    conn = get_db()
    c = conn.cursor()
    now = datetime.utcnow()
    if range == '30d':
        start = now - timedelta(days=30)
    elif range == '90d':
        start = now - timedelta(days=90)
    else:
        start = now - timedelta(days=7)
    try:
        c.execute("SELECT event, COUNT(*) FROM analytics WHERE timestamp >= ? GROUP BY event", (start.strftime('%Y-%m-%d %H:%M:%S'),))
        grouped = c.fetchall()
        c.execute("SELECT COUNT(*) FROM analytics WHERE timestamp >= ?", (start.strftime('%Y-%m-%d %H:%M:%S'),))
        total = c.fetchone()[0]
    except Exception as e:
        logger.exception("Analytics query failed: " + str(e))
        grouped = []
        total = 0
    finally:
        conn.close()
    return {"totals": {"all": total}, "by_event": grouped}

# ---------------- FAQ endpoint ----------------
@app.get("/api/faq/")
def get_faq():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT question, answer FROM faq")
    data = c.fetchall()
    conn.close()
    return {"faq": data}

# ---------------- run uvicorn ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

