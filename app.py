from flask import Flask, render_template, request, send_file, jsonify, session, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pymongo
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io, base64, os
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

# Local-la .env file irundha load panna (Hosting-la idhu optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
# Secret key environmental variable-la irundhu edukkum, illana default use pannum
app.secret_key = os.environ.get('SECRET_KEY', 'vh_ai_secure_2026') 
nltk.download('vader_lexicon', quiet=True)

# --- Admin Credentials ---
ADMIN_USER = "admin"
ADMIN_PASS = "12345"

# --- MongoDB Setup (Dynamic for Hosting) ---
# Un Atlas URL inga default-aa kuduthurukkaen
ATLAS_URL = "mongodb+srv://ranjithdev078_db_user:tve7PlCsOILmAnUg@cluster0.sp9xvcq.mongodb.net/ValueHealthDB?retryWrites=true&w=majority"
MONGO_URI = os.environ.get('MONGO_URI', ATLAS_URL)

# --- AI Model Setup ---
model = None
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age']

def get_db():
    client = pymongo.MongoClient(MONGO_URI)
    return client["ValueHealthDB"]

def train_model():
    global model
    try:
        db = get_db()
        col = db["PatientData"]
        data = list(col.find())
        unique_outcomes = set([d.get('Outcome') for d in data])
        
        # Database empty-aa irundha sample data load panna
        if len(data) < 5 or len(unique_outcomes) < 2:
            sample = [
                {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigree": 0.627, "Age": 50, "Outcome": 1},
                {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigree": 0.351, "Age": 31, "Outcome": 0},
                {"Pregnancies": 8, "Glucose": 183, "BloodPressure": 64, "SkinThickness": 0, "Insulin": 0, "BMI": 23.3, "DiabetesPedigree": 0.672, "Age": 32, "Outcome": 1},
                {"Pregnancies": 1, "Glucose": 89, "BloodPressure": 66, "SkinThickness": 23, "Insulin": 94, "BMI": 28.1, "DiabetesPedigree": 0.167, "Age": 21, "Outcome": 0},
                {"Pregnancies": 3, "Glucose": 78, "BloodPressure": 50, "SkinThickness": 32, "Insulin": 88, "BMI": 31.0, "DiabetesPedigree": 0.248, "Age": 26, "Outcome": 1}
            ]
            col.insert_many(sample)
            data = list(col.find())

        df = pd.DataFrame(data)
        X = df[feature_cols]
        y = df['Outcome']
        if len(y.unique()) > 1:
            model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
            print("✅ Clinical Prediction Model Ready!")
    except Exception as e:
        print(f"⚠️ Training Error: {e}")

# Trigger training on startup
train_model()

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != ADMIN_USER or request.form['password'] != ADMIN_PASS:
            error = 'Invalid Credentials. Access Denied!'
        else:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('login'))
    try:
        db = get_db()
        total = db["PatientHistory"].count_documents({})
        urgent = db["PatientHistory"].count_documents({"status": "URGENT"})
        stable = db["PatientHistory"].count_documents({"status": "STABLE"})
        recent = list(db["PatientHistory"].find().sort("_id", -1).limit(5))
        return render_template('dashboard.html', total=total, urgent=urgent, stable=stable, recent=recent)
    except:
        return render_template('dashboard.html', total=0, urgent=0, stable=0, recent=[])

@app.route('/analysis')
def analysis():
    if not session.get('logged_in'): return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'): return redirect(url_for('login'))
    if model is None: train_model()
    try:
        db = get_db()
        p_name = request.form.get('patient_name')
        p_age = request.form.get('age')
        p_gender = request.form.get('gender')
        
        fields = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
        input_values = [float(request.form.get(f)) for f in fields]
        notes = request.form.get('notes', '')
        
        prob = model.predict_proba([input_values])[0][1]
        risk_score = int(prob * 100)
        prediction = model.predict([input_values])[0]
        
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(notes)['compound']
        
        glucose_val = input_values[1]
        bmi_val = input_values[5]
        
        if (prediction == 1 or sentiment < -0.3 or glucose_val > 140):
            status, color = "URGENT", "danger"
        else:
            status, color = "STABLE", "success"
        
        recs = []
        if status == "URGENT": recs.append("Consult an endocrinologist immediately.")
        if glucose_val > 140: recs.append("High Glucose: Limit carbohydrate intake.")
        if bmi_val > 25: recs.append("Elevated BMI: Regular 30-min exercise suggested.")
        if not recs: recs.append("Maintain healthy lifestyle.")

        diet_plan = []
        if glucose_val > 140:
            diet_type = "Low-Sugar Diabetic Plan"
            diet_plan = [
                {"day": "Mon-Wed", "b": "Oats with Nuts", "l": "Brown Rice & Greens", "d": "Wheat Upma"},
                {"day": "Thu-Sat", "b": "Sprouted Salad", "l": "Millets & Curd", "d": "Vegetable Soup"},
                {"day": "Sunday", "b": "2 Idli & Chutney", "l": "Small Portion Rice", "d": "Oats Porridge"}
            ]
        elif bmi_val > 25:
            diet_type = "Weight Loss Plan"
            diet_plan = [
                {"day": "Mon-Wed", "b": "Boiled Moong Dal", "l": "Salad & Grilled Veg", "d": "Papaya Bowl"},
                {"day": "Thu-Sat", "b": "No-Sugar Smoothie", "l": "Pulse-based Meal", "d": "Clear Soup"},
                {"day": "Sunday", "b": "Whole Wheat Bread", "l": "Regular Meal (Small)", "d": "Fruit Bowl"}
            ]
        else:
            diet_type = "General Wellness Plan"
            diet_plan = [
                {"day": "Mon-Sat", "b": "Traditional Tiffin", "l": "Full Balanced Meal", "d": "Light Tiffin"},
                {"day": "Sunday", "b": "Healthy Choice", "l": "Regular Meal", "d": "Fruits"}
            ]

        db["PatientHistory"].insert_one({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": p_name, "age": p_age, "gender": p_gender,
            "glucose": glucose_val, "risk_score": risk_score,
            "status": status, "recommendations": recs
        })
        
        plt.figure(figsize=(8, 4))
        plt.barh(feature_cols, model.feature_importances_, color='#0d6efd')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0); chart = base64.b64encode(buf.getvalue()).decode('utf-8'); plt.close()

        return render_template('index.html', patient_name=p_name, patient_age=p_age, patient_gender=p_gender,
                               status=status, risk_score=risk_score, recommendations=recs, 
                               graph=chart, color=color, show_result=True, 
                               diet_plan=diet_plan, diet_type=diet_type)
    except Exception as e:
        return f"Prediction Error: {e}"

@app.route('/records')
def records():
    if not session.get('logged_in'): return redirect(url_for('login'))
    db = get_db()
    history = list(db["PatientHistory"].find().sort("_id", -1))
    return render_template('records.html', records=history)

@app.route('/download_report', methods=['POST'])
def download_report():
    if not session.get('logged_in'): return redirect(url_for('login'))
    try:
        name, age, gender, status = request.form.get('name'), request.form.get('age'), request.form.get('gender'), request.form.get('status')
        risk = int(request.form.get('risk_score', 0))
        recs = request.form.getlist('recs')
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)
        p.setFont("Helvetica-Bold", 18); p.drawString(100, 800, "Value Health AI Report")
        p.line(100, 785, 500, 785)
        p.setFont("Helvetica", 11); p.drawString(100, 750, f"Name: {name} | Age: {age} | Gender: {gender}")
        
        plt.figure(figsize=(2, 2))
        plt.pie([max(risk, 1), 100-risk], colors=['#dc3545' if status=="URGENT" else '#198754', '#eee'], startangle=90, wedgeprops={'width':0.4})
        plt.text(0, 0, f"{risk}%", ha='center', va='center', fontsize=10, fontweight='bold')
        img_buf = io.BytesIO(); plt.savefig(img_buf, format='png', transparent=True); img_buf.seek(0); plt.close()
        
        p.drawImage(ImageReader(img_buf), 380, 680, width=100, height=100)
        p.setFont("Helvetica-Bold", 12); p.drawString(100, 680, f"STATUS: {status}")
        y = 650; p.drawString(100, y, "AI Recommendations:")
        for r in recs: y -= 20; p.drawString(120, y, f"• {r}")
        p.showPage(); p.save(); buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{name}_Report.pdf", mimetype='application/pdf')
    except Exception as e:
        return f"PDF Error: {e}"

# --- HOSTING CONFIGURATION ---
if __name__ == '__main__':
    # Render automatic-aa allocate pannura port-ah edukkum
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)