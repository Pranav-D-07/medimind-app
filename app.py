from flask import Flask, render_template, request, jsonify, session
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
import os

app = Flask(__name__)
app.secret_key = "medimind_secret_key_prod"

# --- CONFIGURATION ---
DATA_FILE = 'master_training_data.csv'
MODEL_FILE = 'model.joblib'
META_FILE = 'meta.joblib'

# 1. GENDER RESTRICTIONS (Diseases strict to one gender)
GENDER_RESTRICTIONS = {
    'Male': [
        'prostate cancer', 'prostatitis', 'benign prostatic hyperplasia (bph)',
        'testicular cancer', 'testicular torsion', 'varicocele', 
        'balanitis', 'epididymitis', 'hydrocele', 'orchitis'
    ],
    'Female': [
        'vulvodynia', 'pregnancy', 'endometriosis', 'ovarian cyst', 
        'pcos', 'polycystic ovarian syndrome', 'cervical cancer', 
        'uterine fibroids', 'vaginitis', 'menopause', 'breast cancer',
        'preeclampsia', 'gestational diabetes', 'turner syndrome'
    ]
}

# 2. GENDER SYMPTOMS (Keywords strict to one gender)
GENDER_SYMPTOMS = {
    'Male': ['testicular', 'scrotal', 'penis', 'prostate', 'testis'],
    'Female': ['vaginal', 'menstruation', 'period', 'pregnancy', 'uterine', 'ovarian', 'breast', 'vulva']
}

# 3. SEVERITY DICT (Updated for Merged Datasets)
SEVERITY_DICT = {
    # --- CRITICAL EMERGENCIES (RED ALERT - Hospital Now) ---
    'heart attack': 'CRITICAL',
    'stroke': 'CRITICAL',
    'stroke warning (tia)': 'CRITICAL',
    'cardiac arrest': 'CRITICAL',
    'kidney failure': 'CRITICAL',
    'sepsis': 'CRITICAL',
    'meningitis': 'CRITICAL',
    'appendicitis': 'CRITICAL',
    'pulmonary embolism': 'CRITICAL',
    'pneumothorax': 'CRITICAL',
    'hypoglycemia': 'CRITICAL',
    'brain hemorrhage': 'CRITICAL',
    'paralysis': 'CRITICAL',
    'heart failure': 'CRITICAL',
    
    # --- HIGH SEVERITY (ORANGE ALERT - See Doctor Today) ---
    'fracture': 'HIGH',
    'dislocation': 'HIGH',
    'head injury': 'HIGH',
    'kidney infection': 'HIGH',
    'kidney stones': 'HIGH',
    'gallbladder disease': 'HIGH',
    'arrhythmia': 'HIGH',
    'heart valve disease': 'HIGH',
    'heart inflammation': 'HIGH',
    'pneumonia': 'HIGH',
    'tuberculosis': 'HIGH',
    'dengue': 'HIGH',
    'malaria': 'HIGH',
    'typhoid': 'HIGH',
    'cancer (lung)': 'HIGH',
    'cancer (breast)': 'HIGH',
    'cancer (prostate)': 'HIGH',
    'cancer (colon)': 'HIGH',
    'cancer (skin)': 'HIGH',
    'liver disease': 'HIGH',
    
    # --- MILD CONDITIONS (STANDARD - General Physician) ---
    'viral infection': 'MILD',
    'common cold': 'MILD',
    'flu': 'MILD',
    'throat infection': 'MILD',
    'ear infection': 'MILD',
    'eye infection': 'MILD',
    'sinusitis': 'MILD',
    'bronchitis': 'MILD',
    'stomach flu': 'MILD',
    'gastritis/ulcer': 'MILD',
    'urinary tract infection': 'MILD',
    'hemorrhoids': 'MILD',
    'anal fissure/fistula': 'MILD',
    'dermatitis': 'MILD',
    'fungal skin infection': 'MILD',
    'bacterial skin infection': 'MILD',
    'stye': 'MILD',
    'dental injury': 'MILD',
    'physical injury': 'MILD',
    'migraine': 'MILD',
    'headache': 'MILD',
    'back pain/sciatica': 'MILD',
    'anxiety/panic': 'MILD',
    'depression': 'MILD',
    'diabetes': 'MILD',
    'hypertension': 'MILD',
    'gerd': 'MILD',
    'allergy': 'MILD',
    'alcohol-related disorder': 'MILD',
    'substance abuse': 'MILD'
}
# 4. RED FLAG SYMPTOMS (Triggers "Zero Tolerance" Mode)
RED_FLAG_SYMPTOMS = [
    # Heart
    'sharp chest pain', 'chest tightness', 'palpitations', 'crushing chest pain',
    # Stroke / Brain
    'slurring words', 'hemiplegia', 'loss of sensation', 'paralysis',
    'sudden severe headache', 'stiff neck', 'confusion', 'unconsciousness',
    # Severe Bleeding / Internal
    'coughing up blood', 'vomiting blood', 'blood in stool', 'melena', 'black stools'
]

# 5. SYMPTOM MAPPING (Slang -> Official Columns)
SYMPTOM_DICT = {
    "hot": ["fever"],
    "burning": ["fever"],
    "pee": ["urinary issues"],
    "peeing": ["urinary issues"],
    "urine": ["urinary issues"],
    "bloating": ["abdominal distention"],
    "breath": ["shortness of breath"],
    "breathing": ["shortness of breath"],
    "dyspnea": ["shortness of breath"],
    "dizzy": ["dizziness"],
    "ache": ["muscle pain"],
    "faint": ["fainting"],
    "puke": ["vomiting"],
    "sick": ["nausea", "vomiting"],
    "stomach": ["stomach_pain", "abdominal distention"],
    # CRITICAL MAPPINGS (Ensure these catch the user's input)
    "chest": ["sharp chest pain", "chest tightness"], 
    "heart": ["sharp chest pain", "chest tightness"],
    "pain": ["sharp chest pain", "muscle pain"],
    "stiff": ["stiff neck", "muscle stiffness"],
    "bleed": ["hemoptysis", "vomiting blood", "rectal bleeding"],
    "blood": ["hemoptysis", "vomiting blood", "rectal bleeding"]
}

# --- LOAD AI BRAIN ---
model = None
data_columns = []
disease_symptom_map = {} 

try:
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE) # <--- BACK TO NORMAL
        meta = joblib.load(META_FILE)
        data_columns = meta.get('columns', [])
        disease_symptom_map = meta.get('disease_symptom_map', {})
        print("✅ AI Brain & Knowledge Base Loaded.")
    else:
        print("❌ Model files not found. Please run train_model.py first.")
except Exception as e:
    print(f"❌ Error loading resources: {e}")

# --- HELPER FUNCTIONS ---

def is_symptom_allowed(symptom, gender):
    """Returns False if the symptom is biologically impossible for the gender"""
    symptom = symptom.lower()
    if not gender: return True 

    if gender == 'Male':
        for banned in GENDER_SYMPTOMS['Female']:
            if banned in symptom: return False
            
    if gender == 'Female':
        for banned in GENDER_SYMPTOMS['Male']:
            if banned in symptom: return False
            
    return True

def extract_symptoms(text, score_cutoff=80):
    if not text: return []
    text = text.lower()
    found = set()
    
    # 1. Dictionary Match
    for slang, official_list in SYMPTOM_DICT.items():
        if slang in text:
            found.update(official_list)

    # 2. Fuzzy Match
    choices = {col: col.replace('_', ' ') for col in data_columns if col not in ['age', 'gender']}
    matches = process.extract(text, choices, scorer=fuzz.token_set_ratio, limit=10)
    for match, score, _ in matches:
        if score >= score_cutoff: 
            found.add(match)
            
    return list(found)

def get_distinguishing_symptom(disease_a, disease_b, current_symptoms, gender):
    """Finds the best question to ask to separate Disease A from B"""
    sym_a = set(disease_symptom_map.get(disease_a, []))
    sym_b = set(disease_symptom_map.get(disease_b, []))
    useful_symptoms = list(sym_a.symmetric_difference(sym_b))
    useful_symptoms.sort()

    for sym in useful_symptoms:
        if (sym not in current_symptoms and 
            sym not in session.get('negated', []) and 
            is_symptom_allowed(sym, gender)):
            return sym
    return None

# --- ROUTES ---

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None: return jsonify({'error': 'Model not loaded.'}), 500

    data = request.get_json()
    user_text = data.get('symptoms', '').lower().strip()
    
    # --- 1. PROFILE & GENDER ---
    if 'age' in data: session['age'] = data['age']
    if 'gender' in data: session['gender'] = data['gender']
    
    raw_gender = session.get('gender', 'Male')
    user_gender = 'Male'
    if str(raw_gender).lower() in ['female', 'f', '0']:
        user_gender = 'Female'

    # --- 2. SESSION & SYMPTOMS ---
    if 'symptoms' not in session: session['symptoms'] = []
    if 'negated' not in session: session['negated'] = []
    if 'last_question' not in session: session['last_question'] = None

    # Handle Yes/No
    if session.get('last_question'):
        positive_answers = ['yes', 'y', 'yeah', 'sure', 'true', 'correct']
        negative_answers = ['no', 'n', 'nope', 'false', 'not really', 'doubt it']
        
        if any(ans in user_text for ans in positive_answers):
            session['symptoms'].append(session['last_question'])
            session['last_question'] = None
        elif any(ans in user_text for ans in negative_answers):
            session['negated'].append(session['last_question'])
            session['last_question'] = None

    # Extract & Add Symptoms
    new_syms = extract_symptoms(user_text)
    for s in new_syms:
        if s not in session['symptoms'] and s not in session['negated']:
            if is_symptom_allowed(s, user_gender):
                session['symptoms'].append(s)

    if not session['symptoms']:
        return jsonify({'recommendation': "I'm listening. Please describe your symptoms."})

    # --- 3. INPUT PREPARATION ---
    input_vector = np.zeros(len(data_columns))
    for s in session['symptoms']:
        if s in data_columns: 
            input_vector[data_columns.index(s)] = 1
            
    if 'age' in data_columns: 
        input_vector[data_columns.index('age')] = int(session.get('age', 30))
    if 'gender' in data_columns:
        g_val = 1 if user_gender == 'Male' else 0
        input_vector[data_columns.index('gender')] = g_val

    input_df = pd.DataFrame([input_vector], columns=data_columns)

    # --- 4. PREDICT ---
    try:
        probs = model.predict_proba(input_df)[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # --- 5. GENDER FILTER ---
    for i, disease in enumerate(model.classes_):
        d_lower = disease.lower()
        if user_gender == 'Male' and d_lower in [x.lower() for x in GENDER_RESTRICTIONS['Female']]:
            probs[i] = 0.0
        if user_gender == 'Female' and d_lower in [x.lower() for x in GENDER_RESTRICTIONS['Male']]:
            probs[i] = 0.0

    if probs.sum() > 0: probs = probs / probs.sum()

    # Get Top 3 Candidates (As requested)
    top_indices = np.argsort(probs)[::-1][:3]
    candidates = []
    for i in top_indices:
        candidates.append({
            'disease': model.classes_[i],
            'prob': probs[i] * 100,
            'severity': SEVERITY_DICT.get(model.classes_[i], 'MILD')
        })

    # --- 6. SMART PARANOID SORTING ---
    
    # A. Check for specific Red Flags in user input
    has_red_flag = any(s in RED_FLAG_SYMPTOMS for s in session['symptoms'])
    
    # B. Set Dynamic Threshold
    # If Red Flag exists: 0.1% (Zero Tolerance) -> Triggers alert even if prob is tiny.
    # If Vague Symptoms: 15.0% (Standard) -> Prevents "Nausea" -> Heart Attack panic.
    safety_threshold = 0.1 if has_red_flag else 15.0
    
    primary = candidates[0]
    
    # Check Top 3 for any Hidden Killers exceeding threshold
    for c in candidates:
        if c['severity'] == 'CRITICAL' and c['prob'] > safety_threshold:
            primary = c
            break  # HIJACK: We found a critical risk. Make it Priority #1.
            
    # Re-calculate Runner-Up based on new Primary
    critical_runner_up = None
    for c in candidates:
        if c['disease'] != primary['disease'] and c['severity'] in ['CRITICAL', 'HIGH'] and c['prob'] > 10.0:
            critical_runner_up = c
            break

    # --- 7. DECISION LOGIC ---
    next_question = None

    # STRATEGY A: IMMEDIATE EMERGENCY (Skip questions)
    if primary['severity'] == 'CRITICAL':
        pass # Fall through to Final Result immediately (Trigger Red Card)

    # STRATEGY B: Check for Hidden Dangers (Runner Up)
    elif critical_runner_up:
        next_question = get_distinguishing_symptom(
            primary['disease'], 
            critical_runner_up['disease'], 
            session['symptoms'],
            user_gender
        )
        if next_question:
            session['last_question'] = next_question
            clean_q = next_question.replace('_', ' ')
            return jsonify({
                'question': f"I want to be safe. Do you also experience <b>{clean_q}</b>?"
            })

    # STRATEGY C: Clarify Low Confidence
    elif primary['prob'] < 70 and len(candidates) > 1:
        next_question = get_distinguishing_symptom(
            primary['disease'], 
            candidates[1]['disease'], 
            session['symptoms'],
            user_gender
        )
        if next_question:
            session['last_question'] = next_question
            clean_q = next_question.replace('_', ' ')
            return jsonify({
                'question': f"Do you have <b>{clean_q}</b>?"
            })

    # --- 8. FINAL RESULT ---
    reco = "Please consult a doctor."
    
    if primary['severity'] == 'CRITICAL':
        reco = "⚠️ <b>EMERGENCY: Please visit a hospital immediately.</b>"
    elif critical_runner_up:
         reco = f"Likely {primary['disease']}, but cannot rule out {critical_runner_up['disease']}. <b>Please seek medical attention.</b>"
    elif primary['prob'] < 40:
        reco = "My confidence is low. Please see a general physician for a checkup."

    return jsonify({
        'diagnosis': [{
            'condition': primary['disease'],
            'probability': f"{primary['prob']:.1f}%",
            'recommendation': reco
        }]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)