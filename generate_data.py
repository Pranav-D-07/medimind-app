import pandas as pd
import numpy as np
import random

# --- MEDICAL KNOWLEDGE BASE (Expanded) ---
# Format: Disease -> {mandatory symptoms}, {optional symptoms}
# Note: Using exact column names from your original dataset

DISEASE_RULES = {
    # --- RESPIRATORY (12) ---
    "Common Cold": {
        "mandatory": ["cough", "nasal congestion", "sore throat"],
        "optional": ["sneezing", "runny nose", "mild fever", "headache"]
    },
    "Influenza": {
        "mandatory": ["high fever", "muscle pain", "fatigue"],
        "optional": ["cough", "sore throat", "headache", "chills"]
    },
    "COVID-19": {
        "mandatory": ["fever", "cough", "shortness of breath"],
        "optional": ["loss of smell", "fatigue", "sore throat", "headache"]
    },
    "Pneumonia": {
        "mandatory": ["high fever", "productive cough", "shortness of breath"],
        "optional": ["chest pain", "chills", "fatigue", "sweating"]
    },
    "Bronchitis": {
        "mandatory": ["persistent cough", "production of sputum", "chest discomfort"],
        "optional": ["mild fever", "fatigue", "shortness of breath"]
    },
    "Asthma": {
        "mandatory": ["wheezing", "shortness of breath", "chest tightness"],
        "optional": ["cough", "difficulty breathing", "anxiety"]
    },
    "COPD (Chronic Bronchitis/Emphysema)": {
        "mandatory": ["chronic cough", "shortness of breath", "exertional breathlessness"],
        "optional": ["sputum production", "wheeze", "fatigue"]
    },
    "Tuberculosis": {
        "mandatory": ["persistent cough", "weight loss", "night sweats"],
        "optional": ["fever", "chest pain", "blood in sputum"]
    },
    "Sinusitis": {
        "mandatory": ["facial pain", "nasal congestion", "thick nasal discharge"],
        "optional": ["reduced smell", "headache", "cough"]
    },
    "Laryngitis": {
        "mandatory": ["hoarseness", "sore throat", "voice loss"],
        "optional": ["dry cough", "throat pain", "mild fever"]
    },
    "Pulmonary Embolism": {
        "mandatory": ["sudden shortness of breath", "pleuritic chest pain", "rapid heart rate"],
        "optional": ["cough", "sweating", "dizziness"]
    },
    "ARDS (Acute Respiratory Distress Syndrome)": {
        "mandatory": ["severe shortness of breath", "rapid breathing", "low oxygen levels"],
        "optional": ["fever", "confusion", "cough"]
    },

    # --- CARDIAC (10) ---
    "Myocardial Infarction (Heart Attack)": {
        "mandatory": ["chest pain", "shortness of breath", "chest pressure"],
        "optional": ["sweating", "nausea", "radiating arm pain", "dizziness"]
    },
    "Angina": {
        "mandatory": ["exertional chest pain", "chest tightness", "relief with rest"],
        "optional": ["shortness of breath", "fatigue", "sweating"]
    },
    "Heart Failure": {
        "mandatory": ["shortness of breath", "leg swelling", "fatigue"],
        "optional": ["persistent cough", "reduced exercise tolerance", "weight gain"]
    },
    "Hypertension": {
        "mandatory": ["headache", "dizziness", "blurred vision"],
        "optional": ["chest pain", "nosebleed", "fatigue"]
    },
    "Arrhythmia": {
        "mandatory": ["palpitations", "irregular heartbeat", "fluttering sensation"],
        "optional": ["dizziness", "syncope", "shortness of breath"]
    },
    "Pericarditis": {
        "mandatory": ["sharp chest pain", "worse when lying down", "relief when leaning forward"],
        "optional": ["fever", "shortness of breath", "palpitations"]
    },
    "Endocarditis": {
        "mandatory": ["fever", "heart murmur", "fatigue"],
        "optional": ["night sweats", "shortness of breath", "joint pain"]
    },
    "Cardiomyopathy": {
        "mandatory": ["shortness of breath", "fatigue", "swelling of legs"],
        "optional": ["palpitations", "chest discomfort", "dizziness"]
    },
    "Aortic Stenosis": {
        "mandatory": ["exertional syncope", "exertional chest pain", "exertional breathlessness"],
        "optional": ["fatigue", "palpitations", "dizziness"]
    },
    "Deep Vein Thrombosis (DVT)": {
        "mandatory": ["leg pain", "unilateral leg swelling", "tenderness"],
        "optional": ["redness", "warmth of limb", "visible veins"]
    },

    # --- GASTROINTESTINAL (12) ---
    "Gastroenteritis": {
        "mandatory": ["vomiting", "diarrhea", "stomach pain"],
        "optional": ["nausea", "fever", "dehydration"]
    },
    "Appendicitis": {
        "mandatory": ["abdominal pain", "pain in lower right abdomen", "loss of appetite"],
        "optional": ["nausea", "vomiting", "fever"]
    },
    "Peptic Ulcer Disease": {
        "mandatory": ["burning stomach pain", "epigastric pain", "worse on empty stomach"],
        "optional": ["nausea", "vomiting", "bloating"]
    },
    "GERD (Acid Reflux)": {
        "mandatory": ["heartburn", "acid taste in mouth", "regurgitation"],
        "optional": ["chest pain", "sore throat", "chronic cough"]
    },
    "Irritable Bowel Syndrome (IBS)": {
        "mandatory": ["abdominal pain", "altered bowel habits", "bloating"],
        "optional": ["mucus in stool", "relief after defecation", "urgency"]
    },
    "Inflammatory Bowel Disease (Crohn's/Ulcerative Colitis)": {
        "mandatory": ["persistent diarrhea", "abdominal pain", "rectal bleeding"],
        "optional": ["weight loss", "fever", "fatigue"]
    },
    "Cholecystitis (Gallbladder inflammation)": {
        "mandatory": ["right upper quadrant pain", "pain after fatty meal", "fever"],
        "optional": ["nausea", "vomiting", "jaundice"]
    },
    "Pancreatitis": {
        "mandatory": ["severe upper abdominal pain", "radiation to back", "nausea"],
        "optional": ["vomiting", "fever", "tender abdomen"]
    },
    "Hepatitis (Viral)": {
        "mandatory": ["jaundice", "dark urine", "fatigue"],
        "optional": ["abdominal pain", "nausea", "loss of appetite"]
    },
    "Diverticulitis": {
        "mandatory": ["left lower abdominal pain", "fever", "change in bowel habits"],
        "optional": ["nausea", "bloating", "rectal bleeding"]
    },
    "Constipation (Chronic)": {
        "mandatory": ["infrequent stools", "difficulty passing stool", "hard stools"],
        "optional": ["abdominal discomfort", "bloating", "straining"]
    },
    "Gastroesophageal Motility Disorder (Gastroparesis)": {
        "mandatory": ["early satiety", "nausea", "bloating"],
        "optional": ["vomiting", "weight loss", "abdominal pain"]
    },

    # --- ENDOCRINE / METABOLIC (8) ---
    "Diabetes Mellitus (Type 2)": {
        "mandatory": ["excessive thirst", "frequent urination", "increased hunger"],
        "optional": ["fatigue", "blurred vision", "slow healing sores"]
    },
    "Type 1 Diabetes": {
        "mandatory": ["polyuria", "polydipsia", "weight loss"],
        "optional": ["fatigue", "blurred vision", "nausea"]
    },
    "Hypothyroidism": {
        "mandatory": ["fatigue", "weight gain", "cold intolerance"],
        "optional": ["dry skin", "constipation", "hair loss"]
    },
    "Hyperthyroidism": {
        "mandatory": ["weight loss", "heat intolerance", "palpitations"],
        "optional": ["tremor", "anxiety", "sweating"]
    },
    "Cushing's Syndrome": {
        "mandatory": ["weight gain", "round face", "central obesity"],
        "optional": ["skin thinning", "easy bruising", "fatigue"]
    },
    "Addison's Disease": {
        "mandatory": ["fatigue", "weight loss", "hyperpigmentation"],
        "optional": ["low blood pressure", "salt craving", "nausea"]
    },
    "Hyperlipidemia": {
        "mandatory": ["high cholesterol (lab finding)", "xanthomas", "family history dyslipidemia"],
        "optional": ["chest pain", "fatigue", "none"]
    },
    "Hypoglycemia": {
        "mandatory": ["shaking", "sweating", "confusion"],
        "optional": ["hunger", "dizziness", "irritability"]
    },

    # --- NEUROLOGICAL / NEUROPSYCHIATRIC (12) ---
    "Migraine": {
        "mandatory": ["severe headache", "nausea", "sensitivity to light"],
        "optional": ["vomiting", "aura", "visual disturbance"]
    },
    "Tension Headache": {
        "mandatory": ["dull headache", "bilateral head pressure", "scalp tenderness"],
        "optional": ["neck pain", "fatigue", "mild sensitivity to light"]
    },
    "Cluster Headache": {
        "mandatory": ["severe unilateral headache", "eye watering", "nasal congestion"],
        "optional": ["restlessness", "sweating", "ptosis"]
    },
    "Epilepsy": {
        "mandatory": ["recurrent seizures", "loss of awareness", "abnormal movements"],
        "optional": ["post-ictal confusion", "tongue biting", "incontinence"]
    },
    "Stroke (Ischemic/Hemorrhagic)": {
        "mandatory": ["sudden face droop", "arm weakness", "speech difficulty"],
        "optional": ["confusion", "vision changes", "severe headache"]
    },
    "Parkinson's Disease": {
        "mandatory": ["resting tremor", "bradykinesia", "rigidity"],
        "optional": ["postural instability", "shuffling gait", "masked face"]
    },
    "Alzheimer's Disease": {
        "mandatory": ["memory loss", "difficulty with daily tasks", "disorientation"],
        "optional": ["language difficulty", "behavior changes", "wandering"]
    },
    "Multiple Sclerosis": {
        "mandatory": ["optic neuritis", "motor weakness", "sensory disturbance"],
        "optional": ["fatigue", "bladder dysfunction", "balance issues"]
    },
    "Peripheral Neuropathy": {
        "mandatory": ["numbness", "tingling", "burning sensation"],
        "optional": ["loss of sensation", "pain", "weakness"]
    },
    "Meningitis": {
        "mandatory": ["fever", "neck stiffness", "headache"],
        "optional": ["photophobia", "nausea", "altered mental status"]
    },
    " Guillain-Barre Syndrome": {
        "mandatory": ["ascending weakness", "areflexia", "paresthesia"],
        "optional": ["respiratory weakness", "facial weakness", "autonomic changes"]
    },
    "Anxiety Disorder": {
        "mandatory": ["excessive worry", "restlessness", "palpitations"],
        "optional": ["insomnia", "sweating", "difficulty concentrating"]
    },

    # --- INFECTIOUS DISEASES (10) ---
    "Dengue": {
        "mandatory": ["high fever", "headache", "pain behind eyes"],
        "optional": ["joint pain", "rash", "nausea"]
    },
    "Malaria": {
        "mandatory": ["high fever", "shaking chills", "sweating"],
        "optional": ["headache", "nausea", "muscle pain"]
    },
    "Typhoid Fever": {
        "mandatory": ["sustained fever", "headache", "abdominal pain"],
        "optional": ["constipation or diarrhea", "rose spots", "weakness"]
    },
    "Urinary Tract Infection (UTI)": {
        "mandatory": ["painful urination", "frequent urination", "urinary urgency"],
        "optional": ["cloudy urine", "pelvic pain", "blood in urine"]
    },
    "Sepsis": {
        "mandatory": ["fever or low temperature", "rapid heart rate", "rapid breathing"],
        "optional": ["confusion", "low urine output", "chills"]
    },
    "HIV Infection (Chronic)": {
        "mandatory": ["recurrent infections", "weight loss", "chronic fatigue"],
        "optional": ["night sweats", "lymphadenopathy", "thrush"]
    },
    "Influenza Complication (Secondary Bacterial Infection)": {
        "mandatory": ["high fever", "productive cough", "worsening symptoms after initial illness"],
        "optional": ["chest pain", "shortness of breath", "fatigue"]
    },
    "Hepatitis B (Chronic)": {
        "mandatory": ["jaundice", "fatigue", "dark urine"],
        "optional": ["abdominal pain", "loss of appetite", "nausea"]
    },
    "Shigellosis": {
        "mandatory": ["diarrhea", "abdominal cramp", "fever"],
        "optional": ["bloody stool", "nausea", "dehydration"]
    },
    "Rickettsial Infection (e.g., Scrub Typhus)": {
        "mandatory": ["fever", "eschar or rash", "headache"],
        "optional": ["myalgia", "cough", "lymphadenopathy"]
    },

    # --- DERMATOLOGICAL (8) ---
    "Eczema (Atopic Dermatitis)": {
        "mandatory": ["itchy skin", "dry skin", "red rash"],
        "optional": ["cracked skin", "oozing", "thickened patches"]
    },
    "Psoriasis": {
        "mandatory": ["scaly skin patches", "itchy skin", "well-demarcated plaques"],
        "optional": ["joint pain", "flaky skin", "dry skin"]
    },
    "Acne Vulgaris": {
        "mandatory": ["pimples", "blackheads", "oily skin"],
        "optional": ["painful nodules", "scarring", "whiteheads"]
    },
    "Contact Dermatitis": {
        "mandatory": ["localized rash", "itching", "redness"],
        "optional": ["blisters", "swelling", "pain"]
    },
    "Cellulitis": {
        "mandatory": ["localized redness", "pain", "warmth"],
        "optional": ["fever", "swelling", "lymphangitic streaking"]
    },
    "Fungal Skin Infection (Tinea)": {
        "mandatory": ["ring-shaped rash", "itching", "scaling"],
        "optional": ["redness", "peeling", "cracked skin"]
    },
    "Rosacea": {
        "mandatory": ["facial flushing", "persistent redness", "visible blood vessels"],
        "optional": ["papules and pustules", "eye irritation", "swelling"]
    },
    "Chickenpox (Varicella)": {
        "mandatory": ["itchy vesicular rash", "fever", "general malaise"],
        "optional": ["loss of appetite", "headache", "sore throat"]
    },

    # --- MUSCULOSKELETAL / RHEUMATOLOGIC (8) ---
    "Osteoarthritis": {
        "mandatory": ["joint pain", "joint stiffness", "worse with use"],
        "optional": ["reduced range of motion", "crepitus", "swelling"]
    },
    "Rheumatoid Arthritis": {
        "mandatory": ["joint pain", "morning stiffness", "symmetric joint swelling"],
        "optional": ["fatigue", "weight loss", "fever"]
    },
    "Gout": {
        "mandatory": ["sudden severe joint pain", "red swollen joint", "tenderness"],
        "optional": ["tophi", "fever", "limited mobility"]
    },
    "Fibromyalgia": {
        "mandatory": ["widespread pain", "fatigue", "sleep disturbance"],
        "optional": ["cognitive difficulties", "headache", "irritable bowel symptoms"]
    },
    "Tendonitis": {
        "mandatory": ["localized tendon pain", "worse with movement", "tenderness"],
        "optional": ["swelling", "reduced strength", "stiffness"]
    },
    "Polymyalgia Rheumatica": {
        "mandatory": ["proximal muscle pain", "morning stiffness", "elevated inflammatory markers (lab)"],
        "optional": ["fever", "fatigue", "weight loss"]
    },
    "Spondylosis (Degenerative Disc Disease)": {
        "mandatory": ["neck or back pain", "stiffness", "radiating limb pain"],
        "optional": ["numbness", "weakness", "reduced mobility"]
    },
    "Ankylosing Spondylitis": {
        "mandatory": ["chronic low back pain", "morning stiffness", "improvement with exercise"],
        "optional": ["reduced spinal mobility", "hip pain", "eye inflammation"]
    },

    # --- RENAL / UROLOGIC (4) ---
    "Acute Kidney Injury": {
        "mandatory": ["reduced urine output", "swelling", "fatigue"],
        "optional": ["nausea", "confusion", "shortness of breath"]
    },
    "Chronic Kidney Disease": {
        "mandatory": ["fatigue", "reduced urine output", "edema"],
        "optional": ["nausea", "itchy skin", "loss of appetite"]
    },
    "Kidney Stones (Nephrolithiasis)": {
        "mandatory": ["severe flank pain", "hematuria", "nausea"],
        "optional": ["urinary urgency", "fever", "vomiting"]
    },
    "Benign Prostatic Hyperplasia (BPH)": {
        "mandatory": ["weak urine stream", "incomplete bladder emptying", "frequent urination"],
        "optional": ["nocturia", "urinary urgency", "straining"]
    },

    # --- HEMATOLOGIC / IMMUNOLOGIC (4) ---
    "Anemia (Iron deficiency)": {
        "mandatory": ["fatigue", "pallor", "shortness of breath"],
        "optional": ["dizziness", "cold intolerance", "fast heartbeat"]
    },
    "Leukemia (Chronic)": {
        "mandatory": ["fatigue", "recurrent infections", "lymphadenopathy"],
        "optional": ["fever", "weight loss", "easy bruising"]
    },
    "Autoimmune Lupus (SLE)": {
        "mandatory": ["joint pain", "fatigue", "photosensitive rash"],
        "optional": ["fever", "hair loss", "kidney involvement"]
    },
    "Immune Thrombocytopenia (ITP)": {
        "mandatory": ["easy bruising", "petechiae", "low platelets (lab)"],
        "optional": ["mucosal bleeding", "fatigue", "heavy menses"]
    },

    # --- OPHTHALMOLOGIC / ENT (6) ---
    "Conjunctivitis (Pink Eye)": {
        "mandatory": ["eye redness", "eye discharge", "itchy or gritty sensation"],
        "optional": ["tearing", "eye pain", "blurred vision"]
    },
    "Otitis Media (Middle ear infection)": {
        "mandatory": ["ear pain", "fever", "reduced hearing"],
        "optional": ["ear discharge", "irritability", "tugging at ear"]
    },
    "Sinus Infection (Chronic Rhinosinusitis)": {
        "mandatory": ["nasal congestion", "facial pressure", "reduced smell"],
        "optional": ["postnasal drip", "cough", "fatigue"]
    },
    "Glaucoma (Open-angle)": {
        "mandatory": ["gradual loss of peripheral vision", "optic nerve changes (exam)", "increased intraocular pressure (exam)"],
        "optional": ["eye pain", "headache", "blurred vision"]
    },
    "Tonsillitis": {
        "mandatory": ["sore throat", "fever", "swollen tonsils"],
        "optional": ["difficulty swallowing", "bad breath", "ear pain"]
    },
    "Allergic Rhinitis": {
        "mandatory": ["sneezing", "runny nose", "itchy eyes"],
        "optional": ["nasal congestion", "postnasal drip", "cough"]
    },

    # --- PREGNANCY / OB-GYN (4) ---
    "Ectopic Pregnancy": {
        "mandatory": ["abdominal pain", "vaginal bleeding", "amenorrhea"],
        "optional": ["shoulder pain", "dizziness", "low blood pressure"]
    },
    "Preeclampsia": {
        "mandatory": ["high blood pressure", "proteinuria", "swelling of hands/face"],
        "optional": ["headache", "vision changes", "upper abdominal pain"]
    },
    "Pelvic Inflammatory Disease (PID)": {
        "mandatory": ["lower abdominal pain", "abnormal vaginal discharge", "fever"],
        "optional": ["painful intercourse", "irregular bleeding", "nausea"]
    },
    "Menopause (Symptomatic)": {
        "mandatory": ["hot flashes", "irregular periods", "night sweats"],
        "optional": ["mood changes", "sleep disturbance", "vaginal dryness"]
    },

    # --- INFANT / PEDIATRIC COMMON (4) ---
    "Bronchiolitis (Infants)": {
        "mandatory": ["wheezing", "difficulty breathing", "cough"],
        "optional": ["fever", "nasal congestion", "poor feeding"]
    },
    "Hand Foot Mouth Disease": {
        "mandatory": ["mouth sores", "rash on hands/feet", "fever"],
        "optional": ["loss of appetite", "irritability", "sore throat"]
    },
    "Kawasaki Disease": {
        "mandatory": ["fever lasting >5 days", "conjunctival injection", "oral mucosal changes"],
        "optional": ["rash", "swollen hands/feet", "lymphadenopathy"]
    },
    "Neonatal Jaundice (Physiologic)": {
        "mandatory": ["yellow skin", "yellow eyes", "onset in first week"],
        "optional": ["poor feeding", "lethargy", "dark urine"]
    },

    # --- MISCELLANEOUS / PUBLIC HEALTH (2) ---
    "Heat Exhaustion": {
        "mandatory": ["heavy sweating", "weakness", "cold clammy skin"],
        "optional": ["nausea", "headache", "dizziness"]
    },
    "Dehydration (Acute)": {
        "mandatory": ["dry mouth", "reduced urine output", "dizziness"],
        "optional": ["thirst", "weakness", "sunken eyes"]
    }
}


# Collect all unique symptoms from the rules
ALL_SYMPTOMS = set()
for d in DISEASE_RULES.values():
    ALL_SYMPTOMS.update(d["mandatory"])
    ALL_SYMPTOMS.update(d["optional"])
ALL_SYMPTOMS = sorted(list(ALL_SYMPTOMS))

def generate_dataset(num_samples_per_disease=200):
    data = []
    print(f"Generating training data for {len(DISEASE_RULES)} diseases...")
    
    for disease, rules in DISEASE_RULES.items():
        for _ in range(num_samples_per_disease):
            # Start with all 0s
            row = {sym: 0 for sym in ALL_SYMPTOMS}
            row["prognosis"] = disease
            
            # Add Mandatory Symptoms (High Probability: 90-100%)
            for sym in rules["mandatory"]:
                if random.random() < 0.95: 
                    row[sym] = 1
            
            # Add Optional Symptoms (Medium Probability: 30-50%)
            for sym in rules["optional"]:
                if random.random() < 0.40:
                    row[sym] = 1
            
            # Add Noise (Low Probability: 1% - Simulate unrelated complaints)
            for sym in ALL_SYMPTOMS:
                if row[sym] == 0 and random.random() < 0.01:
                    row[sym] = 1

            data.append(row)

    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    filename = "master_training_data.csv"
    df.to_csv(filename, index=False)
    
    print(f"✅ Generated {filename}")
    print(f"   - Rows: {len(df)}")
    print(f"   - Columns: {len(df.columns)}")
    print(f"   - Diseases: {len(DISEASE_RULES)}")
    print("\n👉 Now run: python train_model.py")