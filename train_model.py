import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# CONFIGURATION
DATA_FILE = 'master_training_data.csv'
MODEL_FILE = 'model.joblib'
META_FILE = 'meta.joblib'

# --- PURIFIER FUNCTION (Standard) ---
def purify_data(df):
    print("   🧹 Purifying data...")
    initial_cols = len(df.columns)

    # 1. FIX TECHNICAL DUPLICATES
    if 'regurgitation.1' in df.columns and 'regurgitation' in df.columns:
        df['regurgitation'] = df[['regurgitation', 'regurgitation.1']].max(axis=1)
        df = df.drop('regurgitation.1', axis=1)

    # 2. GROUP SYNONYMS
    synonyms = {
        'abdominal distention': ['stomach bloating', 'swollen abdomen'],
        'shortness of breath': ['dyspnea', 'difficulty breathing', 'hurts to breath'],
        'cold': ['feeling cold', 'chills', 'feeling hot and cold'],
        'fever': ['feeling hot', 'high fever'],  
        'dizziness': ['lightheadedness', 'vertigo'], 
        'eye pain': ['pain in eye', 'eye burns or stings', 'itchiness of eye', 'eye strain'],
        'urinary issues': ['polyuria', 'frequent urination']
    }

    for keep, drop_list in synonyms.items():
        if keep not in df.columns: continue
        cols_to_merge = [c for c in drop_list if c in df.columns]
        if cols_to_merge:
            df[keep] = df[[keep] + cols_to_merge].max(axis=1)
            df = df.drop(cols_to_merge, axis=1)

    # 3. DROP TRASH
    trash_columns = [
        'feeling ill', 'restlessness', 'irregular belly button', 
        'white discharge from eye', 'scanty menstrual flow', 
        'neck weakness', 'ankle stiffness or tightness', 
        'back swelling', 'pus in sputum', 'infrequent menstruation', 
        'itching of scrotum', 'jaw pain', 'mass on vulva', 'elbow lump or mass', 
        'eyelid retracted', 'change in skin mole size or color', 
        'tongue bleeding', 'bleeding in mouth', 'posture problems', 
        'knee cramps or spasms', 'disturbance of smell or taste', 
        'discharge in stools', 'feet turned in', 
        'incontinence of stool', 'foot or toe cramps or spasms', 
        'hip swelling', 'nailbiting', 'mass on ear', 'throat irritation', 
        'swollen tongue', 'elbow stiffness or tightness', 'skin oiliness', 
        'sleepwalking', 'thirst', 'pupils unequal', 'hip lump or mass', 
        'low back swelling', 'hip weakness', 
        'underweight', 'abnormal appearing tongue', 'arm cramps or spasms', 
        'pallor', 'shoulder cramps or spasms', 'skin pain', 'nose deformity', 
        'lump over jaw', 'problems with orgasm', 'stuttering or stammering', 
        'skin on head or neck looks infected', 'low back stiffness or tightness', 
        'tongue pain', 'joint stiffness or tightness', 'abnormal size or shape of ear', 
        'pus in urine', 'low back weakness', 'elbow cramps or spasms', 'wrist weakness'
    ]
    
    df = df.drop(columns=[c for c in trash_columns if c in df.columns], errors='ignore')

    print(f"   ✨ Purified! Reduced columns from {initial_cols} to {len(df.columns)}")
    return df

def train():
    print(f"1. Loading Data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE).fillna(0)
    
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).map({
            'M': 1, 'Male': 1, '1': 1, '1.0': 1,
            'F': 0, 'Female': 0, '0': 0, '0.0': 0
        }).fillna(0).astype('int8')

    df = purify_data(df)

    # Filter Rare Diseases (Keep anything with at least 2 examples)
    class_counts = df['prognosis'].value_counts()
    valid_diseases = class_counts[class_counts >= 2].index 
    df = df[df['prognosis'].isin(valid_diseases)]

    print("2. Building Knowledge Base...")
    disease_symptom_map = {}
    for disease in df['prognosis'].unique():
        d_rows = df[df['prognosis'] == disease]
        numeric_cols = d_rows.select_dtypes(include=[np.number])
        symptom_sums = numeric_cols.sum()
        # Keep symptoms > 20% freq
        common_syms = symptom_sums[symptom_sums > (len(d_rows) * 0.2)].index.tolist()
        common_syms = [x for x in common_syms if x not in ['age', 'gender']]
        disease_symptom_map[disease] = common_syms

    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    print("3. Training BALANCED Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # --- BALANCED CONFIGURATION ---
    # We increase trees and depth for accuracy.
    # We REMOVE CalibratedClassifierCV to save RAM (that was the real memory killer).
    model = RandomForestClassifier(
        n_estimators=80,      # Bumped up from 20 -> 80 (Smarter)
        max_depth=25,         # Bumped up from 10 -> 25 (Deeper thinking)
        min_samples_leaf=1,   # Restore detail
        class_weight='balanced', 
        n_jobs=1,             # Keep 1 CPU for RAM safety
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print(f"   -> Accuracy: {model.score(X_test, y_test):.2%}")

    print("4. Saving Balanced Brain...")
    # Compress=3 is a good balance
    joblib.dump(model, MODEL_FILE, compress=3)
    
    meta = {
        'columns': X.columns.tolist(),
        'classes': model.classes_.tolist(),
        'disease_symptom_map': disease_symptom_map
    }
    joblib.dump(meta, META_FILE, compress=3)
    print("✅ Done. High Accuracy + Low RAM footprint.")

if __name__ == "__main__":
    train()