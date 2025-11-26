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

def purify_data(df):
    print("   🧹 Purifying data...")
    synonyms = {
        'abdominal distention': ['stomach bloating', 'swollen abdomen'],
        'shortness of breath': ['dyspnea', 'difficulty breathing'],
        'cold': ['feeling cold', 'chills'],
        'fever': ['feeling hot', 'high fever'],  
        'dizziness': ['lightheadedness', 'vertigo'], 
        'eye pain': ['pain in eye', 'eye burns or stings'],
        'urinary issues': ['polyuria', 'frequent urination']
    }
    for keep, drop_list in synonyms.items():
        if keep not in df.columns: continue
        cols_to_merge = [c for c in drop_list if c in df.columns]
        if cols_to_merge:
            df[keep] = df[[keep] + cols_to_merge].max(axis=1)
            df = df.drop(cols_to_merge, axis=1)

    trash_columns = [
        'feeling ill', 'restlessness', 'irregular belly button', 'white discharge from eye', 
        'scanty menstrual flow', 'neck weakness', 'ankle stiffness or tightness', 'back swelling', 
        'pus in sputum', 'infrequent menstruation', 'itching of scrotum', 'jaw pain', 
        'mass on vulva', 'elbow lump or mass', 'eyelid retracted', 'change in skin mole size or color', 
        'tongue bleeding', 'bleeding in mouth', 'posture problems', 'knee cramps or spasms', 
        'discharge in stools', 'feet turned in', 'incontinence of stool', 'foot or toe cramps or spasms', 
        'hip swelling', 'nailbiting', 'mass on ear', 'throat irritation', 'swollen tongue', 
        'elbow stiffness or tightness', 'skin oiliness', 'sleepwalking', 'thirst', 'pupils unequal', 
        'hip lump or mass', 'low back swelling', 'hip weakness', 'underweight', 'abnormal appearing tongue', 
        'arm cramps or spasms', 'pallor', 'shoulder cramps or spasms', 'skin pain', 'nose deformity', 
        'lump over jaw', 'problems with orgasm', 'stuttering or stammering', 'skin on head or neck looks infected', 
        'low back stiffness or tightness', 'tongue pain', 'joint stiffness or tightness', 
        'abnormal size or shape of ear', 'pus in urine', 'low back weakness', 'elbow cramps or spasms', 
        'wrist weakness'
    ]
    df = df.drop(columns=[c for c in trash_columns if c in df.columns], errors='ignore')
    return df

def train():
    print(f"1. Loading Data...")
    df = pd.read_csv(DATA_FILE).fillna(0)
    
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).map({'M': 1, 'Male': 1, '1': 1, 'F': 0, 'Female': 0, '0': 0}).fillna(0).astype('int8')

    df = purify_data(df)
    
    class_counts = df['prognosis'].value_counts()
    valid_diseases = class_counts[class_counts >= 2].index 
    df = df[df['prognosis'].isin(valid_diseases)]

    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    print("2. Training COMPACT HIGH-IQ Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # --- SAFE CONFIGURATION ---
    model = RandomForestClassifier(
        n_estimators=150,      # High Intelligence
        max_depth=22,          # Strict Limit (Prevents 500MB bloat)
        min_samples_leaf=2,    # Prunes noise (Saves 40% space)
        class_weight='balanced', 
        n_jobs=-1,             
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(f"   -> Accuracy: {model.score(X_test, y_test):.2%}")

    print("3. Saving Compressed Brain (Max Compression)...")
    # Compress=9 takes longer to save, but makes the file TINY.
    joblib.dump(model, MODEL_FILE, compress=9)
    
    size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
    print(f"   -> Model Size: {size_mb:.2f} MB")

    if size_mb > 99:
        print("   ⚠️ STOP! Still too big. Do not push.")
    else:
        print("   ✅ PERFECT! Safe for GitHub.")
    
    # Tiny Meta
    d_map = {}
    for disease in df['prognosis'].unique():
         d_rows = df[df['prognosis'] == disease]
         nums = d_rows.select_dtypes(include=[np.number]).sum()
         top = nums[nums > (len(d_rows)*0.2)].index.tolist()
         d_map[disease] = [x for x in top if x not in ['age', 'gender']]
         
    meta = {
        'columns': X.columns.tolist(),
        'classes': model.classes_.tolist(),
        'disease_symptom_map': d_map
    }
    joblib.dump(meta, META_FILE, compress=9)

if __name__ == "__main__":
    train()