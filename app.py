from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# === Configuration ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'users.db')
TRAINING_CSV_PATH = os.path.join(BASE_DIR, 'Training.csv')

# === Initialize Database ===
def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                mobile TEXT NOT NULL,
                age INTEGER NOT NULL,
                address TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

init_db()

# === Disease Info Placeholder ===
disease_info = {
    'Fungal infection': {
        'description': 'A fungal infection is caused by fungi that invade the skin, nails, or mucous membranes. Common types include ringworm, athlete’s foot, and yeast infections.',
        'symptoms': ['itching', 'skin_rash', 'nodal_skin_eruptions'],
        'precautions': [
            'Keep the affected area clean and dry.',
            'Avoid sharing personal items like towels or clothing.',
            'Use antifungal creams or medications as prescribed.',
            'Wear loose, breathable clothing.'
        ]
    },
    'Allergy': {
        'description': 'An allergy is an immune system reaction to a substance (allergen) like pollen, dust, or certain foods, causing symptoms such as sneezing, itching, or swelling.',
        'symptoms': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
        'precautions': [
            'Avoid known allergens.',
            'Use air purifiers to reduce airborne allergens.',
            'Take antihistamines as recommended.',
            'Keep windows closed during high pollen seasons.'
        ]
    },
    'GERD': {
        'description': 'Gastroesophageal Reflux Disease (GERD) is a chronic condition where stomach acid flows back into the esophagus, causing heartburn and irritation.',
        'symptoms': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting'],
        'precautions': [
            'Avoid spicy, fatty, or acidic foods.',
            'Eat smaller meals more frequently.',
            'Avoid lying down immediately after eating.',
            'Elevate the head of your bed while sleeping.'
        ]
    },
    'Chronic cholestasis': {
        'description': 'Chronic cholestasis is a condition where bile flow from the liver is reduced or blocked, leading to symptoms like jaundice and itching.',
        'symptoms': ['yellowish_skin', 'dark_urine', 'itching', 'abdominal_pain'],
        'precautions': [
            'Follow a low-fat diet.',
            'Avoid alcohol consumption.',
            'Take medications as prescribed to improve bile flow.',
            'Consult a specialist for liver health monitoring.'
        ]
    },
    'Drug Reaction': {
        'description': 'A drug reaction is an adverse response to medication, ranging from mild rashes to severe allergic reactions.',
        'symptoms': ['skin_rash', 'itching', 'fever', 'malaise'],
        'precautions': [
            'Stop the suspected medication (consult a doctor first).',
            'Report reactions to your healthcare provider.',
            'Avoid self-medicating without professional advice.',
            'Keep a list of medications you’re allergic to.'
        ]
    },
    'Peptic ulcer disease': {
        'description': 'Peptic ulcer disease involves sores in the stomach lining or small intestine, often due to H. pylori infection or NSAID use.',
        'symptoms': ['stomach_pain', 'nausea', 'vomiting', 'indigestion'],
        'precautions': [
            'Avoid spicy and acidic foods.',
            'Limit NSAID use; consult a doctor for alternatives.',
            'Take prescribed antacids or antibiotics.',
            'Eat smaller, frequent meals.'
        ]
    },
    'AIDS': {
        'description': 'Acquired Immunodeficiency Syndrome (AIDS) is the advanced stage of HIV infection, weakening the immune system.',
        'symptoms': ['weight_loss', 'high_fever', 'fatigue', 'swelled_lymph_nodes'],
        'precautions': [
            'Practice safe sex and use condoms.',
            'Avoid sharing needles or unsterile equipment.',
            'Take antiretroviral therapy as prescribed.',
            'Regularly monitor health with a doctor.'
        ]
    },
    'Diabetes': {
        'description': 'Diabetes is a chronic condition affecting blood sugar regulation, either due to insufficient insulin (Type 1) or insulin resistance (Type 2).',
        'symptoms': ['excessive_hunger', 'polyuria', 'fatigue', 'blurred_and_distorted_vision'],
        'precautions': [
            'Monitor blood sugar levels regularly.',
            'Follow a balanced diet low in refined sugars.',
            'Exercise regularly to maintain weight.',
            'Take medications or insulin as prescribed.'
        ]
    },
    'Gastroenteritis': {
        'description': 'Gastroenteritis is inflammation of the stomach and intestines, often caused by viral or bacterial infections.',
        'symptoms': ['diarrhoea', 'vomiting', 'abdominal_pain', 'dehydration'],
        'precautions': [
            'Stay hydrated with water or oral rehydration solutions.',
            'Wash hands frequently to prevent spread.',
            'Avoid contaminated food or water.',
            'Rest and avoid solid foods temporarily.'
        ]
    },
    'Bronchial Asthma': {
        'description': 'Bronchial asthma is a chronic respiratory condition causing airway inflammation and breathing difficulties.',
        'symptoms': ['breathlessness', 'cough', 'chest_pain', 'phlegm'],
        'precautions': [
            'Avoid triggers like dust, smoke, or allergens.',
            'Use an inhaler as prescribed.',
            'Practice breathing exercises.',
            'Keep rescue medication accessible.'
        ]
    },
    'Hypertension': {
        'description': 'Hypertension, or high blood pressure, is a condition where blood pressure is persistently elevated, straining the heart and vessels.',
        'symptoms': ['headache', 'dizziness', 'chest_pain', 'fatigue'],
        'precautions': [
            'Reduce salt intake in your diet.',
            'Exercise regularly to maintain a healthy weight.',
            'Limit alcohol and avoid smoking.',
            'Monitor blood pressure regularly.'
        ]
    },
    'Migraine': {
        'description': 'Migraine is a neurological condition characterized by intense, recurring headaches often accompanied by sensory disturbances.',
        'symptoms': ['headache', 'nausea', 'visual_disturbances', 'dizziness'],
        'precautions': [
            'Identify and avoid triggers (e.g., stress, certain foods).',
            'Maintain a regular sleep schedule.',
            'Stay hydrated.',
            'Use prescribed medications for relief.'
        ]
    },
    'Cervical spondylosis': {
        'description': 'Cervical spondylosis is age-related wear and tear of the spinal disks in the neck, leading to stiffness and pain.',
        'symptoms': ['neck_pain', 'stiff_neck', 'headache', 'weakness_in_limbs'],
        'precautions': [
            'Maintain good posture while sitting or standing.',
            'Perform neck exercises as advised.',
            'Avoid heavy lifting.',
            'Use a supportive pillow for sleep.'
        ]
    },
    'Paralysis (brain hemorrhage)': {
        'description': 'Paralysis from brain hemorrhage occurs when bleeding in the brain damages areas controlling movement.',
        'symptoms': ['weakness_of_one_body_side', 'loss_of_balance', 'slurred_speech', 'headache'],
        'precautions': [
            'Control blood pressure to prevent recurrence.',
            'Follow rehabilitation therapy.',
            'Avoid smoking and excessive alcohol.',
            'Seek immediate medical help for symptoms.'
        ]
    },
    'Jaundice': {
        'description': 'Jaundice is a condition where the skin and eyes turn yellow due to high bilirubin levels, often linked to liver issues.',
        'symptoms': ['yellowish_skin', 'yellowing_of_eyes', 'dark_urine', 'fatigue'],
        'precautions': [
            'Avoid alcohol and fatty foods.',
            'Stay hydrated.',
            'Follow a liver-friendly diet.',
            'Seek medical evaluation for underlying causes.'
        ]
    },
    'Malaria': {
        'description': 'Malaria is a mosquito-borne parasitic infection causing fever and chills, prevalent in tropical regions.',
        'symptoms': ['high_fever', 'chills', 'sweating', 'headache'],
        'precautions': [
            'Use mosquito nets and repellents.',
            'Take antimalarial prophylaxis if traveling to endemic areas.',
            'Eliminate standing water to reduce mosquito breeding.',
            'Seek prompt treatment if symptoms appear.'
        ]
    },
    'Chicken pox': {
        'description': 'Chicken pox is a viral infection causing an itchy rash with blisters, primarily affecting children.',
        'symptoms': ['skin_rash', 'itching', 'high_fever', 'fatigue'],
        'precautions': [
            'Avoid scratching to prevent scarring.',
            'Isolate to prevent spread.',
            'Use calamine lotion for itching.',
            'Vaccinate children to prevent infection.'
        ]
    },
    'Dengue': {
        'description': 'Dengue is a mosquito-borne viral disease causing high fever and severe joint pain, sometimes leading to complications.',
        'symptoms': ['high_fever', 'joint_pain', 'headache', 'pain_behind_the_eyes'],
        'precautions': [
            'Use mosquito repellents and nets.',
            'Wear long-sleeved clothing.',
            'Stay hydrated during fever.',
            'Monitor for severe symptoms and seek medical care.'
        ]
    },
    'Typhoid': {
        'description': 'Typhoid is a bacterial infection (Salmonella typhi) spread through contaminated food or water, affecting the digestive system.',
        'symptoms': ['high_fever', 'abdominal_pain', 'weakness', 'loss_of_appetite'],
        'precautions': [
            'Drink only boiled or bottled water.',
            'Avoid raw or undercooked foods.',
            'Wash hands frequently.',
            'Get vaccinated if at risk.'
        ]
    },
    'Hepatitis A': {
        'description': 'Hepatitis A is a viral liver infection spread through contaminated food or water, causing inflammation.',
        'symptoms': ['yellowish_skin', 'fatigue', 'nausea', 'abdominal_pain'],
        'precautions': [
            'Practice good hygiene and handwashing.',
            'Drink safe, clean water.',
            'Avoid raw shellfish or unwashed produce.',
            'Get vaccinated before travel to endemic areas.'
        ]
    },
    'Hepatitis B': {
        'description': 'Hepatitis B is a viral infection affecting the liver, transmitted through blood or bodily fluids.',
        'symptoms': ['yellowish_skin', 'dark_urine', 'fatigue', 'joint_pain'],
        'precautions': [
            'Get vaccinated against Hepatitis B.',
            'Avoid sharing needles or razors.',
            'Practice safe sex.',
            'Screen blood transfusions.'
        ]
    },
    'Hepatitis C': {
        'description': 'Hepatitis C is a blood-borne viral infection causing chronic liver inflammation, often asymptomatic initially.',
        'symptoms': ['fatigue', 'yellowish_skin', 'dark_urine', 'abdominal_pain'],
        'precautions': [
            'Avoid sharing personal items like toothbrushes.',
            'Use sterile equipment for tattoos or piercings.',
            'Screen for infection if at risk.',
            'Follow antiviral treatment if diagnosed.'
        ]
    },
    'Hepatitis D': {
        'description': 'Hepatitis D is a liver infection requiring Hepatitis B co-infection, worsening liver damage.',
        'symptoms': ['yellowish_skin', 'fatigue', 'abdominal_pain', 'nausea'],
        'precautions': [
            'Prevent Hepatitis B through vaccination.',
            'Avoid blood exposure.',
            'Monitor liver health if co-infected.',
            'Seek specialized medical care.'
        ]
    },
    'Hepatitis E': {
        'description': 'Hepatitis E is a waterborne viral infection affecting the liver, common in areas with poor sanitation.',
        'symptoms': ['yellowish_skin', 'fatigue', 'nausea', 'dark_urine'],
        'precautions': [
            'Drink clean, boiled water.',
            'Avoid contaminated food.',
            'Practice good sanitation.',
            'Rest and hydrate during recovery.'
        ]
    },
    'Alcoholic hepatitis': {
        'description': 'Alcoholic hepatitis is liver inflammation caused by excessive alcohol consumption.',
        'symptoms': ['yellowish_skin', 'abdominal_pain', 'nausea', 'history_of_alcohol_consumption'],
        'precautions': [
            'Stop alcohol consumption immediately.',
            'Follow a nutritious diet.',
            'Seek medical treatment for liver support.',
            'Join a support program for alcohol cessation.'
        ]
    },
    'Tuberculosis': {
        'description': 'Tuberculosis (TB) is a bacterial infection (Mycobacterium tuberculosis) primarily affecting the lungs.',
        'symptoms': ['cough', 'chest_pain', 'blood_in_sputum', 'weight_loss'],
        'precautions': [
            'Complete the full course of TB medication.',
            'Cover your mouth when coughing.',
            'Improve ventilation in living spaces.',
            'Get tested if exposed.'
        ]
    },
    'Common Cold': {
        'description': 'The common cold is a viral upper respiratory infection causing nasal congestion and mild symptoms.',
        'symptoms': ['runny_nose', 'cough', 'throat_irritation', 'congestion'],
        'precautions': [
            'Wash hands frequently.',
            'Avoid close contact with sick individuals.',
            'Stay hydrated and rest.',
            'Use over-the-counter remedies for relief.'
        ]
    },
    'Pneumonia': {
        'description': 'Pneumonia is an infection causing inflammation in the lung air sacs, often bacterial or viral.',
        'symptoms': ['cough', 'high_fever', 'breathlessness', 'chest_pain'],
        'precautions': [
            'Get vaccinated (e.g., pneumococcal vaccine).',
            'Avoid smoking to protect lung health.',
            'Treat promptly with antibiotics if bacterial.',
            'Rest and maintain hydration.'
        ]
    },
    'Dimorphic hemorrhoids (piles)': {
        'description': 'Hemorrhoids are swollen veins in the rectum or anus, causing discomfort and sometimes bleeding.',
        'symptoms': ['pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'],
        'precautions': [
            'Increase fiber intake to soften stools.',
            'Stay hydrated to prevent constipation.',
            'Avoid straining during bowel movements.',
            'Use topical treatments for relief.'
        ]
    },
    'Heart attack': {
        'description': 'A heart attack occurs when blood flow to the heart is blocked, damaging heart muscle.',
        'symptoms': ['chest_pain', 'breathlessness', 'sweating', 'fast_heart_rate'],
        'precautions': [
            'Seek emergency medical help immediately.',
            'Maintain a heart-healthy diet.',
            'Exercise regularly.',
            'Manage stress and cholesterol levels.'
        ]
    },
    'Varicose veins': {
        'description': 'Varicose veins are enlarged, twisted veins, often in the legs, due to weak vein walls or valves.',
        'symptoms': ['swollen_legs', 'prominent_veins_on_calf', 'painful_walking'],
        'precautions': [
            'Elevate legs to improve circulation.',
            'Wear compression stockings.',
            'Avoid standing or sitting for long periods.',
            'Exercise to boost blood flow.'
        ]
    },
    'Hypothyroidism': {
        'description': 'Hypothyroidism is an underactive thyroid gland causing slow metabolism and fatigue.',
        'symptoms': ['weight_gain', 'fatigue', 'cold_hands_and_feets', 'brittle_nails'],
        'precautions': [
            'Take thyroid hormone replacement as prescribed.',
            'Eat iodine-rich foods (e.g., fish).',
            'Monitor thyroid levels regularly.',
            'Avoid excessive stress.'
        ]
    },
    'Hyperthyroidism': {
        'description': 'Hyperthyroidism is an overactive thyroid gland speeding up metabolism.',
        'symptoms': ['weight_loss', 'fast_heart_rate', 'sweating', 'anxiety'],
        'precautions': [
            'Take antithyroid medications as prescribed.',
            'Avoid caffeine and stimulants.',
            'Monitor thyroid function regularly.',
            'Rest to manage symptoms.'
        ]
    },
    'Hypoglycemia': {
        'description': 'Hypoglycemia is low blood sugar, often affecting diabetics, causing shakiness and confusion.',
        'symptoms': ['sweating', 'shivering', 'dizziness', 'irritability'],
        'precautions': [
            'Eat regular, balanced meals.',
            'Carry a quick sugar source (e.g., candy).',
            'Monitor blood sugar levels.',
            'Adjust insulin or medication with a doctor.'
        ]
    },
    'Osteoarthritis': {
        'description': 'Osteoarthritis is a degenerative joint disease causing cartilage breakdown and pain.',
        'symptoms': ['joint_pain', 'swelling_joints', 'movement_stiffness', 'painful_walking'],
        'precautions': [
            'Maintain a healthy weight to reduce joint stress.',
            'Exercise to strengthen muscles around joints.',
            'Use pain relief as prescribed.',
            'Avoid overexertion of affected joints.'
        ]
    },
    'Arthritis': {
        'description': 'Arthritis is joint inflammation, including types like rheumatoid arthritis, causing pain and stiffness.',
        'symptoms': ['joint_pain', 'swelling_joints', 'stiff_neck', 'muscle_weakness'],
        'precautions': [
            'Stay active with low-impact exercises.',
            'Use anti-inflammatory medications as advised.',
            'Apply heat or cold to affected areas.',
            'Consult a rheumatologist for management.'
        ]
    },
    '(Vertigo) Paroxysmal Positional Vertigo': {
        'description': 'Paroxysmal Positional Vertigo is a disorder causing brief episodes of dizziness triggered by head position changes.',
        'symptoms': ['spinning_movements', 'dizziness', 'loss_of_balance', 'nausea'],
        'precautions': [
            'Perform Epley maneuver under guidance.',
            'Avoid sudden head movements.',
            'Sleep with head elevated.',
            'Consult a specialist for persistent symptoms.'
        ]
    },
    'Acne': {
        'description': 'Acne is a skin condition involving clogged pores, leading to pimples and inflammation.',
        'symptoms': ['pus_filled_pimples', 'blackheads', 'scurring', 'skin_rash'],
        'precautions': [
            'Wash face gently twice daily.',
            'Avoid touching the face with dirty hands.',
            'Use non-comedogenic skincare products.',
            'Consult a dermatologist for severe cases.'
        ]
    },
    'Urinary tract infection': {
        'description': 'A urinary tract infection (UTI) is a bacterial infection in the urinary system, causing pain and frequent urination.',
        'symptoms': ['burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine'],
        'precautions': [
            'Drink plenty of water.',
            'Urinate frequently; don’t hold it in.',
            'Maintain good hygiene.',
            'Take antibiotics as prescribed.'
        ]
    },
    'Psoriasis': {
        'description': 'Psoriasis is an autoimmune condition causing rapid skin cell buildup, leading to scaly patches.',
        'symptoms': ['skin_rash', 'skin_peeling', 'silver_like_dusting', 'inflammatory_nails'],
        'precautions': [
            'Moisturize skin regularly.',
            'Avoid triggers like stress or injury.',
            'Use prescribed topical treatments.',
            'Protect skin from harsh weather.'
        ]
    },
    'Impetigo': {
        'description': 'Impetigo is a contagious bacterial skin infection causing red sores and crusty patches.',
        'symptoms': ['red_sore_around_nose', 'yellow_crust_ooze', 'blister', 'itching'],
        'precautions': [
            'Keep the area clean and covered.',
            'Avoid scratching or touching sores.',
            'Use antibiotic ointment as prescribed.',
            'Wash hands frequently to prevent spread.'
        ]
    }
}
note = (
    "Note: This information is system-generated and intended for general awareness only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider for personalized guidance."
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO users (fullname, email, password, mobile, age, address) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (data['fullname'], data['email'], data['password'], 
                     data['mobile'], data['age'], data['address']))
                conn.commit()
                return redirect('/login')
            except sqlite3.IntegrityError:
                return render_template('signup.html', error="Email already exists.")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
            user = cursor.fetchone()
        if user:
            session['user_id'] = user[0]
            return redirect('/dashboard')
        return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/login')

# === ML Models Training ===
data = pd.read_csv(TRAINING_CSV_PATH)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
nb_model = BernoulliNB().fit(X_train, y_train)
svm_model = SVC(kernel='linear', random_state=42).fit(X_train, y_train)

all_symptoms = list(X.columns)
symptom_map = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
symptom_cooccurrence = X.T.dot(X)
np.fill_diagonal(symptom_cooccurrence.values, 0)

def get_related_symptoms(selected, top_n=5):
    indices = [symptom_map[s] for s in selected if s in symptom_map]
    if not indices:
        return []
    scores = symptom_cooccurrence.iloc[indices].sum()
    related_indices = scores.argsort()[-top_n:][::-1]
    return [all_symptoms[i] for i in related_indices if all_symptoms[i] not in selected][:top_n]

@app.route('/suggest_symptoms', methods=['POST'])
def suggest_symptoms():
    if 'user_id' not in session:
        return redirect('/login')
    description = request.get_json().get('description', '').lower()
    vectorizer = TfidfVectorizer().fit_transform([description] + all_symptoms)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top = sorted(zip(all_symptoms, similarities), key=lambda x: x[1], reverse=True)
    return jsonify([s for s, sim in top if sim > 0.1][:3])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect('/login')

    selected = request.form.getlist('symptoms')
    suggestions = get_related_symptoms(selected)

    if suggestions and 'confirmed' not in request.form:
        return render_template('confirm_symptoms.html', 
                               selected_symptoms=selected, 
                               suggested_symptoms=suggestions)

    final_symptoms = selected + request.form.getlist('suggested_symptoms')
    input_vector = [1 if s in final_symptoms else 0 for s in all_symptoms]
    input_array = np.array([input_vector])

    dt_pred = le.inverse_transform(dt_model.predict(input_array))[0]
    rf_pred = le.inverse_transform(rf_model.predict(input_array))[0]
    nb_pred = le.inverse_transform(nb_model.predict(input_array))[0]
    svm_pred = le.inverse_transform(svm_model.predict(input_array))[0]

    prediction_counts = Counter([dt_pred, rf_pred, nb_pred, svm_pred])
    majority_disease, votes = prediction_counts.most_common(1)[0]

    predictions = {
        'Decision Tree': dt_pred,
        'Random Forest': rf_pred,
        'Naive Bayes': nb_pred,
        'SVM': svm_pred,
        'Majority Vote': majority_disease
    }

    info = disease_info.get(majority_disease, {})
    expected = info.get('symptoms', [])
    match = [s for s in final_symptoms if s in expected]
    mismatch = [s for s in final_symptoms if s not in expected]
    match_ratio = len(match) / len(expected) if expected else 0

    warning = None
    if final_symptoms and (match_ratio < 0.5 or (votes <= 2 and len(set(prediction_counts)) > 2)):
        warning = (f"The symptoms you selected may not fully align with the predicted disease: {majority_disease}. "
                   "Check the comparison below for better understanding.")

    return render_template('result.html',
        predictions=predictions,
        predicted_disease=majority_disease,
        description=info.get('description', 'No description available.'),
        precautions=info.get('precautions', ['Consult a doctor.']),
        selected_symptoms=final_symptoms,
        expected_symptoms=expected,
        matching_symptoms=match,
        unmatched_symptoms=mismatch,
        warning_message=warning,
        note=note
    )

if __name__ == '__main__':
    app.run(debug=True)
