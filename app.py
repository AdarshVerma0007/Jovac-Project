from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # To save/load models

app = Flask(__name__)

# Load the pre-trained model and LabelEncoder
model = joblib.load('cow_disease_model.pkl')
le = joblib.load('label_encoder.pkl')

# Symptoms and Precautions for each disease
symptoms_precautions = {
    'Mastitis': {
        'Symptoms': 'Swollen udder, Decreased milk production, Milk with blood or pus',
        'Precautions': 'Proper milking hygiene, Use clean equipment, Antibiotic treatment'
    },
    'Bovine Respiratory Disease (BRD)': {
        'Symptoms': 'Coughing, Nasal discharge, Fever, Lethargy',
        'Precautions': 'Vaccination, Avoid overcrowding, Ensure proper ventilation'
    },
   
}
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/templates/cow.html')
def cow():
    return render_template('cow.html')
@app.route('/predict', methods=['POST'])
def predict():
    
    pulse_rate = float(request.form['pulse'])
    temperature = float(request.form['body_temperature'])
    ecg_voltage = float(request.form['ecg'])

    
    input_data = pd.DataFrame([[pulse_rate, temperature, ecg_voltage]],
                              columns=['Pulse Rate', 'Temperature (°C)', 'ECG Voltage (mV)'])

    predicted_disease_encoded = model.predict(input_data)[0]
    predicted_disease = le.inverse_transform([predicted_disease_encoded])[0]

    
    if predicted_disease == 'Healthy':
        symptoms = 'No symptoms, the cow is healthy.'
        precautions = 'No specific precautions needed, maintain regular care and monitoring.'
    else:
        
        symptoms = symptoms_precautions.get(predicted_disease, {}).get('Symptoms', 'Symptoms not available')
        precautions = symptoms_precautions.get(predicted_disease, {}).get('Precautions', 'Precautions not available')

    return render_template('result.html', disease=predicted_disease, symptoms=symptoms, precautions=precautions)


if __name__ == '__main__':
    app.run(port='8080', debug=True)
