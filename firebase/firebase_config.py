import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRED_PATH = os.path.join(BASE_DIR, 'firebase', 'serviceAccountKey.json')

# Get your database URL from Firebase Console
# Go to Realtime Database -> Copy the URL at the top
# It looks like: https://xai-risk-framework-default-rtdb.firebaseio.com/

DATABASE_URL = "https://xai-risk-framework-default-rtdb.firebaseio.com/"


def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(CRED_PATH)
        firebase_admin.initialize_app(cred, {
            'databaseURL': DATABASE_URL
        })
        print("Firebase initialized successfully")


def save_prediction(company_name, input_data, prediction_result):
    initialize_firebase()
    ref = db.reference('predictions')
    
    record = {
        'company_name': company_name,
        'input_data': input_data,
        'risk_level': prediction_result['risk_level'],
        'risk_label': prediction_result['risk_label'],
        'confidence': prediction_result['confidence'],
        'probabilities': prediction_result['probabilities'],
        'timestamp': datetime.now().isoformat()
    }
    
    new_ref = ref.push(record)
    return new_ref.key


def get_all_predictions():
    initialize_firebase()
    ref = db.reference('predictions')
    data = ref.get()
    
    if data is None:
        return []
    
    predictions = []
    for key, value in data.items():
        value['id'] = key
        predictions.append(value)
    
    # Sort by timestamp (newest first)
    predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return predictions


def delete_prediction(prediction_id):
    initialize_firebase()
    ref = db.reference(f'predictions/{prediction_id}')
    ref.delete()


def get_prediction_stats():
    predictions = get_all_predictions()
    
    stats = {
        'total': len(predictions),
        'Low Risk': 0,
        'Medium Risk': 0,
        'High Risk': 0,
        'Critical Risk': 0
    }
    
    for pred in predictions:
        label = pred.get('risk_label', '')
        if label in stats:
            stats[label] += 1
    
    return stats