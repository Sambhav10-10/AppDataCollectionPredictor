"""
Flask API for App Data Collection Prediction Model
Provides endpoints for predictions and model comparison
"""

import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models
models = {}
model_accuracies = {}
feature_names = []
label_names = []
X_test = None
Y_test = None
dataset_apps = []  # Store all apps for dropdown

# ============================================================================
# LOAD AND TRAIN MODELS
# ============================================================================

def load_data(filepath):
    """Load JSON dataset"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_features_and_labels_no_leakage(dataset):
    """Extract features and labels WITHOUT data leakage"""
    
    features = []
    labels = []
    
    for app in dataset:
        app_info = app.get('appInfo', {})
        collected_data = app_info.get('collectedData', [])
        shared_data = app_info.get('sharedData', [])
        security_practices = app_info.get('securityPractices', [])
        
        # Extract output labels
        collected_types = {item.get('type') for item in collected_data}
        
        label_personal = 1 if 'Personal info' in collected_types else 0
        label_location = 1 if 'Location' in collected_types else 0
        label_contacts = 1 if 'Contacts' in collected_types else 0
        label_financial = 1 if 'Financial info' in collected_types else 0
        
        # Extract input features (NO LEAKAGE)
        
        # Feature 1: shared_with_third_parties
        sensitive_types = {'Personal info', 'Location', 'Contacts', 'Financial info'}
        shared_sensitive = any(
            item.get('type') in sensitive_types 
            for item in shared_data
        )
        feature_shared = 1 if shared_sensitive else 0
        
        # Feature 2: used_for_advertising
        advertising_in_collected = any(
            'Advertising or marketing' in item.get('purpose', '')
            for item in collected_data
        )
        advertising_in_shared = any(
            'Advertising or marketing' in item.get('purpose', '')
            for item in shared_data
        )
        feature_advertising = 1 if (advertising_in_collected or advertising_in_shared) else 0
        
        # Feature 3: encryption_enabled
        feature_encryption = 1 if any('encrypt' in str(p).lower() for p in security_practices) else 0
        
        # Feature 4: data_deletion_allowed
        feature_deletion = 1 if any('delete' in str(p).lower() for p in security_practices) else 0
        
        # Feature 5: collects_sensitive_data
        feature_sensitive = 1 if bool(sensitive_types & collected_types) else 0
        
        # Feature 6: num_data_types_collected
        feature_num_types = len(collected_types)
        
        # Feature 7: collects_multiple_types
        feature_multiple = 1 if len(collected_types) > 2 else 0
        
        # Feature 8: num_data_types_shared
        feature_num_shared = len({item.get('type') for item in shared_data})
        
        features.append([
            feature_shared,
            feature_advertising,
            feature_encryption,
            feature_deletion,
            feature_sensitive,
            feature_num_types,
            feature_multiple,
            feature_num_shared
        ])
        
        labels.append([
            label_personal,
            label_location,
            label_contacts,
            label_financial
        ])
    
    return np.array(features), np.array(labels)

def extract_features_from_app(app):
    """
    Extract features from a single app (for dropdown selection).
    Returns a dictionary with all 8 feature values.
    """
    app_info = app.get('appInfo', {})
    collected_data = app_info.get('collectedData', [])
    shared_data = app_info.get('sharedData', [])
    security_practices = app_info.get('securityPractices', [])
    
    # Extract output labels for reference
    collected_types = {item.get('type') for item in collected_data}
    
    # Extract input features (same logic as training)
    sensitive_types = {'Personal info', 'Location', 'Contacts', 'Financial info'}
    
    # Feature 1: shared_with_third_parties
    shared_sensitive = any(
        item.get('type') in sensitive_types 
        for item in shared_data
    )
    feature_shared = 1 if shared_sensitive else 0
    
    # Feature 2: used_for_advertising
    advertising_in_collected = any(
        'Advertising or marketing' in item.get('purpose', '')
        for item in collected_data
    )
    advertising_in_shared = any(
        'Advertising or marketing' in item.get('purpose', '')
        for item in shared_data
    )
    feature_advertising = 1 if (advertising_in_collected or advertising_in_shared) else 0
    
    # Feature 3: encryption_enabled
    feature_encryption = 1 if any('encrypt' in str(p).lower() for p in security_practices) else 0
    
    # Feature 4: data_deletion_allowed
    feature_deletion = 1 if any('delete' in str(p).lower() for p in security_practices) else 0
    
    # Feature 5: collects_sensitive_data
    feature_sensitive = 1 if bool(sensitive_types & collected_types) else 0
    
    # Feature 6: num_data_types_collected
    feature_num_types = len(collected_types)
    
    # Feature 7: collects_multiple_types
    feature_multiple = 1 if len(collected_types) > 2 else 0
    
    # Feature 8: num_data_types_shared
    feature_num_shared = len({item.get('type') for item in shared_data})
    
    return {
        'shared_with_third_parties': feature_shared,
        'used_for_advertising': feature_advertising,
        'encryption_enabled': feature_encryption,
        'data_deletion_allowed': feature_deletion,
        'collects_sensitive_data': feature_sensitive,
        'num_data_types_collected': feature_num_types,
        'collects_multiple_types': feature_multiple,
        'num_data_types_shared': feature_num_shared
    }

def train_models():
    """Train all 3 models and return accuracies"""
    global models, model_accuracies, X_test, Y_test, feature_names, label_names, dataset_apps
    
    print("\n[INFO] Loading data...")
    dataset = load_data('game_app_data_safety.json')
    dataset_apps = dataset  # Store for dropdown
    
    print("[INFO] Extracting features and labels...")
    X, Y = extract_features_and_labels_no_leakage(dataset)
    
    feature_names = [
        'shared_with_third_parties',
        'used_for_advertising',
        'encryption_enabled',
        'data_deletion_allowed',
        'collects_sensitive_data',
        'num_data_types_collected',
        'collects_multiple_types',
        'num_data_types_shared'
    ]
    
    label_names = [
        'Personal Info',
        'Location',
        'Contacts',
        'Financial Info'
    ]
    
    print("[INFO] Splitting data...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Train Decision Tree
    print("[INFO] Training Decision Tree...")
    dt_model = MultiOutputClassifier(
        DecisionTreeClassifier(max_depth=8, min_samples_split=4, random_state=42)
    )
    dt_model.fit(X_train, Y_train)
    dt_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(Y_test, dt_pred)
    models['Decision Tree'] = dt_model
    model_accuracies['Decision Tree'] = float(dt_acc)
    
    # Train Random Forest
    print("[INFO] Training Random Forest...")
    rf_model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            class_weight='balanced_subsample',
            random_state=42
        )
    )
    rf_model.fit(X_train, Y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(Y_test, rf_pred)
    models['Random Forest'] = rf_model
    model_accuracies['Random Forest'] = float(rf_acc)
    
    # Train Logistic Regression
    print("[INFO] Training Logistic Regression...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    lr_model = MultiOutputClassifier(lr_pipeline)
    lr_model.fit(X_train, Y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(Y_test, lr_pred)
    models['Logistic Regression'] = lr_model
    model_accuracies['Logistic Regression'] = float(lr_acc)
    
    print(f"✓ All models trained!")
    print(f"  Decision Tree: {dt_acc*100:.2f}%")
    print(f"  Random Forest: {rf_acc*100:.2f}%")
    print(f"  Logistic Regression: {lr_acc*100:.2f}%")

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the frontend"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    
    Expects JSON:
    {
        "shared_with_third_parties": 0/1,
        "used_for_advertising": 0/1,
        "encryption_enabled": 0/1,
        "data_deletion_allowed": 0/1,
        "collects_sensitive_data": 0/1,
        "num_data_types_collected": number,
        "collects_multiple_types": 0/1,
        "num_data_types_shared": number
    }
    """
    try:
        data = request.json
        
        # Create feature vector
        feature_vector = np.array([[
            data.get('shared_with_third_parties', 0),
            data.get('used_for_advertising', 0),
            data.get('encryption_enabled', 0),
            data.get('data_deletion_allowed', 0),
            data.get('collects_sensitive_data', 0),
            data.get('num_data_types_collected', 0),
            data.get('collects_multiple_types', 0),
            data.get('num_data_types_shared', 0)
        ]])
        
        # Get best model (Random Forest)
        best_model = models['Random Forest']
        prediction = best_model.predict(feature_vector)[0]
        
        # Format response
        results = {}
        for i, label in enumerate(label_names):
            results[label] = bool(prediction[i] == 1)
        
        return jsonify({
            'status': 'success',
            'predictions': results,
            'features': data
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get model comparison data"""
    try:
        return jsonify({
            'status': 'success',
            'models': model_accuracies,
            'best_model': max(model_accuracies, key=model_accuracies.get)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get system information"""
    return jsonify({
        'status': 'success',
        'title': 'App Data Collection Predictor',
        'description': 'Predicts what data an app collects using behavioral features',
        'features': feature_names,
        'labels': label_names,
        'best_model': max(model_accuracies, key=model_accuracies.get),
        'best_accuracy': max(model_accuracies.values())
    })

@app.route('/api/apps', methods=['GET'])
def get_apps():
    """Get list of all apps with their IDs"""
    try:
        apps_list = [{
            'appId': app.get('appId', 'Unknown'),
            'display_name': app.get('appId', 'Unknown').split('.')[-1]  # Show last part of package name
        } for app in dataset_apps]
        
        return jsonify({
            'status': 'success',
            'apps': apps_list,
            'count': len(apps_list)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/apps/<appId>', methods=['GET'])
def get_app_features(appId):
    """Extract and return features for a specific app"""
    try:
        # Find the app in dataset
        app = None
        for a in dataset_apps:
            if a.get('appId') == appId:
                app = a
                break
        
        if not app:
            return jsonify({
                'status': 'error',
                'message': 'App not found'
            }), 404
        
        # Extract features
        features = extract_features_from_app(app)
        
        return jsonify({
            'status': 'success',
            'appId': appId,
            'features': features,
            'display_name': appId.split('.')[-1]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# ============================================================================
# MAIN
# ============================================================================
# MAIN
# ============================================================================
# Train models when app starts (works with Gunicorn too)
train_models()
if __name__ == '__main__':
    print("\n" + "="*80)
    print("INITIALIZING APP DATA COLLECTION PREDICTOR")
    print("="*80)
    
    try:
        train_models()
        print("\n✓ Starting Flask server on http://localhost:5000")
        print("\nEndpoints:")
        print("  GET  http://localhost:5000/                 (Frontend)")
        print("  GET  http://localhost:5000/api/info         (System info)")
        print("  GET  http://localhost:5000/api/models       (Model comparison)")
        print("  POST http://localhost:5000/api/predict      (Make prediction)")
        print("\n" + "="*80 + "\n")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        raise
